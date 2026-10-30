use std::borrow::Cow;
use std::cell::{OnceCell, RefCell};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::quant::Quantization;
use crate::store::VectorRef;

#[cfg(target_arch = "aarch64")]
mod neon;
pub mod scalar;
#[cfg(target_arch = "x86_64")]
mod x86;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    /// reported distance is the squared euclidean distance
    L2,
    /// vectors are normalised on the way in; distance is 1 - cos
    Cosine,
    /// distance is the negated dot product, so smaller is still better
    InnerProduct,
}

#[derive(Clone, Copy)]
pub struct Kernels {
    pub name: &'static str,
    pub dot: fn(&[f32], &[f32]) -> f32,
    pub l2_sq: fn(&[f32], &[f32]) -> f32,
    /// sum(q[i] * codes[i]) with codes widened to f32
    pub dot_u8: fn(&[f32], &[u8]) -> f32,
}

impl Kernels {
    pub const SCALAR: Kernels =
        Kernels { name: "scalar", dot: scalar::dot, l2_sq: scalar::l2_sq, dot_u8: scalar::dot_u8 };

    /// every kernel set this CPU can run, scalar first
    pub fn available() -> Vec<Kernels> {
        let mut out = vec![Kernels::SCALAR];
        #[cfg(target_arch = "x86_64")]
        {
            out.extend(x86::avx2());
            out.extend(x86::avx512());
        }
        #[cfg(target_arch = "aarch64")]
        out.extend(neon::neon());
        out
    }

    fn detect() -> Kernels {
        let all = Kernels::available();
        if let Ok(want) = std::env::var("VECDB_SIMD") {
            if let Some(k) = all.iter().find(|k| k.name.starts_with(want.as_str())) {
                return *k;
            }
            tracing::warn!(%want, "VECDB_SIMD names an unavailable kernel, ignoring");
        }
        *all.last().unwrap()
    }
}

type HammingFn = fn(&[u32], &[u32]) -> u32;

/// differing bits between two packed vectors
#[inline]
pub fn hamming(a: &[u32], b: &[u32]) -> u32 {
    static F: OnceLock<HammingFn> = OnceLock::new();
    F.get_or_init(pick_hamming)(a, b)
}

fn pick_hamming() -> HammingFn {
    #[cfg(target_arch = "x86_64")]
    if kernels().name != "scalar" && is_x86_feature_detected!("popcnt") {
        return x86::hamming_popcnt;
    }
    // aarch64 lowers count_ones to NEON cnt on its own
    scalar::hamming
}

pub fn kernels() -> &'static Kernels {
    static K: OnceLock<Kernels> = OnceLock::new();
    K.get_or_init(Kernels::detect)
}

pub struct PreparedQuery<'a> {
    pub(crate) v: Cow<'a, [f32]>,
    sum: f32,
    norm_sq: f32,
    // pq only: per-query partial distances, built on the first comparison
    adc: OnceCell<Vec<f32>>,
    use_adc: bool,
    // binary only: the query's own sign bits
    bits: Vec<u32>,
}

thread_local! {
    static DECODED: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

impl PreparedQuery<'_> {
    pub fn as_slice(&self) -> &[f32] {
        &self.v
    }
}

#[derive(Clone, Copy)]
pub struct Space {
    pub metric: Metric,
    pub dim: usize,
    pub quant: Quantization,
    k: &'static Kernels,
}

impl Space {
    pub fn new(metric: Metric, dim: usize, quant: Quantization) -> Self {
        Space { metric, dim, quant, k: kernels() }
    }

    pub fn with_kernels(mut self, k: &'static Kernels) -> Self {
        self.k = k;
        self
    }

    pub fn kernel_name(&self) -> &'static str {
        self.k.name
    }

    /// wraps a slice that is already in index space (stored vectors, normalised input)
    pub fn prepare<'a>(&self, v: &'a [f32]) -> PreparedQuery<'a> {
        let (sum, norm_sq) = match self.quant {
            Quantization::None => (0.0, 0.0),
            Quantization::Sq8 => (v.iter().sum(), (self.k.dot)(v, v)),
            Quantization::Pq { .. } | Quantization::Binary => (0.0, 0.0),
        };
        let bits = self.query_bits(v);
        PreparedQuery { v: Cow::Borrowed(v), sum, norm_sq, adc: OnceCell::new(), use_adc: false, bits }
    }

    /// like `prepare`, but for a vector that is compared against many stored ones:
    /// with pq the lookup table pays for itself, decoding each neighbour does not
    pub fn prepare_indexed<'a>(&self, v: &'a [f32]) -> PreparedQuery<'a> {
        let mut q = self.prepare(v);
        q.use_adc = matches!(self.quant, Quantization::Pq { .. });
        q
    }

    /// user-supplied vector: checked, and normalised for cosine
    pub fn prepare_user(&self, v: &[f32]) -> Result<PreparedQuery<'static>> {
        let v = self.sanitize(v)?;
        let (sum, norm_sq) = match self.quant {
            Quantization::None => (0.0, 0.0),
            Quantization::Sq8 => (v.iter().sum(), (self.k.dot)(&v, &v)),
            Quantization::Pq { .. } | Quantization::Binary => (0.0, 0.0),
        };
        let use_adc = matches!(self.quant, Quantization::Pq { .. });
        let bits = self.query_bits(&v);
        Ok(PreparedQuery { v: Cow::Owned(v), sum, norm_sq, adc: OnceCell::new(), use_adc, bits })
    }

    fn query_bits(&self, v: &[f32]) -> Vec<u32> {
        if self.quant != Quantization::Binary {
            return Vec::new();
        }
        let mut bits = vec![0u32; self.quant.stride_words(self.dim)];
        crate::quant::binarize(v, &mut bits);
        bits
    }

    pub fn sanitize(&self, v: &[f32]) -> Result<Vec<f32>> {
        if v.len() != self.dim {
            return Err(Error::DimensionMismatch { expected: self.dim, got: v.len() });
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(Error::InvalidVector("contains NaN or infinity"));
        }
        let mut out = v.to_vec();
        if self.metric == Metric::Cosine {
            let n = (self.k.dot)(&out, &out).sqrt();
            if n == 0.0 {
                return Err(Error::InvalidVector("zero vector under cosine metric"));
            }
            let inv = 1.0 / n;
            out.iter_mut().for_each(|x| *x *= inv);
        }
        Ok(out)
    }

    #[inline]
    pub fn dist(&self, q: &PreparedQuery<'_>, v: VectorRef<'_>) -> f32 {
        match v {
            VectorRef::F32(x) => match self.metric {
                Metric::L2 => (self.k.l2_sq)(&q.v, x),
                Metric::InnerProduct => -(self.k.dot)(&q.v, x),
                Metric::Cosine => 1.0 - (self.k.dot)(&q.v, x),
            },
            VectorRef::Sq8 { codes, min, scale, norm_sq } => {
                // x_i = min + scale * c_i  =>  q.x = min * sum(q) + scale * q.c
                let dot = min * q.sum + scale * (self.k.dot_u8)(&q.v, &codes[..self.dim]);
                match self.metric {
                    Metric::L2 => (q.norm_sq + norm_sq - 2.0 * dot).max(0.0),
                    Metric::InnerProduct => -dot,
                    Metric::Cosine => 1.0 - dot,
                }
            }
            // same number whatever the metric: how many sign bits differ
            VectorRef::Binary { bits } => hamming(&q.bits, bits) as f32,
            VectorRef::Pq { codes, book } => {
                if q.use_adc {
                    let t = q.adc.get_or_init(|| book.adc_table(&q.v, self.metric));
                    let mut s = 0.0;
                    for (j, &c) in codes.iter().enumerate() {
                        s += t[j * crate::pq::K + c as usize];
                    }
                    return if self.metric == Metric::Cosine { 1.0 + s } else { s };
                }
                DECODED.with(|b| {
                    let mut b = b.borrow_mut();
                    b.resize(self.dim, 0.0);
                    book.decode(codes, &mut b);
                    match self.metric {
                        Metric::L2 => (self.k.l2_sq)(&q.v, &b),
                        Metric::InnerProduct => -(self.k.dot)(&q.v, &b),
                        Metric::Cosine => 1.0 - (self.k.dot)(&q.v, &b),
                    }
                })
            }
        }
    }
}

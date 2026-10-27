use std::borrow::Cow;
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

pub fn kernels() -> &'static Kernels {
    static K: OnceLock<Kernels> = OnceLock::new();
    K.get_or_init(Kernels::detect)
}

pub struct PreparedQuery<'a> {
    pub(crate) v: Cow<'a, [f32]>,
    sum: f32,
    norm_sq: f32,
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
        };
        PreparedQuery { v: Cow::Borrowed(v), sum, norm_sq }
    }

    /// user-supplied vector: checked, and normalised for cosine
    pub fn prepare_user(&self, v: &[f32]) -> Result<PreparedQuery<'static>> {
        let v = self.sanitize(v)?;
        let (sum, norm_sq) = match self.quant {
            Quantization::None => (0.0, 0.0),
            Quantization::Sq8 => (v.iter().sum(), (self.k.dot)(&v, &v)),
        };
        Ok(PreparedQuery { v: Cow::Owned(v), sum, norm_sq })
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
        }
    }
}

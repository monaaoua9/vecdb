use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Quantization {
    None,
    /// 8 bits per component, min/scale kept per vector
    Sq8,
    /// product quantization, one byte per `dim / m` components; needs a trained codebook
    Pq {
        m: u16,
    },
    /// one bit per component (its sign), distance is the Hamming distance. 32x smaller than
    /// f32; recall depends on `Config::rerank` and on the data being roughly zero-centred
    Binary,
}

/// min, scale, |decoded|^2 stored after the code bytes
const PARAM_WORDS: usize = 3;

impl Quantization {
    /// record size in 32-bit words
    pub fn stride_words(self, dim: usize) -> usize {
        match self {
            Quantization::None => dim,
            Quantization::Sq8 => code_words(dim) + PARAM_WORDS,
            Quantization::Pq { m } => (m as usize).div_ceil(4),
            Quantization::Binary => dim.div_ceil(32),
        }
    }

    pub fn bytes_per_vector(self, dim: usize) -> usize {
        self.stride_words(dim) * 4
    }
}

pub fn code_words(dim: usize) -> usize {
    dim.div_ceil(4)
}

/// sets bit i when `v[i] > 0`; `out` must be `dim.div_ceil(32)` words, the padding bits stay zero
pub fn binarize(v: &[f32], out: &mut [u32]) {
    out.fill(0);
    for (i, &x) in v.iter().enumerate() {
        if x > 0.0 {
            out[i / 32] |= 1 << (i % 32);
        }
    }
}

/// bits back to +1 / -1 components
pub fn unbinarize(bits: &[u32], out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = if bits[i / 32] >> (i % 32) & 1 == 1 { 1.0 } else { -1.0 };
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Sq8Params {
    pub min: f32,
    pub scale: f32,
    pub norm_sq: f32,
}

/// `codes` must be `code_words(v.len()) * 4` bytes, the padding is zeroed
pub fn encode(v: &[f32], codes: &mut [u8]) -> Sq8Params {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in v {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    let scale = (hi - lo) / 255.0;
    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };

    codes.fill(0);
    let mut norm_sq = 0.0f32;
    for (c, &x) in codes.iter_mut().zip(v) {
        let q = ((x - lo) * inv).round().clamp(0.0, 255.0) as u8;
        *c = q;
        let d = lo + scale * q as f32;
        norm_sq += d * d;
    }
    Sq8Params { min: lo, scale, norm_sq }
}

pub fn decode(codes: &[u8], p: &Sq8Params, out: &mut [f32]) {
    for (o, &c) in out.iter_mut().zip(codes) {
        *o = p.min + p.scale * c as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_error_is_bounded() {
        let v: Vec<f32> = (0..100).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut codes = vec![0u8; code_words(v.len()) * 4];
        let p = encode(&v, &mut codes);
        let mut back = vec![0.0; v.len()];
        decode(&codes, &p, &mut back);
        for (a, b) in v.iter().zip(&back) {
            assert!((a - b).abs() <= p.scale * 0.51 + 1e-6);
        }
    }

    #[test]
    fn constant_vector() {
        let v = [0.5f32; 7];
        let mut codes = vec![0u8; 8];
        let p = encode(&v, &mut codes);
        let mut back = [0.0; 7];
        decode(&codes, &p, &mut back);
        assert!(back.iter().all(|&x| x == 0.5));
    }

    #[test]
    fn binary_roundtrip_keeps_signs() {
        let v: Vec<f32> = (0..70).map(|i| if i % 3 == 0 { -0.5 } else { 0.25 }).collect();
        let mut bits = vec![0u32; Quantization::Binary.stride_words(70)];
        assert_eq!(bits.len(), 3);
        binarize(&v, &mut bits);
        let mut back = vec![0.0; 70];
        unbinarize(&bits, &mut back);
        for (a, b) in v.iter().zip(&back) {
            assert_eq!(a.signum(), *b);
        }
        // padding must stay clear or hamming distances would drift
        assert_eq!(bits[2] >> 6, 0);
    }

    #[test]
    fn memory_ratio() {
        let full = 4 * 768;
        let q = Quantization::Sq8.bytes_per_vector(768);
        assert!((q as f64) < full as f64 * 0.26);
    }
}

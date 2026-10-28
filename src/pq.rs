use crate::distance::Metric;
use crate::error::{Error, Result};

/// one byte per subspace
pub const K: usize = 256;

/// `m` subspaces of `dim / m` components, 256 centroids each
pub struct Codebook {
    dim: usize,
    m: usize,
    dsub: usize,
    cents: Vec<f32>,
}

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn nearest(cents: &[f32], v: &[f32], dsub: usize) -> usize {
    let mut best = (f32::INFINITY, 0);
    for (c, cen) in cents.chunks_exact(dsub).enumerate() {
        let mut d = 0.0;
        for (a, b) in cen.iter().zip(v) {
            let t = a - b;
            d += t * t;
        }
        if d < best.0 {
            best = (d, c);
        }
    }
    best.1
}

fn kmeans(data: &[f32], dim: usize, off: usize, dsub: usize, iters: usize, mut rng: u64, out: &mut [f32]) {
    let n = data.len() / dim;
    let sub: Vec<f32> =
        (0..n).flat_map(|i| data[i * dim + off..i * dim + off + dsub].iter().copied()).collect();

    // distinct random points as the starting centroids
    let mut idx: Vec<usize> = (0..n).collect();
    for c in 0..K {
        let r = c + splitmix(&mut rng) as usize % (n - c);
        idx.swap(c, r);
        out[c * dsub..(c + 1) * dsub].copy_from_slice(&sub[idx[c] * dsub..(idx[c] + 1) * dsub]);
    }

    let mut sums = vec![0f32; K * dsub];
    let mut counts = vec![0u32; K];
    for _ in 0..iters {
        sums.fill(0.0);
        counts.fill(0);
        for p in sub.chunks_exact(dsub) {
            let c = nearest(out, p, dsub);
            counts[c] += 1;
            for (s, x) in sums[c * dsub..(c + 1) * dsub].iter_mut().zip(p) {
                *s += x;
            }
        }
        for c in 0..K {
            let dst = &mut out[c * dsub..(c + 1) * dsub];
            if counts[c] == 0 {
                let r = splitmix(&mut rng) as usize % n;
                dst.copy_from_slice(&sub[r * dsub..(r + 1) * dsub]);
            } else {
                let inv = 1.0 / counts[c] as f32;
                for (d, s) in dst.iter_mut().zip(&sums[c * dsub..(c + 1) * dsub]) {
                    *d = s * inv;
                }
            }
        }
    }
}

impl Codebook {
    /// `data` is row-major, already in index space (normalised for cosine)
    pub fn train(
        dim: usize,
        m: usize,
        data: &[f32],
        iters: usize,
        seed: u64,
        threads: usize,
    ) -> Result<Codebook> {
        let bad = |s: &str| Err(Error::InvalidConfig(s.to_owned()));
        if m == 0 || !dim.is_multiple_of(m) {
            return bad("pq: dim must be a multiple of m");
        }
        if !data.len().is_multiple_of(dim) || data.len() / dim < K {
            return bad("pq: need at least 256 training vectors");
        }
        let dsub = dim / m;
        let mut cents = vec![0f32; m * K * dsub];
        let per = m.div_ceil(threads.clamp(1, m));
        std::thread::scope(|s| {
            for (ci, chunk) in cents.chunks_mut(per * K * dsub).enumerate() {
                s.spawn(move || {
                    for (k, out) in chunk.chunks_mut(K * dsub).enumerate() {
                        let j = ci * per + k;
                        kmeans(data, dim, j * dsub, dsub, iters, seed ^ ((j as u64 + 1) << 32), out);
                    }
                });
            }
        });
        Ok(Codebook { dim, m, dsub, cents })
    }

    pub fn from_parts(dim: usize, m: usize, cents: Vec<f32>) -> Result<Codebook> {
        if m == 0 || !dim.is_multiple_of(m) || cents.len() != m * K * (dim / m) {
            return Err(Error::Corrupt("pq codebook has the wrong shape".into()));
        }
        Ok(Codebook { dim, m, dsub: dim / m, cents })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(16 + self.cents.len() * 4 + 4);
        b.extend_from_slice(b"VECDBPQ1");
        b.extend_from_slice(&(self.dim as u32).to_le_bytes());
        b.extend_from_slice(&(self.m as u32).to_le_bytes());
        b.extend_from_slice(bytemuck::cast_slice(&self.cents));
        let crc = crc32fast::hash(&b);
        b.extend_from_slice(&crc.to_le_bytes());
        b
    }

    pub fn from_bytes(b: &[u8]) -> Result<Codebook> {
        let bad = || Error::Corrupt("pq codebook file is damaged".into());
        if b.len() < 20 || &b[..8] != b"VECDBPQ1" {
            return Err(bad());
        }
        let (body, crc) = b.split_at(b.len() - 4);
        if crc32fast::hash(body) != u32::from_le_bytes(crc.try_into().unwrap()) {
            return Err(bad());
        }
        let dim = u32::from_le_bytes(body[8..12].try_into().unwrap()) as usize;
        let m = u32::from_le_bytes(body[12..16].try_into().unwrap()) as usize;
        if !body[16..].len().is_multiple_of(4) {
            return Err(bad());
        }
        let cents: Vec<f32> = body[16..].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        Codebook::from_parts(dim, m, cents)
    }

    pub fn centroids(&self) -> &[f32] {
        &self.cents
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn encode(&self, v: &[f32], out: &mut [u8]) {
        let ds = self.dsub;
        for j in 0..self.m {
            let cents = &self.cents[j * K * ds..(j + 1) * K * ds];
            out[j] = nearest(cents, &v[j * ds..(j + 1) * ds], ds) as u8;
        }
    }

    pub fn decode(&self, codes: &[u8], out: &mut [f32]) {
        let ds = self.dsub;
        for j in 0..self.m {
            let c = codes[j] as usize;
            let src = &self.cents[(j * K + c) * ds..(j * K + c + 1) * ds];
            out[j * ds..(j + 1) * ds].copy_from_slice(src);
        }
    }

    /// m * 256 partial distances; summing one entry per subspace gives the
    /// distance (minus 1 for cosine) between `q` and the encoded vector
    pub fn adc_table(&self, q: &[f32], metric: Metric) -> Vec<f32> {
        let ds = self.dsub;
        let mut t = vec![0f32; self.m * K];
        for j in 0..self.m {
            let qs = &q[j * ds..(j + 1) * ds];
            for c in 0..K {
                let cen = &self.cents[(j * K + c) * ds..(j * K + c + 1) * ds];
                t[j * K + c] = match metric {
                    Metric::L2 => cen.iter().zip(qs).map(|(a, b)| (a - b) * (a - b)).sum(),
                    _ => -cen.iter().zip(qs).map(|(a, b)| a * b).sum::<f32>(),
                };
            }
        }
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, dim: usize) -> Vec<f32> {
        let mut s = 7u64;
        (0..n * dim).map(|_| (splitmix(&mut s) >> 40) as f32 / (1u64 << 24) as f32).collect()
    }

    #[test]
    fn roundtrip_beats_the_mean() {
        let (dim, m) = (16, 4);
        let d = data(2000, dim);
        let b = Codebook::train(dim, m, &d, 8, 1, 2).unwrap();
        let (mut err, mut var) = (0.0, 0.0);
        let mut codes = vec![0u8; m];
        let mut back = vec![0f32; dim];
        for v in d.chunks_exact(dim) {
            b.encode(v, &mut codes);
            b.decode(&codes, &mut back);
            err += v.iter().zip(&back).map(|(a, b)| (a - b).powi(2)).sum::<f32>();
            var += v.iter().map(|a| (a - 0.5).powi(2)).sum::<f32>();
        }
        assert!(err < var * 0.5, "err {err} var {var}");
    }

    #[test]
    fn adc_matches_decoded_distance() {
        let (dim, m) = (8, 2);
        let d = data(600, dim);
        let b = Codebook::train(dim, m, &d, 4, 3, 1).unwrap();
        let q = &d[..dim];
        let t = b.adc_table(q, Metric::L2);
        let mut codes = vec![0u8; m];
        let mut back = vec![0f32; dim];
        for v in d.chunks_exact(dim).skip(10).take(20) {
            b.encode(v, &mut codes);
            b.decode(&codes, &mut back);
            let exact: f32 = q.iter().zip(&back).map(|(a, b)| (a - b).powi(2)).sum();
            let adc: f32 = (0..m).map(|j| t[j * K + codes[j] as usize]).sum();
            assert!((exact - adc).abs() < 1e-4);
        }
    }

    #[test]
    fn bytes_roundtrip_and_detect_damage() {
        let b = Codebook::train(8, 2, &data(600, 8), 2, 1, 1).unwrap();
        let mut raw = b.to_bytes();
        let back = Codebook::from_bytes(&raw).unwrap();
        assert_eq!(back.centroids(), b.centroids());
        raw[40] ^= 1;
        assert!(Codebook::from_bytes(&raw).is_err());
    }

    #[test]
    fn too_few_vectors_is_an_error() {
        assert!(Codebook::train(8, 2, &data(100, 8), 2, 1, 1).is_err());
    }
}

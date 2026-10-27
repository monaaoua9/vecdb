#![allow(dead_code)]
use std::collections::HashSet;

use vecdb::{Hit, Index, SearchParams};

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    // sum of 4 uniforms, close enough to a bell curve
    pub fn gauss(&mut self) -> f32 {
        (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.7
    }
}

/// Low intrinsic dimension (10 latent factors mixed up into `dim`) plus cluster structure.
/// Real embeddings look like this; iid noise in every coordinate does not and is unfairly hard for any ANN index.
pub fn clustered(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    const LATENT: usize = 10;
    let mut r = Rng::new(0xABCD);
    let mix: Vec<Vec<f32>> =
        (0..dim).map(|_| (0..LATENT).map(|_| r.gauss() / (LATENT as f32).sqrt()).collect()).collect();
    let centres: Vec<Vec<f32>> = (0..16).map(|_| (0..LATENT).map(|_| r.gauss() * 2.0).collect()).collect();
    let mut r = Rng::new(seed);
    (0..n)
        .map(|_| {
            let c = &centres[(r.next_u64() % 16) as usize];
            let z: Vec<f32> = c.iter().map(|x| x + r.gauss() * 0.5).collect();
            mix.iter()
                .map(|row| row.iter().zip(&z).map(|(a, b)| a * b).sum::<f32>() + r.gauss() * 0.05)
                .collect()
        })
        .collect()
}

pub fn recall(index: &Index, queries: &[Vec<f32>], k: usize, ef: usize) -> f64 {
    let mut hit = 0usize;
    for q in queries {
        let truth: HashSet<u64> = index.exact_search(q, k).unwrap().iter().map(|h| h.id).collect();
        let got: Vec<Hit> = index.search_with(q, &SearchParams::new(k).ef(ef)).unwrap();
        hit += got.iter().filter(|h| truth.contains(&h.id)).count();
    }
    hit as f64 / (queries.len() * k) as f64
}

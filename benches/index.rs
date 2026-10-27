use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use vecdb::{Config, Index, Metric, Payload, Quantization, SearchParams};

const N: usize = 20_000;
const DIM: usize = 128;

struct Lcg(u64);

impl Lcg {
    fn gauss(&mut self) -> f32 {
        let mut s = 0.0;
        for _ in 0..4 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            s += (self.0 >> 40) as f32 / (1u64 << 24) as f32;
        }
        s - 2.0
    }
}

fn data(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut r = Lcg(seed);
    (0..n).map(|_| (0..DIM).map(|_| r.gauss()).collect()).collect()
}

fn build(quant: Quantization, vectors: &[Vec<f32>]) -> Index {
    let mut cfg = Config::new(DIM, Metric::Cosine);
    cfg.quantization = quant;
    cfg.max_elements = N + 10_000;
    let idx = Index::new(cfg).unwrap();
    for (i, v) in vectors.iter().enumerate() {
        idx.insert(i as u64, v, Payload::new()).unwrap();
    }
    idx
}

fn search(c: &mut Criterion) {
    let vectors = data(N, 1);
    let queries = data(256, 2);
    let mut g = c.benchmark_group("search_k10");
    for (name, quant) in [("f32", Quantization::None), ("sq8", Quantization::Sq8)] {
        let idx = build(quant, &vectors);
        for ef in [32usize, 64, 128] {
            let p = SearchParams::new(10).ef(ef);
            g.bench_with_input(BenchmarkId::new(name, ef), &ef, |b, _| {
                let mut i = 0;
                b.iter(|| {
                    i = (i + 1) % queries.len();
                    black_box(idx.search_with(&queries[i], &p).unwrap())
                })
            });
        }
    }
    g.finish();
}

fn insert(c: &mut Criterion) {
    let vectors = data(N, 1);
    let extra = data(1000, 3);
    let mut g = c.benchmark_group("insert");
    g.sample_size(10);
    g.bench_function("1000_into_20k", |b| {
        b.iter_batched(
            || build(Quantization::None, &vectors),
            |idx| {
                for (i, v) in extra.iter().enumerate() {
                    idx.insert((N + i) as u64, v, Payload::new()).unwrap();
                }
                idx
            },
            BatchSize::PerIteration,
        )
    });
    g.finish();
}

criterion_group!(benches, search, insert);
criterion_main!(benches);

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use vecdb::distance::Kernels;

fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            (x >> 8) as f32 / (1 << 24) as f32 - 0.5
        })
        .collect()
}

fn kernels(c: &mut Criterion) {
    for dim in [128usize, 768, 1536] {
        let a = fill(dim, 1);
        let b = fill(dim, 2);
        let codes: Vec<u8> = fill(dim, 3).iter().map(|x| ((x + 0.5) * 255.0) as u8).collect();

        let mut g = c.benchmark_group(format!("dim{dim}"));
        g.throughput(Throughput::Elements(dim as u64));
        for k in Kernels::available() {
            g.bench_with_input(BenchmarkId::new("dot", k.name), &k, |bn, k| {
                bn.iter(|| (k.dot)(black_box(&a), black_box(&b)))
            });
            g.bench_with_input(BenchmarkId::new("l2_sq", k.name), &k, |bn, k| {
                bn.iter(|| (k.l2_sq)(black_box(&a), black_box(&b)))
            });
            g.bench_with_input(BenchmarkId::new("dot_u8", k.name), &k, |bn, k| {
                bn.iter(|| (k.dot_u8)(black_box(&a), black_box(&codes)))
            });
        }
        g.finish();
    }
}

criterion_group!(benches, kernels);
criterion_main!(benches);

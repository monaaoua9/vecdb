//! Recall / latency sweep.
//!
//!   cargo run --release --example recall -- [n] [dim] [threads] [iid|latent]
//!
//! `iid` (default) is gaussian noise in every coordinate: the worst case for
//! graph indexes, there is no structure to exploit. `latent` mixes a few
//! hidden factors into `dim` coordinates, which is much closer to what real
//! embedding models produce.

use std::collections::HashSet;
use std::time::Instant;

use vecdb::{Config, Index, Item, Metric, Payload, Quantization, SearchParams};

struct Rng(u64);

impl Rng {
    fn gauss(&mut self) -> f32 {
        let mut s = 0.0;
        for _ in 0..4 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            s += (self.0 >> 40) as f32 / (1u64 << 24) as f32;
        }
        (s - 2.0) * 1.7
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let dim: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let threads: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));

    let latent = args.next().is_some_and(|s| s == "latent");

    let mut rng = Rng(12345);
    const FACTORS: usize = 12;
    let mix: Vec<Vec<f32>> = (0..dim).map(|_| (0..FACTORS).map(|_| rng.gauss()).collect()).collect();
    let make = |rng: &mut Rng| -> Vec<f32> {
        if latent {
            let z: Vec<f32> = (0..FACTORS).map(|_| rng.gauss()).collect();
            mix.iter()
                .map(|row| row.iter().zip(&z).map(|(a, b)| a * b).sum::<f32>() + rng.gauss() * 0.1)
                .collect()
        } else {
            (0..dim).map(|_| rng.gauss()).collect()
        }
    };
    let items: Vec<Item> =
        (0..n).map(|i| Item { id: i as u64, vector: make(&mut rng), payload: Payload::new() }).collect();
    let queries: Vec<Vec<f32>> = (0..200).map(|_| make(&mut rng)).collect();

    for quant in [Quantization::None, Quantization::Sq8] {
        let mut cfg = Config::new(dim, Metric::Cosine);
        cfg.quantization = quant;
        cfg.max_elements = n;
        let idx = Index::new(cfg).unwrap();

        let t = Instant::now();
        idx.insert_bulk(&items, threads).unwrap();
        let secs = t.elapsed().as_secs_f64();
        let s = idx.stats();
        println!(
            "\n{quant:?}: kernel={} n={n} dim={dim} build={secs:.1}s ({:.0} inserts/s on {threads} threads) vectors={:.1} MiB",
            s.kernel,
            n as f64 / secs,
            s.heap_vector_bytes as f64 / (1 << 20) as f64,
        );

        let truth: Vec<HashSet<u64>> = queries
            .iter()
            .map(|q| idx.exact_search(q, 10).unwrap().into_iter().map(|h| h.id).collect())
            .collect();

        println!("{:>6} {:>10} {:>10} {:>10}", "ef", "recall@10", "p50 us", "QPS");
        for ef in [16, 32, 64, 128, 256] {
            let p = SearchParams::new(10).ef(ef);
            let mut lat = Vec::with_capacity(queries.len());
            let mut hit = 0;
            for (q, t) in queries.iter().zip(&truth) {
                let t0 = Instant::now();
                let r = idx.search_with(q, &p).unwrap();
                lat.push(t0.elapsed().as_secs_f64() * 1e6);
                hit += r.iter().filter(|h| t.contains(&h.id)).count();
            }
            lat.sort_by(|a, b| a.total_cmp(b));
            let total: f64 = lat.iter().sum::<f64>() / 1e6;
            println!(
                "{ef:>6} {:>10.4} {:>10.0} {:>10.0}",
                hit as f64 / (queries.len() * 10) as f64,
                lat[lat.len() / 2],
                queries.len() as f64 / total,
            );
        }
    }
}

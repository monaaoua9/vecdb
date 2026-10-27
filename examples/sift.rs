//! SIFT1M (http://corpus-texmex.irisa.fr/) recall / latency sweep.
//!
//!   cargo run --release --example sift -- <dir with sift_base.fvecs, sift_query.fvecs,
//!                                          sift_groundtruth.ivecs> [threads]
//!
//! Same parameters and protocol as benches/compare/hnswlib_sift.cpp (L2, M=16,
//! ef_construction=200, k=10, single-threaded queries) so the two tables can be
//! compared side by side on one machine.

use std::fs;
use std::path::Path;
use std::time::Instant;

use vecdb::{Config, Index, Item, Metric, Payload, SearchParams};

fn read_vecs(path: &Path) -> (usize, usize, Vec<u32>) {
    let raw = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let dim = u32::from_le_bytes(raw[..4].try_into().unwrap()) as usize;
    let rec = 4 + 4 * dim;
    let n = raw.len() / rec;
    let mut out = Vec::with_capacity(n * dim);
    for chunk in raw.chunks_exact(rec) {
        out.extend(chunk[4..].chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())));
    }
    (n, dim, out)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: sift <dir> [threads]");
    let threads: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    let dir = Path::new(&dir);

    let (n, dim, base) = read_vecs(&dir.join("sift_base.fvecs"));
    let (nq, _, queries) = read_vecs(&dir.join("sift_query.fvecs"));
    let (_, kg, gt) = read_vecs(&dir.join("sift_groundtruth.ivecs"));
    let f = |bits: &[u32]| -> Vec<f32> { bits.iter().map(|b| f32::from_bits(*b)).collect() };

    let mut cfg = Config::new(dim, Metric::L2);
    cfg.max_elements = n;
    cfg.m = 16;
    cfg.ef_construction = 200;
    let idx = Index::new(cfg).unwrap();

    let t0 = Instant::now();
    for start in (0..n).step_by(50_000) {
        let end = (start + 50_000).min(n);
        let items: Vec<Item> = (start..end)
            .map(|i| Item { id: i as u64, vector: f(&base[i * dim..(i + 1) * dim]), payload: Payload::new() })
            .collect();
        idx.insert_bulk(&items, threads).unwrap();
    }
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "vecdb ({}): n={n} dim={dim} build={secs:.1}s ({:.0} inserts/s on {threads} threads)",
        idx.stats().kernel,
        n as f64 / secs
    );

    println!("{:>6} {:>10} {:>10} {:>10}", "ef", "recall@10", "p50 us", "QPS");
    let qf: Vec<Vec<f32>> = (0..nq).map(|q| f(&queries[q * dim..(q + 1) * dim])).collect();
    for ef in [16, 32, 64, 128, 256, 512] {
        let p = SearchParams::new(10).ef(ef);
        let mut lat = Vec::with_capacity(nq);
        let mut hit = 0;
        for (q, v) in qf.iter().enumerate() {
            let t = Instant::now();
            let r = idx.search_with(v, &p).unwrap();
            lat.push(t.elapsed().as_secs_f64() * 1e6);
            let truth = &gt[q * kg..q * kg + 10];
            hit += r.iter().filter(|h| truth.contains(&(h.id as u32))).count();
        }
        lat.sort_by(|a, b| a.total_cmp(b));
        let total: f64 = lat.iter().sum::<f64>() / 1e6;
        println!(
            "{ef:>6} {:>10.4} {:>10.0} {:>10.0}",
            hit as f64 / (nq * 10) as f64,
            lat[lat.len() / 2],
            nq as f64 / total
        );
    }
}

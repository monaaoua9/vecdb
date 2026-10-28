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
        out.extend(chunk[4..].as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)));
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

    // SIFT_SNAPSHOT=<file>: load it if present, otherwise build and save it there,
    // so repeated runs skip the 4-minute build
    let cache = std::env::var_os("SIFT_SNAPSHOT").map(std::path::PathBuf::from);
    let idx = match cache.as_ref().filter(|p| p.exists()) {
        Some(p) => {
            let t = Instant::now();
            let idx = Index::open(p).unwrap();
            println!(
                "vecdb ({}): loaded {} vectors in {:.1}s",
                idx.stats().kernel,
                idx.len(),
                t.elapsed().as_secs_f64()
            );
            idx
        }
        None => {
            let mut cfg = Config::new(dim, Metric::L2);
            cfg.max_elements = n;
            let quant = std::env::var("SIFT_QUANT").unwrap_or_default();
            if quant == "sq8" {
                cfg.quantization = vecdb::Quantization::Sq8;
            }
            let pq_m = quant.strip_prefix("pq").and_then(|m| m.parse::<u16>().ok());
            if let Some(m) = pq_m {
                cfg.quantization = vecdb::Quantization::Pq { m };
            }
            cfg.rerank = std::env::var_os("SIFT_RERANK").is_some();
            cfg.m = 16;
            cfg.ef_construction = 200;
            let idx = match pq_m {
                Some(_) => {
                    // every 20th vector, 50k rows
                    let sample: Vec<f32> =
                        (0..n).step_by(20).flat_map(|i| f(&base[i * dim..(i + 1) * dim])).collect();
                    let t = Instant::now();
                    let idx = Index::new_pq(cfg, &sample, threads).unwrap();
                    println!("pq codebook trained in {:.1}s", t.elapsed().as_secs_f64());
                    idx
                }
                None => Index::new(cfg).unwrap(),
            };

            let t0 = Instant::now();
            for start in (0..n).step_by(50_000) {
                let end = (start + 50_000).min(n);
                let items: Vec<Item> = (start..end)
                    .map(|i| Item {
                        id: i as u64,
                        vector: f(&base[i * dim..(i + 1) * dim]),
                        payload: Payload::new(),
                    })
                    .collect();
                idx.insert_bulk(&items, threads).unwrap();
            }
            let secs = t0.elapsed().as_secs_f64();
            println!(
                "vecdb ({}): n={n} dim={dim} build={secs:.1}s ({:.0} inserts/s on {threads} threads)",
                idx.stats().kernel,
                n as f64 / secs
            );
            if let Some(p) = &cache {
                idx.save(p).unwrap();
            }
            idx
        }
    };

    println!("{:>6} {:>10} {:>10} {:>10}", "ef", "recall@10", "p50 us", "QPS");
    let qf: Vec<Vec<f32>> = (0..nq).map(|q| f(&queries[q * dim..(q + 1) * dim])).collect();
    // laptop clocks wander, so each ef runs SIFT_PASSES times (default 3) and the fastest pass is reported
    let passes: usize = std::env::var("SIFT_PASSES").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let only: Option<usize> = std::env::var("SIFT_EF").ok().and_then(|s| s.parse().ok());
    for ef in [16, 32, 64, 128, 256, 512].into_iter().filter(|e| only.is_none_or(|o| o == *e)) {
        let p = SearchParams::new(10).ef(ef);
        let mut best: Option<(f64, f64, f64)> = None; // (qps, p50, recall)
        for _ in 0..passes {
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
            let cur = (nq as f64 / total, lat[lat.len() / 2], hit as f64 / (nq * 10) as f64);
            if best.is_none_or(|b| cur.0 > b.0) {
                best = Some(cur);
            }
        }
        let (qps, p50, rec) = best.unwrap();
        println!("{ef:>6} {rec:>10.4} {p50:>10.0} {qps:>10.0}");
    }
}

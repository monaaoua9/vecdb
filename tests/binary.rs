mod common;

use std::collections::HashSet;

use common::clustered;
use vecdb::{Config, Index, Metric, Payload, Quantization, SearchParams};

fn config(rerank: bool, n: usize, dim: usize, quant: Quantization) -> Config {
    let mut cfg = Config::new(dim, Metric::Cosine);
    cfg.quantization = quant;
    cfg.rerank = rerank;
    cfg.max_elements = n + 10;
    cfg.ef_construction = 120;
    cfg
}

fn fill(cfg: Config, data: &[Vec<f32>]) -> Index {
    let idx = Index::new(cfg).unwrap();
    for (i, v) in data.iter().enumerate() {
        idx.insert(i as u64, v, Payload::new()).unwrap();
    }
    idx
}

// ground truth comes from an unquantized index, not from the index under test
fn recall_vs_f32(idx: &Index, truth: &Index, qs: &[Vec<f32>], ef: usize) -> f64 {
    let mut hit = 0;
    for q in qs {
        let want: HashSet<u64> = truth.exact_search(q, 10).unwrap().iter().map(|h| h.id).collect();
        let got = idx.search_with(q, &SearchParams::new(10).ef(ef)).unwrap();
        hit += got.iter().filter(|h| want.contains(&h.id)).count();
    }
    hit as f64 / (qs.len() * 10) as f64
}

#[test]
fn rerank_recovers_recall() {
    let (n, dim) = (4000, 128);
    let data = clustered(n, dim, 31);
    let qs = clustered(100, dim, 777);
    let truth = fill(config(false, n, dim, Quantization::None), &data);
    let plain = fill(config(false, n, dim, Quantization::Binary), &data);
    let reranked = fill(config(true, n, dim, Quantization::Binary), &data);

    let (p, r) = (recall_vs_f32(&plain, &truth, &qs, 128), recall_vs_f32(&reranked, &truth, &qs, 128));
    println!("binary recall@10 = {p:.4}, with rerank {r:.4}");
    assert!(p > 0.2, "plain recall {p}");
    assert!(r >= 0.9, "rerank recall {r}");
    assert!(r > p + 0.2);
}

#[test]
fn vectors_take_a_thirty_second_of_the_space() {
    let f = Quantization::None.bytes_per_vector(768);
    let b = Quantization::Binary.bytes_per_vector(768);
    assert_eq!(f / b, 32);
}

#[test]
fn survives_snapshot_roundtrip() {
    let data = clustered(1500, 64, 31);
    let qs = clustered(30, 64, 777);
    let idx = fill(config(true, 1500, 64, Quantization::Binary), &data);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("snapshot-000000000001.vdb");
    idx.save(&path).unwrap();
    let back = Index::open(&path).unwrap();
    for q in &qs[..20] {
        let a: Vec<u64> = idx.search(q, 5).unwrap().iter().map(|h| h.id).collect();
        let b: Vec<u64> = back.search(q, 5).unwrap().iter().map(|h| h.id).collect();
        assert_eq!(a, b);
    }
}

#[test]
fn hamming_distance_is_reported_without_rerank() {
    let mut cfg = Config::new(64, Metric::Cosine);
    cfg.quantization = Quantization::Binary;
    cfg.max_elements = 10;
    let idx = Index::new(cfg).unwrap();
    let mut v = vec![1.0f32; 64];
    idx.insert(1, &v, Payload::new()).unwrap();
    for x in v.iter_mut().take(5) {
        *x = -1.0;
    }
    idx.insert(2, &v, Payload::new()).unwrap();
    let hits = idx.search(&vec![1.0; 64], 2).unwrap();
    assert_eq!((hits[0].id, hits[0].distance), (1, 0.0));
    assert_eq!((hits[1].id, hits[1].distance), (2, 5.0));
}

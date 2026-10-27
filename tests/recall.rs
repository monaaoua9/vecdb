mod common;

use common::{clustered, recall};
use vecdb::{Config, Index, Metric, Payload, Quantization};

fn build(metric: Metric, quant: Quantization, n: usize, dim: usize) -> (Index, Vec<Vec<f32>>) {
    let data = clustered(n, dim, 42);
    let mut cfg = Config::new(dim, metric);
    cfg.quantization = quant;
    cfg.max_elements = n + 100;
    cfg.ef_construction = 120;
    let idx = Index::new(cfg).unwrap();
    for (i, v) in data.iter().enumerate() {
        idx.insert(i as u64, v, Payload::new()).unwrap();
    }
    let queries = clustered(100, dim, 4242);
    (idx, queries)
}

#[test]
fn l2_recall() {
    let (idx, qs) = build(Metric::L2, Quantization::None, 4000, 48);
    let r = recall(&idx, &qs, 10, 64);
    println!("l2 recall@10 = {r:.4}");
    assert!(r >= 0.97, "recall {r}");
}

#[test]
fn cosine_recall() {
    let (idx, qs) = build(Metric::Cosine, Quantization::None, 4000, 48);
    let r = recall(&idx, &qs, 10, 64);
    println!("cosine recall@10 = {r:.4}");
    assert!(r >= 0.97, "recall {r}");
}

#[test]
fn inner_product_recall() {
    let (idx, qs) = build(Metric::InnerProduct, Quantization::None, 4000, 48);
    let r = recall(&idx, &qs, 10, 128);
    println!("ip recall@10 = {r:.4}");
    assert!(r >= 0.9, "recall {r}");
}

#[test]
fn sq8_recall_and_memory() {
    let (idx, qs) = build(Metric::L2, Quantization::Sq8, 4000, 64);
    let r = recall(&idx, &qs, 10, 96);
    println!("sq8 recall@10 = {r:.4}");
    assert!(r >= 0.9, "recall {r}");

    let full = Quantization::None.bytes_per_vector(64) as f64;
    let q = Quantization::Sq8.bytes_per_vector(64) as f64;
    println!("sq8 bytes/vector: {q} vs {full} ({:.0}% smaller)", (1.0 - q / full) * 100.0);
    assert!(q / full < 0.4);
}

#[test]
fn recall_improves_with_ef() {
    let (idx, qs) = build(Metric::L2, Quantization::None, 3000, 32);
    let lo = recall(&idx, &qs, 10, 10);
    let hi = recall(&idx, &qs, 10, 200);
    assert!(hi >= lo);
    assert!(hi >= 0.98);
}

#[test]
fn self_query_returns_itself() {
    let (idx, _) = build(Metric::L2, Quantization::None, 1000, 16);
    let v = idx.get(123).unwrap().vector;
    let hits = idx.search(&v, 1).unwrap();
    assert_eq!(hits[0].id, 123);
    assert!(hits[0].distance < 1e-6);
}

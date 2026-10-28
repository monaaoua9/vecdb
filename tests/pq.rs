mod common;

use common::clustered;
use std::collections::HashSet;
use vecdb::{Config, Index, Item, Metric, Payload, Quantization, SearchParams};

const DIM: usize = 32;

fn cfg(metric: Metric) -> Config {
    let mut c = Config::new(DIM, metric);
    c.quantization = Quantization::Pq { m: 8 };
    c.max_elements = 5000;
    c
}

fn flat(data: &[Vec<f32>]) -> Vec<f32> {
    data.iter().flatten().copied().collect()
}

fn build(metric: Metric, data: &[Vec<f32>]) -> Index {
    let idx = Index::new_pq(cfg(metric), &flat(data), 2).unwrap();
    let items: Vec<Item> = data
        .iter()
        .enumerate()
        .map(|(i, v)| Item { id: i as u64, vector: v.clone(), payload: Payload::new() })
        .collect();
    idx.insert_bulk(&items, 4).unwrap();
    idx
}

fn truth(data: &[Vec<f32>], q: &[f32], k: usize) -> HashSet<u64> {
    let mut d: Vec<(f32, u64)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum(), i as u64))
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0));
    d.iter().take(k).map(|x| x.1).collect()
}

fn recall(idx: &Index, data: &[Vec<f32>], qs: &[Vec<f32>]) -> f64 {
    let mut hit = 0;
    for q in qs {
        let t = truth(data, q, 10);
        let got = idx.search_with(q, &SearchParams::new(10).ef(128)).unwrap();
        hit += got.iter().filter(|h| t.contains(&h.id)).count();
    }
    hit as f64 / (qs.len() * 10) as f64
}

#[test]
fn recall_is_usable() {
    let data = clustered(3000, DIM, 1);
    let qs = clustered(60, DIM, 99);
    let idx = build(Metric::L2, &data);
    let r = recall(&idx, &data, &qs);
    assert!(r > 0.6, "recall {r}");
}

#[test]
fn snapshot_roundtrip_keeps_codebook() {
    let data = clustered(1500, DIM, 2);
    let qs = clustered(20, DIM, 98);
    let idx = build(Metric::L2, &data);
    let before: Vec<_> = qs.iter().map(|q| idx.search(q, 5).unwrap()).collect();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pq.vdb");
    idx.save(&path).unwrap();
    let back = Index::open(&path).unwrap();
    assert_eq!(back.len(), 1500);
    for (q, b) in qs.iter().zip(&before) {
        assert_eq!(&back.search(q, 5).unwrap(), b);
    }
}

#[test]
fn compaction_keeps_working() {
    let data = clustered(1200, DIM, 3);
    let idx = build(Metric::L2, &data);
    for i in 0..300 {
        idx.delete(i).unwrap();
    }
    let c = idx.compacted(2).unwrap();
    assert_eq!(c.len(), 900);
    let hits = c.search(&data[700], 1).unwrap();
    assert_eq!(hits[0].id, 700);
}

#[test]
fn cosine_finds_itself() {
    let data = clustered(1500, DIM, 4);
    let idx = build(Metric::Cosine, &data);
    let hits = idx.search(&data[10], 3).unwrap();
    assert!(hits.iter().any(|h| h.id == 10));
}

#[test]
fn inner_product_orders_by_score() {
    let data = clustered(1500, DIM, 4);
    let idx = build(Metric::InnerProduct, &data);
    let q = &data[10];
    let hits = idx.search(q, 5).unwrap();
    assert_eq!(hits.len(), 5);
    assert!(hits.windows(2).all(|w| w[0].distance <= w[1].distance));
    // best hit should score close to the true best dot product
    let best = data.iter().map(|v| v.iter().zip(q).map(|(a, b)| a * b).sum::<f32>()).fold(f32::MIN, f32::max);
    assert!(-hits[0].distance > best * 0.9, "{} vs {best}", -hits[0].distance);
}

#[test]
fn plain_new_refuses_pq() {
    assert!(Index::new(cfg(Metric::L2)).is_err());
}

#[test]
fn records_are_about_a_thirty_second() {
    let data = clustered(1000, DIM, 5);
    let idx = build(Metric::L2, &data);
    let s = idx.stats();
    // 8 code bytes per vector against 128 for f32
    assert_eq!(s.heap_vector_bytes, 1024 * 8);
}

fn rerank_cfg() -> Config {
    let mut c = cfg(Metric::L2);
    c.rerank = true;
    c
}

fn build_rerank(data: &[Vec<f32>]) -> Index {
    let idx = Index::new_pq(rerank_cfg(), &flat(data), 2).unwrap();
    let items: Vec<Item> = data
        .iter()
        .enumerate()
        .map(|(i, v)| Item { id: i as u64, vector: v.clone(), payload: Payload::new() })
        .collect();
    idx.insert_bulk(&items, 4).unwrap();
    idx
}

#[test]
fn rerank_lifts_recall_and_gives_exact_distances() {
    let data = clustered(3000, DIM, 11);
    let qs = clustered(60, DIM, 97);
    let plain = recall(&build(Metric::L2, &data), &data, &qs);
    let idx = build_rerank(&data);
    let r = recall(&idx, &data, &qs);
    assert!(r > plain + 0.05, "rerank {r} vs plain {plain}");

    // the reported distance is the true one, not the lookup-table estimate
    let h = idx.search(&data[5], 1).unwrap();
    assert_eq!(h[0].id, 5);
    assert!(h[0].distance < 1e-6);
    assert_eq!(idx.get(5).unwrap().vector, data[5]);
}

#[test]
fn rerank_survives_snapshot_and_compaction() {
    let data = clustered(1500, DIM, 12);
    let idx = build_rerank(&data);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rr.vdb");
    idx.save(&path).unwrap();
    let back = Index::open(&path).unwrap();
    assert_eq!(back.get(700).unwrap().vector, data[700]);
    assert_eq!(back.search(&data[700], 1).unwrap()[0].id, 700);

    for i in 0..200 {
        back.delete(i).unwrap();
    }
    let c = back.compacted(2).unwrap();
    assert_eq!(c.get(900).unwrap().vector, data[900]);
}

#[test]
fn rerank_needs_quantization() {
    let mut c = Config::new(DIM, Metric::L2);
    c.rerank = true;
    assert!(Index::new(c).is_err());
}

mod common;

use std::collections::HashSet;

use common::clustered;
use vecdb::{Config, Filter, FilterStrategy, Index, Metric, Payload, SearchParams, Value};

const N: usize = 4000;

fn build() -> (Index, Vec<Vec<f32>>) {
    let dim = 24;
    let data = clustered(N, dim, 9);
    let mut cfg = Config::new(dim, Metric::L2);
    cfg.max_elements = N;
    cfg.brute_force_below = 300;
    let idx = Index::new(cfg).unwrap();
    for (i, v) in data.iter().enumerate() {
        let mut p = Payload::new();
        p.insert("bucket".into(), Value::Int((i % 10) as i64));
        p.insert("tag".into(), Value::Str(if i % 100 == 0 { "rare" } else { "common" }.into()));
        p.insert("score".into(), Value::Float(i as f64 / N as f64));
        idx.insert(i as u64, v, p).unwrap();
    }
    (idx, clustered(30, dim, 99))
}

fn truth(idx: &Index, q: &[f32], f: &Filter, k: usize) -> HashSet<u64> {
    idx.exact_search(q, N)
        .unwrap()
        .into_iter()
        .filter(|h| f.matches(&idx.get(h.id).unwrap().payload))
        .take(k)
        .map(|h| h.id)
        .collect()
}

fn check(idx: &Index, qs: &[Vec<f32>], f: Filter, strat: FilterStrategy, min_recall: f64) {
    let k = 10;
    let mut got = 0;
    let mut want = 0;
    for q in qs {
        let t = truth(idx, q, &f, k);
        let hits =
            idx.search_with(q, &SearchParams::new(k).ef(100).filter(f.clone()).strategy(strat)).unwrap();
        for h in &hits {
            assert!(f.matches(&idx.get(h.id).unwrap().payload), "filter violated by {}", h.id);
        }
        got += hits.iter().filter(|h| t.contains(&h.id)).count();
        want += t.len();
    }
    let r = got as f64 / want as f64;
    println!("{strat:?} recall = {r:.3}");
    assert!(r >= min_recall, "{strat:?} recall {r}");
}

#[test]
fn selective_filter_is_exact_via_prefilter() {
    let (idx, qs) = build();
    let f = Filter::eq("tag", "rare"); // 40 docs, planner scans them exactly
    check(&idx, &qs, f.clone(), FilterStrategy::Auto, 1.0);
    check(&idx, &qs, f, FilterStrategy::PreFilter, 1.0);
}

#[test]
fn broad_filter_uses_graph_walk() {
    let (idx, qs) = build();
    check(&idx, &qs, Filter::eq("bucket", 3), FilterStrategy::Auto, 0.95);
    check(&idx, &qs, Filter::eq("bucket", 3), FilterStrategy::PostFilter, 0.9);
}

#[test]
fn non_indexable_range_filter() {
    let (idx, qs) = build();
    let f = Filter::range("score", Some(0.2), Some(0.5));
    check(&idx, &qs, f.clone(), FilterStrategy::Auto, 0.95);
    check(&idx, &qs, f, FilterStrategy::PreFilter, 1.0);
}

#[test]
fn compound_filters() {
    let (idx, qs) = build();
    let f = Filter::And(vec![
        Filter::AnyOf("bucket".into(), vec![Value::Int(1), Value::Int(2)]),
        Filter::range("score", None, Some(0.6)),
    ]);
    check(&idx, &qs, f, FilterStrategy::Auto, 0.95);
    let g = Filter::Not(Box::new(Filter::eq("tag", "common")));
    check(&idx, &qs, g, FilterStrategy::Auto, 1.0);
}

#[test]
fn filter_matching_nothing() {
    let (idx, qs) = build();
    let hits = idx.search_with(&qs[0], &SearchParams::new(5).filter(Filter::eq("tag", "nope"))).unwrap();
    assert!(hits.is_empty());
}

#[test]
fn deleted_vectors_never_surface() {
    let (idx, qs) = build();
    for i in (0..N).step_by(2) {
        idx.delete(i as u64).unwrap();
    }
    assert_eq!(idx.len(), N / 2);
    for q in &qs {
        for h in idx.search(q, 20).unwrap() {
            assert!(h.id % 2 == 1, "deleted id {} returned", h.id);
        }
    }
    let hits = idx.search_with(&qs[0], &SearchParams::new(50).filter(Filter::eq("tag", "rare"))).unwrap();
    assert!(hits.iter().all(|h| h.id % 2 == 1));
}

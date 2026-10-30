//! Property check for the filter planner: whatever shape the filter has, every access path
//! (exact pre-filter scan, count, scroll) must agree with brute-force `Filter::matches`.
// is_multiple_of needs a newer rust than our msrv
#![allow(clippy::manual_is_multiple_of)]

mod common;

use std::collections::BTreeSet;

use common::Rng;
use vecdb::{Config, Filter, FilterStrategy, Index, Metric, Payload, SearchParams, Value};

const N: u64 = 300;

fn leaf(r: &mut Rng) -> Filter {
    let field = ["a", "b", "c"][(r.next_u64() % 3) as usize];
    match r.next_u64() % 5 {
        0 => Filter::eq(field, (r.next_u64() % 4) as i64),
        1 => Filter::eq(field, format!("s{}", r.next_u64() % 3).as_str()),
        2 => Filter::AnyOf(field.into(), vec![Value::Int(1), Value::Int(2), Value::Str("s0".into())]),
        3 => Filter::range(field, Some((r.next_u64() % 3) as f64), Some(1.0 + (r.next_u64() % 3) as f64)),
        _ => Filter::eq(field, r.next_u64() % 2 == 0),
    }
}

fn tree(r: &mut Rng, depth: usize) -> Filter {
    if depth == 0 || r.next_u64() % 3 == 0 {
        return leaf(r);
    }
    match r.next_u64() % 3 {
        0 => Filter::And((0..1 + r.next_u64() % 3).map(|_| tree(r, depth - 1)).collect()),
        1 => Filter::Or((0..1 + r.next_u64() % 3).map(|_| tree(r, depth - 1)).collect()),
        _ => Filter::Not(Box::new(tree(r, depth - 1))),
    }
}

fn payload(r: &mut Rng) -> Payload {
    let mut p = Payload::new();
    for key in ["a", "b", "c"] {
        match r.next_u64() % 4 {
            0 => {}
            1 => drop(p.insert(key.into(), Value::Int((r.next_u64() % 4) as i64))),
            2 => drop(p.insert(key.into(), Value::Str(format!("s{}", r.next_u64() % 3)))),
            _ => drop(p.insert(key.into(), Value::Bool(r.next_u64() % 2 == 0))),
        }
    }
    p
}

#[test]
fn every_path_agrees_with_brute_force() {
    let mut r = Rng::new(2024);
    let mut cfg = Config::new(2, Metric::L2);
    cfg.max_elements = 1000;
    let idx = Index::new(cfg).unwrap();
    for i in 0..N {
        idx.insert(i, &[i as f32, (i % 7) as f32], payload(&mut r)).unwrap();
    }
    // churn so the inverted index has seen removals and rewrites
    for i in (0..N).step_by(5) {
        idx.delete(i).unwrap();
    }
    for i in (1..N).step_by(9) {
        if idx.contains(i) {
            let p = payload(&mut r);
            if i % 2 == 0 { idx.set_payload(i, p) } else { idx.merge_payload(i, p) }.unwrap();
        }
    }

    for round in 0..150 {
        let f = tree(&mut r, 3);
        let want: BTreeSet<u64> = (0..N)
            .filter_map(|i| idx.get(i))
            .filter(|rec| f.matches(&rec.payload))
            .map(|rec| rec.id)
            .collect();

        assert_eq!(idx.count(Some(&f)), want.len(), "count, round {round}: {f:?}");

        let scrolled: BTreeSet<u64> =
            idx.scroll(Some(&f), 0, 10_000, false).items.iter().map(|rec| rec.id).collect();
        assert_eq!(scrolled, want, "scroll, round {round}: {f:?}");

        for strategy in [FilterStrategy::PreFilter, FilterStrategy::Auto] {
            let p = SearchParams::new(N as usize).filter(f.clone()).strategy(strategy);
            let hits: BTreeSet<u64> =
                idx.search_with(&[10.0, 3.0], &p).unwrap().iter().map(|h| h.id).collect();
            assert_eq!(hits, want, "{strategy:?}, round {round}: {f:?}");
        }
    }
}

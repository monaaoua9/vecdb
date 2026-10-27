mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use common::{clustered, recall};
use vecdb::{Config, Index, Item, Metric, Payload};

#[test]
fn inserts_and_searches_race_safely() {
    let (n, dim, writers) = (6000usize, 32usize, 4usize);
    let data = clustered(n, dim, 1);
    let mut cfg = Config::new(dim, Metric::L2);
    cfg.max_elements = n;
    cfg.ef_construction = 100;
    let idx = Index::new(cfg).unwrap();
    let done = AtomicBool::new(false);
    let queries = clustered(50, dim, 2);

    std::thread::scope(|s| {
        let mut handles = vec![];
        for w in 0..writers {
            let (idx, data) = (&idx, &data);
            handles.push(s.spawn(move || {
                for i in (w..n).step_by(writers) {
                    idx.insert(i as u64, &data[i], Payload::new()).unwrap();
                }
            }));
        }
        for _ in 0..3 {
            let (idx, done, queries) = (&idx, &done, &queries);
            s.spawn(move || {
                while !done.load(Ordering::Relaxed) {
                    for q in queries {
                        let hits = idx.search(q, 10).unwrap();
                        assert!(hits.len() <= 10);
                        assert!(hits.windows(2).all(|w| w[0].distance <= w[1].distance));
                    }
                }
            });
        }
        for h in handles {
            h.join().unwrap();
        }
        done.store(true, Ordering::Relaxed);
    });

    assert_eq!(idx.len(), n);
    let r = recall(&idx, &queries, 10, 80);
    println!("recall after concurrent build = {r:.4}");
    assert!(r >= 0.95, "recall {r}");

    let mut missing = 0;
    for i in (0..n).step_by(7) {
        let hits = idx.search(&data[i], 5).unwrap();
        if !hits.iter().any(|h| h.id == i as u64) {
            missing += 1;
        }
    }
    assert!(missing <= n / 7 / 100, "{missing} vectors unreachable");
}

#[test]
fn bulk_insert_matches_count() {
    let dim = 16;
    let data = clustered(2000, dim, 5);
    let items: Vec<_> = data
        .into_iter()
        .enumerate()
        .map(|(i, v)| Item { id: i as u64, vector: v, payload: Payload::new() })
        .collect();
    let mut cfg = Config::new(dim, Metric::Cosine);
    cfg.max_elements = 2000;
    let idx = Index::new(cfg).unwrap();
    idx.insert_bulk(&items, 4).unwrap();
    assert_eq!(idx.len(), 2000);
}

#[test]
fn duplicate_ids_race_has_one_winner() {
    let mut cfg = Config::new(4, Metric::L2);
    cfg.max_elements = 100;
    let idx = Index::new(cfg).unwrap();
    let wins = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| {
                if idx.insert(99, &[1.0, 2.0, 3.0, 4.0], Payload::new()).is_ok() {
                    wins.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });
    assert_eq!(wins.load(Ordering::Relaxed), 1);
    assert_eq!(idx.len(), 1);
}

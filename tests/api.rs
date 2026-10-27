use vecdb::{Config, Error, Index, Metric, Payload, SearchParams};

fn small() -> Index {
    let mut cfg = Config::new(3, Metric::L2);
    cfg.max_elements = 4;
    Index::new(cfg).unwrap()
}

#[test]
fn rejects_bad_input() {
    let idx = small();
    assert!(matches!(idx.insert(1, &[1.0, 2.0], Payload::new()), Err(Error::DimensionMismatch { .. })));
    assert!(matches!(idx.insert(1, &[1.0, f32::NAN, 0.0], Payload::new()), Err(Error::InvalidVector(_))));
    assert!(matches!(idx.search(&[0.0; 5], 3), Err(Error::DimensionMismatch { .. })));
    assert!(idx.is_empty());
}

#[test]
fn cosine_rejects_zero_vector() {
    let idx = Index::new(Config::new(3, Metric::Cosine)).unwrap();
    assert!(idx.insert(1, &[0.0; 3], Payload::new()).is_err());
}

#[test]
fn capacity_is_enforced() {
    let idx = small();
    for i in 0..4 {
        idx.insert(i, &[i as f32, 0.0, 0.0], Payload::new()).unwrap();
    }
    assert!(matches!(idx.insert(9, &[9.0, 0.0, 0.0], Payload::new()), Err(Error::CapacityExceeded(4))));
}

#[test]
fn empty_index_and_k_zero() {
    let idx = small();
    assert!(idx.search(&[0.0; 3], 5).unwrap().is_empty());
    idx.insert(1, &[1.0, 0.0, 0.0], Payload::new()).unwrap();
    assert!(idx.search_with(&[0.0; 3], &SearchParams::new(0)).unwrap().is_empty());
}

#[test]
fn k_larger_than_len() {
    let idx = small();
    idx.insert(1, &[1.0, 0.0, 0.0], Payload::new()).unwrap();
    idx.insert(2, &[2.0, 0.0, 0.0], Payload::new()).unwrap();
    let hits = idx.search(&[0.0; 3], 10).unwrap();
    assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![1, 2]);
    assert_eq!(hits[1].distance, 4.0);
}

#[test]
fn delete_then_reinsert() {
    let idx = small();
    idx.insert(1, &[1.0, 0.0, 0.0], Payload::new()).unwrap();
    idx.delete(1).unwrap();
    assert!(matches!(idx.delete(1), Err(Error::NotFound(1))));
    assert!(idx.get(1).is_none());
    idx.insert(1, &[5.0, 0.0, 0.0], Payload::new()).unwrap();
    assert_eq!(idx.get(1).unwrap().vector, vec![5.0, 0.0, 0.0]);
    assert_eq!(idx.search(&[5.0, 0.0, 0.0], 1).unwrap()[0].id, 1);
}

#[test]
fn invalid_config() {
    assert!(Index::new(Config::new(0, Metric::L2)).is_err());
    let mut c = Config::new(4, Metric::L2);
    c.m = 1;
    assert!(Index::new(c).is_err());
}

#[test]
fn compaction_drops_tombstones_and_keeps_data() {
    use vecdb::{Filter, Value};
    let mut cfg = Config::new(4, Metric::L2);
    cfg.max_elements = 1000;
    let idx = Index::new(cfg).unwrap();
    for i in 0..500u64 {
        let mut p = Payload::new();
        p.insert("grp".into(), Value::Int((i % 5) as i64));
        idx.insert(i, &[i as f32, (i % 7) as f32, 1.0, 0.0], p).unwrap();
    }
    for i in 0..500u64 {
        if i % 5 != 0 {
            idx.delete(i).unwrap();
        }
    }
    assert!(idx.tombstone_ratio() > 0.7);

    let c = idx.compacted(2).unwrap();
    assert_eq!(c.len(), 100);
    assert_eq!(c.tombstone_ratio(), 0.0);
    assert_eq!(c.stats().deleted, 0);
    assert!(c.get(10).is_some() && c.get(11).is_none());

    let q = [250.2, 5.0, 1.0, 0.0];
    assert_eq!(c.search(&q, 1).unwrap()[0].id, idx.search(&q, 1).unwrap()[0].id);
    let f = SearchParams::new(5).filter(Filter::eq("grp", 0));
    assert_eq!(c.search_with(&q, &f).unwrap().len(), 5);
}

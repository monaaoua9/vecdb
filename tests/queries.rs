use vecdb::{Config, Error, Filter, Index, Metric, Payload, SearchParams, Value};

// points on a line: id i sits at x = i, so distances are easy to reason about
fn line(n: u64) -> Index {
    let mut cfg = Config::new(2, Metric::L2);
    cfg.max_elements = 1000;
    let idx = Index::new(cfg).unwrap();
    for i in 0..n {
        let mut p = Payload::new();
        p.insert("even".into(), Value::Bool(i % 2 == 0));
        idx.insert(i, &[i as f32, 0.0], p).unwrap();
    }
    idx
}

#[test]
fn batch_matches_single_queries() {
    let idx = line(300);
    let qs: Vec<Vec<f32>> = (0..40).map(|i| vec![i as f32 * 7.0 + 0.2, 0.0]).collect();
    let p = SearchParams::new(3);
    let batch = idx.search_batch(&qs, &p, 4).unwrap();
    assert_eq!(batch.len(), 40);
    for (q, got) in qs.iter().zip(&batch) {
        assert_eq!(got, &idx.search_with(q, &p).unwrap());
    }
    assert!(idx.search_batch(&[], &p, 4).unwrap().is_empty());
    let bad = idx.search_batch(&[vec![1.0, 0.0], vec![1.0]], &p, 2);
    assert!(matches!(bad, Err(Error::DimensionMismatch { .. })));
}

#[test]
fn range_returns_only_what_is_inside() {
    let idx = line(300);
    // squared distance <= 25 around x=100 means ids 95..=105
    let hits = idx.search_range(&[100.0, 0.0], 25.0, &SearchParams::new(100)).unwrap();
    let mut ids: Vec<u64> = hits.iter().map(|h| h.id).collect();
    assert_eq!(ids[0], 100);
    ids.sort();
    assert_eq!(ids, (95..=105).collect::<Vec<_>>());

    // the cap wins over the radius
    assert_eq!(idx.search_range(&[100.0, 0.0], 25.0, &SearchParams::new(4)).unwrap().len(), 4);
    // wide radius needs several doublings past the first 32
    assert_eq!(idx.search_range(&[150.0, 0.0], 1e9, &SearchParams::new(200)).unwrap().len(), 200);
    assert!(idx.search_range(&[1000.0, 0.0], 1.0, &SearchParams::new(10)).unwrap().is_empty());
}

#[test]
fn range_respects_filters() {
    let idx = line(100);
    let p = SearchParams::new(50).filter(Filter::eq("even", true));
    let hits = idx.search_range(&[50.0, 0.0], 100.0, &p).unwrap();
    assert!(hits.iter().all(|h| h.id % 2 == 0));
    assert_eq!(hits.len(), 11); // 40, 42 .. 60
}

#[test]
fn recommend_moves_towards_positives_and_away_from_negatives() {
    let idx = line(200);
    let p = SearchParams::new(3);

    let hits = idx.recommend(&[100], &[], &p).unwrap();
    assert!(!hits.iter().any(|h| h.id == 100));
    assert!(hits.iter().all(|h| (98..=102).contains(&h.id)));

    // positive 100, negative 90: target is x = 110
    let hits = idx.recommend(&[100], &[90], &p).unwrap();
    assert!(hits.iter().all(|h| h.id > 100));
    assert_eq!(hits[0].id, 110);

    assert!(matches!(idx.recommend(&[], &[1], &p), Err(Error::InvalidVector(_))));
    assert!(matches!(idx.recommend(&[9999], &[], &p), Err(Error::NotFound(9999))));
}

#[test]
fn mmr_spreads_results_out() {
    // two tight clumps: a plain top-4 stays in the near clump, mmr reaches for the other one
    let mut cfg = Config::new(2, Metric::L2);
    cfg.max_elements = 100;
    let idx = Index::new(cfg).unwrap();
    for i in 0..10u64 {
        idx.insert(i, &[i as f32 * 0.01, 0.0], Payload::new()).unwrap();
        idx.insert(100 + i, &[3.0 + i as f32 * 0.01, 0.0], Payload::new()).unwrap();
    }
    let p = SearchParams::new(4);
    let plain = idx.search_with(&[0.0, 0.0], &p).unwrap();
    assert!(plain.iter().all(|h| h.id < 100));

    let diverse = idx.search_mmr(&[0.0, 0.0], &p, 0.3, 20).unwrap();
    assert_eq!(diverse.len(), 4);
    assert_eq!(diverse[0].id, 0);
    assert!(diverse.iter().any(|h| h.id >= 100), "{diverse:?}");

    // lambda = 1 is ordinary nearest-neighbour order
    let same = idx.search_mmr(&[0.0, 0.0], &p, 1.0, 20).unwrap();
    assert_eq!(same.iter().map(|h| h.id).collect::<Vec<_>>(), plain.iter().map(|h| h.id).collect::<Vec<_>>());

    assert!(matches!(idx.search_mmr(&[0.0, 0.0], &p, 1.5, 20), Err(Error::InvalidVector(_))));
}

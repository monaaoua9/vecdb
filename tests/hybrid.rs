use vecdb::{Config, Error, Filter, Index, Metric, Payload, SearchParams, Value};

fn doc(text: &str, lang: &str) -> Payload {
    let mut p = Payload::new();
    p.insert("body".into(), Value::Str(text.into()));
    p.insert("lang".into(), Value::Str(lang.into()));
    p
}

fn corpus() -> Index {
    let mut cfg = Config::new(2, Metric::L2);
    cfg.max_elements = 100;
    let idx = Index::new(cfg).unwrap();
    // vectors: id i at x = i, so vector rank follows id distance
    idx.insert(0, &[0.0, 0.0], doc("rust vector database", "en")).unwrap();
    idx.insert(1, &[1.0, 0.0], doc("cooking pasta at home", "en")).unwrap();
    idx.insert(2, &[2.0, 0.0], doc("الكتابُ الجديد في المدرسة", "ar")).unwrap();
    idx.insert(3, &[50.0, 0.0], doc("database internals and hnsw graph", "en")).unwrap();
    idx.insert(4, &[51.0, 0.0], doc("قاعدة بيانات المتجهات", "ar")).unwrap();
    idx
}

#[test]
fn needs_enabling() {
    let idx = corpus();
    assert!(matches!(idx.search_text("rust", &SearchParams::new(3)), Err(Error::InvalidConfig(_))));
}

#[test]
fn keyword_search_with_arabic_folding() {
    let idx = corpus();
    idx.enable_text_index("body");
    let ids = |q: &str| -> Vec<u64> {
        idx.search_text(q, &SearchParams::new(5)).unwrap().iter().map(|h| h.id).collect()
    };
    assert_eq!(ids("database"), [0, 3]);
    // diacritics and alef variants are folded away on both sides
    assert_eq!(ids("الكتاب"), [2]);
    assert_eq!(ids("المدرسه"), [2]);
    assert!(ids("nothing here").is_empty());

    let ar = SearchParams::new(5).filter(Filter::eq("lang", "ar"));
    assert!(idx.search_text("database", &ar).unwrap().is_empty());
}

#[test]
fn index_follows_edits_and_deletes() {
    let idx = corpus();
    idx.enable_text_index("body");
    let n = |q: &str| idx.search_text(q, &SearchParams::new(5)).unwrap().len();
    idx.delete(0).unwrap();
    assert_eq!(n("database"), 1);
    idx.set_payload(1, doc("a database for pasta", "en")).unwrap();
    assert_eq!(n("database"), 2);
    assert_eq!(n("cooking"), 0);
    idx.upsert(3, &[50.0, 0.0], doc("something else", "en")).unwrap();
    assert_eq!(n("hnsw"), 0);
    idx.insert(9, &[9.0, 0.0], doc("fresh database entry", "en")).unwrap();
    assert_eq!(n("database"), 2);
}

#[test]
fn hybrid_fuses_both_rankings() {
    let idx = corpus();
    idx.enable_text_index("body");
    let p = SearchParams::new(3);
    // vector side alone: 0, 1, 2 (all near x=0). text side: "hnsw" only matches id 3.
    let vec_only: Vec<u64> = idx.search_with(&[0.0, 0.0], &p).unwrap().iter().map(|h| h.id).collect();
    assert_eq!(vec_only, [0, 1, 2]);

    let hybrid = idx.search_hybrid(&[0.0, 0.0], "hnsw graph", &p).unwrap();
    let ids: Vec<u64> = hybrid.iter().map(|h| h.id).collect();
    assert!(ids.contains(&3), "keyword-only match should surface: {ids:?}");
    assert!(hybrid.windows(2).all(|w| w[0].distance <= w[1].distance));

    // a doc that wins both lists comes first
    let both = idx.search_hybrid(&[0.0, 0.0], "rust database", &p).unwrap();
    assert_eq!(both[0].id, 0);
}

#[test]
fn survives_compaction() {
    let idx = corpus();
    idx.enable_text_index("body");
    idx.delete(1).unwrap();
    let c = idx.compacted(2).unwrap();
    assert_eq!(c.text_field().as_deref(), Some("body"));
    assert_eq!(c.search_text("database", &SearchParams::new(5)).unwrap().len(), 2);
}

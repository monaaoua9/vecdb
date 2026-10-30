use std::sync::Arc;

use vecdb::{Collections, Config, Durability, DurableIndex, Error, Metric, Payload, Quantization};

fn base() -> Config {
    let mut c = Config::new(4, Metric::L2);
    c.max_elements = 100;
    c
}

fn open(dir: &std::path::Path) -> Collections {
    let d = Arc::new(DurableIndex::open(dir, base(), Durability::Flush).unwrap());
    let cols = Collections::new(d);
    cols.load_existing().unwrap();
    cols
}

#[test]
fn collections_are_isolated() {
    let dir = tempfile::tempdir().unwrap();
    let cols = open(dir.path());

    let mut cfg = Config::new(2, Metric::Cosine);
    cfg.max_elements = 10;
    let docs = cols.create("docs", cfg).unwrap();
    docs.insert(1, &[1.0, 0.0], Payload::new()).unwrap();

    let default = cols.get("default").unwrap();
    default.insert(1, &[0.0, 0.0, 0.0, 9.0], Payload::new()).unwrap();

    assert_eq!(docs.get(1).unwrap().vector.len(), 2);
    assert_eq!(default.get(1).unwrap().vector.len(), 4);
    assert!(matches!(docs.insert(2, &[1.0; 4], Payload::new()), Err(Error::DimensionMismatch { .. })));
}

#[test]
fn name_rules_and_duplicates() {
    let dir = tempfile::tempdir().unwrap();
    let cols = open(dir.path());
    for bad in ["", "a b", "../x", "a/b", &"x".repeat(65)] {
        assert!(matches!(cols.create(bad, base()), Err(Error::InvalidName(_))), "{bad:?}");
    }
    cols.create("ok_1-2", base()).unwrap();
    assert!(matches!(cols.create("ok_1-2", base()), Err(Error::CollectionExists(_))));
    assert!(matches!(cols.create("default", base()), Err(Error::CollectionExists(_))));

    let mut pq = base();
    pq.quantization = Quantization::Pq { m: 2 };
    assert!(matches!(cols.create("pq", pq), Err(Error::InvalidConfig(_))));
}

#[test]
fn collections_come_back_after_restart() {
    let dir = tempfile::tempdir().unwrap();
    {
        let cols = open(dir.path());
        let mut cfg = Config::new(3, Metric::InnerProduct);
        cfg.max_elements = 50;
        cfg.quantization = Quantization::Sq8;
        let c = cols.create("emb", cfg).unwrap();
        c.insert(7, &[1.0, 2.0, 3.0], Payload::new()).unwrap();
        c.checkpoint().unwrap();
        c.insert(8, &[3.0, 2.0, 1.0], Payload::new()).unwrap();
        // second collection never checkpointed: only the wal has it
        let w = cols.create("walonly", base()).unwrap();
        w.insert(1, &[1.0; 4], Payload::new()).unwrap();
    }
    let cols = open(dir.path());
    let names: Vec<_> = cols.list().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, ["default", "emb", "walonly"]);

    let emb = cols.get("emb").unwrap();
    assert_eq!(emb.len(), 2);
    assert_eq!(emb.config().quantization, Quantization::Sq8);
    assert_eq!(cols.get("walonly").unwrap().len(), 1);
}

#[test]
fn dropped_collection_stays_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let cols = open(dir.path());
    let c = cols.create("tmp", base()).unwrap();
    c.insert(1, &[1.0; 4], Payload::new()).unwrap();
    drop(c);

    cols.drop_collection("tmp").unwrap();
    assert!(cols.get("tmp").is_none());
    assert!(matches!(cols.drop_collection("tmp"), Err(Error::CollectionNotFound(_))));
    assert!(matches!(cols.drop_collection("default"), Err(Error::InvalidName(_))));

    drop(cols);
    let cols = open(dir.path());
    assert!(cols.get("tmp").is_none());

    // and the name can be reused with a different shape
    let mut cfg = Config::new(2, Metric::L2);
    cfg.max_elements = 5;
    let again = cols.create("tmp", cfg).unwrap();
    assert!(again.is_empty());
}

#[test]
fn memory_only_server_can_still_create() {
    let d = Arc::new(DurableIndex::in_memory(vecdb::Index::new(base()).unwrap()));
    let cols = Collections::new(d);
    let c = cols.create("scratch", base()).unwrap();
    c.insert(1, &[1.0; 4], Payload::new()).unwrap();
    assert_eq!(cols.load_existing().unwrap(), 0);
    cols.drop_collection("scratch").unwrap();
}

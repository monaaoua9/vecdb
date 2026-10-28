mod common;

use common::clustered;
use vecdb::{Config, Error, Filter, Index, Metric, Payload, Quantization, SearchParams, Value};

fn payload(i: usize) -> Payload {
    let mut p = Payload::new();
    p.insert("parity".into(), Value::Int((i % 2) as i64));
    p
}

fn make(quant: Quantization, n: usize) -> (Index, Vec<Vec<f32>>) {
    let dim = 20;
    let data = clustered(n, dim, 3);
    let mut cfg = Config::new(dim, Metric::Cosine);
    cfg.quantization = quant;
    cfg.max_elements = n + 500;
    let idx = Index::new(cfg).unwrap();
    for (i, v) in data.iter().enumerate() {
        idx.insert(i as u64, v, payload(i)).unwrap();
    }
    (idx, data)
}

#[test]
fn roundtrip_preserves_results() {
    for quant in [Quantization::None, Quantization::Sq8] {
        let (idx, data) = make(quant, 1500);
        idx.delete(7).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.vdb");
        idx.save(&path).unwrap();

        let back = Index::open(&path).unwrap();
        assert_eq!(back.len(), idx.len());
        assert!(!back.contains(7));
        assert!(back.stats().mapped_bytes > 0);
        for q in data.iter().step_by(97) {
            assert_eq!(idx.search(q, 10).unwrap(), back.search(q, 10).unwrap(), "{quant:?}");
        }
        let f = SearchParams::new(5).filter(Filter::eq("parity", 1));
        assert_eq!(idx.search_with(&data[0], &f).unwrap(), back.search_with(&data[0], &f).unwrap());
    }
}

#[test]
fn opened_index_accepts_inserts() {
    let (idx, data) = make(Quantization::None, 800);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("b.vdb");
    idx.save(&path).unwrap();
    drop(idx);

    let back = Index::open(&path).unwrap();
    let extra = clustered(200, 20, 77);
    for (i, v) in extra.iter().enumerate() {
        back.insert(10_000 + i as u64, v, payload(i)).unwrap();
    }
    assert_eq!(back.len(), 1000);
    assert_eq!(back.search(&extra[5], 1).unwrap()[0].id, 10_005);
    assert_eq!(back.search(&data[5], 1).unwrap()[0].id, 5);
    assert!(matches!(back.insert(5, &data[5], Payload::new()), Err(Error::DuplicateId(5))));
}

#[test]
fn detects_corruption() {
    let (idx, _) = make(Quantization::None, 300);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.vdb");
    idx.save(&path).unwrap();

    let good = std::fs::read(&path).unwrap();

    let mut bad = good.clone();
    bad[20] ^= 0xff; // header field
    std::fs::write(&path, &bad).unwrap();
    assert!(matches!(Index::open(&path), Err(Error::Corrupt(_))));

    let mut bad = good.clone();
    let last = bad.len() - 3;
    bad[last] ^= 0xff; // metadata blob
    std::fs::write(&path, &bad).unwrap();
    assert!(matches!(Index::open(&path), Err(Error::Corrupt(_))));

    std::fs::write(&path, &good[..good.len() / 2]).unwrap();
    assert!(Index::open(&path).is_err());

    std::fs::write(&path, b"nope").unwrap();
    assert!(Index::open(&path).is_err());
}

#[test]
fn save_while_searching() {
    let (idx, data) = make(Quantization::None, 1000);
    let dir = tempfile::tempdir().unwrap();
    std::thread::scope(|s| {
        s.spawn(|| {
            for q in &data[..50] {
                idx.search(q, 5).unwrap();
            }
        });
        idx.save(dir.path().join("d.vdb")).unwrap();
    });
}

#[test]
fn damaged_snapshots_error_instead_of_panicking() {
    use vecdb::{Item, Payload};
    let data = common::clustered(200, 8, 31);
    let mut c = Config::new(8, Metric::L2);
    c.max_elements = 500;
    let idx = Index::new(c).unwrap();
    let items: Vec<Item> = data
        .iter()
        .enumerate()
        .map(|(i, v)| Item { id: i as u64, vector: v.clone(), payload: Payload::new() })
        .collect();
    idx.insert_bulk(&items, 2).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let good = dir.path().join("good.vdb");
    idx.save(&good).unwrap();
    let bytes = std::fs::read(&good).unwrap();
    let probe = dir.path().join("bad.vdb");

    // header, vector section and metadata all get hit; metadata and header are checksummed
    for pos in (0..bytes.len()).step_by(37) {
        let mut b = bytes.clone();
        b[pos] ^= 0xA5;
        std::fs::write(&probe, &b).unwrap();
        let _ = Index::open(&probe);
    }
    for cut in (0..bytes.len()).step_by(211) {
        std::fs::write(&probe, &bytes[..cut]).unwrap();
        assert!(Index::open(&probe).is_err(), "cut at {cut} opened");
    }
}

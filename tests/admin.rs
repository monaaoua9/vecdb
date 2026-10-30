use std::io::Write;
use std::sync::Arc;

use vecdb::admin::{self, backup, export_jsonl, import_jsonl, inspect, restore};
use vecdb::{Collections, Config, Durability, DurableIndex, Error, Metric, Payload, Value, snapshots};

fn cfg() -> Config {
    let mut c = Config::new(3, Metric::L2);
    c.max_elements = 500;
    c
}

fn doc(tag: &str, n: i64, score: f64) -> Payload {
    let mut p = Payload::new();
    p.insert("tag".into(), Value::Str(tag.into()));
    p.insert("n".into(), Value::Int(n));
    p.insert("score".into(), Value::Float(score));
    p.insert("ok".into(), Value::Bool(n % 2 == 0));
    p
}

fn populated(dir: &std::path::Path) -> DurableIndex {
    let d = DurableIndex::open(dir, cfg(), Durability::Flush).unwrap();
    for i in 0..40u64 {
        d.insert(i, &[i as f32, 0.5, -1.0], doc("a", i as i64, i as f64 + 0.25)).unwrap();
    }
    d.checkpoint().unwrap();
    for i in 40..50u64 {
        d.insert(i, &[i as f32, 0.5, -1.0], doc("b", i as i64, 1.0)).unwrap();
    }
    d.delete(3).unwrap();
    d
}

#[test]
fn inspect_reports_snapshot_and_log() {
    let dir = tempfile::tempdir().unwrap();
    let d = populated(dir.path());
    let r = inspect(dir.path()).unwrap();

    let s = r.snapshot.as_ref().unwrap();
    assert_eq!((s.generation, s.live), (1, 40));
    assert_eq!(s.config.dim, 3);
    assert!(r.wals.iter().any(|w| w.records == 11)); // 10 inserts + 1 delete after the snapshot
    assert!(r.problems().is_empty() && r.warnings().is_empty());
    drop(d);
}

#[test]
fn verify_tells_torn_tail_from_damage() {
    let dir = tempfile::tempdir().unwrap();
    drop(populated(dir.path()));

    // half a record on the newest log: recovery handles it, so only a warning
    let wal = vecdb::admin::inspect(dir.path()).unwrap().wals.last().unwrap().path.clone();
    std::fs::OpenOptions::new().append(true).open(&wal).unwrap().write_all(&[9, 0, 0, 0, 1, 2]).unwrap();
    let r = inspect(dir.path()).unwrap();
    assert!(r.problems().is_empty());
    assert_eq!(r.warnings().len(), 1);

    // a damaged snapshot is a real problem
    let (_, snap) = snapshots::latest(dir.path()).unwrap();
    let mut bytes = std::fs::read(&snap).unwrap();
    bytes[10] ^= 0xff;
    std::fs::write(&snap, bytes).unwrap();
    let r = inspect(dir.path()).unwrap();
    assert!(r.problems().iter().any(|p| p.starts_with("snapshot")), "{:?}", r.problems());
}

#[test]
fn backup_while_open_restores_to_the_same_data() {
    let dir = tempfile::tempdir().unwrap();
    let live = Arc::new(populated(dir.path()));
    let cols = Collections::new(live.clone());
    let mut c2 = Config::new(2, Metric::Cosine);
    c2.max_elements = 50;
    let extra = cols.create("docs", c2).unwrap();
    extra.insert(1, &[1.0, 0.0], doc("c", 1, 0.5)).unwrap();
    extra.checkpoint().unwrap();
    extra.insert(2, &[0.0, 1.0], doc("c", 2, 0.5)).unwrap();

    let bak = tempfile::tempdir().unwrap();
    let rep = backup(dir.path(), &bak.path().join("b")).unwrap();
    assert!(rep.files >= 5 && rep.bytes > 0);

    // an existing, non-empty target is refused
    assert!(matches!(backup(dir.path(), bak.path()), Err(Error::BadInput(_))));

    let out = tempfile::tempdir().unwrap();
    let (_, report) = restore(&bak.path().join("b"), &out.path().join("r")).unwrap();
    assert!(report.problems().is_empty());
    assert_eq!(report.collections.len(), 1);

    let back = Arc::new(DurableIndex::open(out.path().join("r"), cfg(), Durability::Flush).unwrap());
    assert_eq!(back.len(), live.len());
    assert!(!back.contains(3));
    assert_eq!(back.get(45).unwrap().payload["tag"], Value::Str("b".into()));
    let cols2 = Collections::new(back);
    assert_eq!(cols2.load_existing().unwrap(), 1);
    assert_eq!(cols2.get("docs").unwrap().len(), 2);
}

#[test]
fn export_import_roundtrip_keeps_types() {
    let dir = tempfile::tempdir().unwrap();
    let src = populated(dir.path());
    let mut buf = Vec::new();
    assert_eq!(export_jsonl(&src, &mut buf).unwrap(), 49);

    let first = String::from_utf8_lossy(&buf).lines().next().unwrap().to_owned();
    assert!(first.contains("\"vector\":[0.0,0.5,-1.0]"), "{first}");

    let other = tempfile::tempdir().unwrap();
    let dst = admin::open_dir(other.path(), Some(cfg())).unwrap();
    assert_eq!(import_jsonl(&dst, buf.as_slice(), 2).unwrap(), 49);
    assert_eq!(dst.len(), 49);
    for id in [0u64, 7, 45] {
        let (a, b) = (src.get(id).unwrap(), dst.get(id).unwrap());
        assert_eq!(a.vector, b.vector);
        assert_eq!(a.payload, b.payload);
    }
    // Int stayed Int and Float stayed Float
    assert_eq!(dst.get(7).unwrap().payload["n"], Value::Int(7));
    assert_eq!(dst.get(40).unwrap().payload["score"], Value::Float(1.0));
}

#[test]
fn import_names_the_bad_line() {
    let dir = tempfile::tempdir().unwrap();
    let d = admin::open_dir(dir.path(), Some(cfg())).unwrap();
    let text = "{\"id\":1,\"vector\":[1,2,3]}\n\n{\"id\":2,\"vector\":[1,2,3],\"payload\":{\"x\":[1]}}\n";
    let err = import_jsonl(&d, text.as_bytes(), 1).unwrap_err();
    assert!(matches!(&err, Error::BadInput(m) if m.starts_with("line 3")), "{err}");

    let err = import_jsonl(&d, "not json\n".as_bytes(), 1).unwrap_err();
    assert!(matches!(&err, Error::BadInput(m) if m.starts_with("line 1")), "{err}");

    // a directory with no snapshot and no description of the index can't be opened blind
    let empty = tempfile::tempdir().unwrap();
    assert!(matches!(admin::open_dir(empty.path(), None), Err(Error::InvalidConfig(_))));
}

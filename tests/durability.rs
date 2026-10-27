mod common;

use common::clustered;
use std::io::Write;
use vecdb::{Config, Durability, DurableIndex, Metric, Payload, snapshots};

fn cfg() -> Config {
    let mut c = Config::new(12, Metric::L2);
    c.max_elements = 2000;
    c
}

fn wal_paths(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut v: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "log"))
        .collect();
    v.sort();
    v
}

#[test]
fn crash_without_checkpoint_loses_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let data = clustered(300, 12, 1);
    {
        let d = DurableIndex::open(dir.path(), cfg(), Durability::Flush).unwrap();
        for (i, v) in data.iter().enumerate() {
            d.insert(i as u64, v, Payload::new()).unwrap();
        }
        d.delete(5).unwrap();
        // dropped without checkpoint: only the wal exists
    }
    assert!(snapshots::latest(dir.path()).is_none());

    let d = DurableIndex::open(dir.path(), cfg(), Durability::Flush).unwrap();
    assert_eq!(d.len(), 299);
    assert!(!d.contains(5));
    assert_eq!(d.search(&data[77], 1).unwrap()[0].id, 77);
}

#[test]
fn snapshot_plus_wal_tail() {
    let dir = tempfile::tempdir().unwrap();
    let data = clustered(400, 12, 2);
    {
        let d = DurableIndex::open(dir.path(), cfg(), Durability::Fsync).unwrap();
        for (i, v) in data[..200].iter().enumerate() {
            d.insert(i as u64, v, Payload::new()).unwrap();
        }
        let (path, n) = d.checkpoint().unwrap().unwrap();
        assert!(path.exists());
        assert_eq!(n, 200);
        for (i, v) in data.iter().enumerate().skip(200) {
            d.insert(i as u64, v, Payload::new()).unwrap();
        }
        d.delete(10).unwrap();
        d.delete(250).unwrap();
    }
    let d = DurableIndex::open(dir.path(), cfg(), Durability::Flush).unwrap();
    assert_eq!(d.len(), 398);
    assert!(!d.contains(10) && !d.contains(250));
    assert_eq!(d.search(&data[399], 1).unwrap()[0].id, 399);
    // the wal that the snapshot covers is gone
    assert_eq!(wal_paths(dir.path()).len(), 1);
}

#[test]
fn torn_wal_tail_is_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let data = clustered(50, 12, 3);
    {
        let d = DurableIndex::open(dir.path(), cfg(), Durability::Flush).unwrap();
        for (i, v) in data.iter().enumerate() {
            d.insert(i as u64, v, Payload::new()).unwrap();
        }
    }
    let wal = wal_paths(dir.path()).pop().unwrap();
    let clean_len = std::fs::metadata(&wal).unwrap().len();
    // half a record, as if the process died mid-write
    std::fs::OpenOptions::new()
        .append(true)
        .open(&wal)
        .unwrap()
        .write_all(&[200, 0, 0, 0, 1, 2, 3, 4, 9, 9])
        .unwrap();

    let d = DurableIndex::open(dir.path(), cfg(), Durability::Flush).unwrap();
    assert_eq!(d.len(), 50);
    assert_eq!(std::fs::metadata(&wal).unwrap().len(), clean_len);
    // and the log keeps working afterwards
    d.insert(1000, &data[0], Payload::new()).unwrap();
    drop(d);
    let d = DurableIndex::open(dir.path(), cfg(), Durability::Flush).unwrap();
    assert_eq!(d.len(), 51);
}

#[test]
fn failed_ops_are_not_logged_and_replay_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    {
        let d = DurableIndex::open(dir.path(), cfg(), Durability::Flush).unwrap();
        d.insert(1, &[0.5; 12], Payload::new()).unwrap();
        assert!(d.insert(1, &[0.5; 12], Payload::new()).is_err());
        assert!(d.insert(2, &[0.5; 3], Payload::new()).is_err());
        assert!(d.delete(99).is_err());
    }
    // open twice in a row: replaying the same wal again must not change anything
    for _ in 0..2 {
        let d = DurableIndex::open(dir.path(), cfg(), Durability::Flush).unwrap();
        assert_eq!(d.len(), 1);
    }
}

#[test]
fn in_memory_mode_has_no_files() {
    let d = DurableIndex::in_memory(vecdb::Index::new(cfg()).unwrap());
    d.insert(1, &[1.0; 12], Payload::new()).unwrap();
    assert!(d.checkpoint().unwrap().is_none());
    assert!(!d.is_durable());
}

#[test]
fn compaction_is_persisted() {
    let dir = tempfile::tempdir().unwrap();
    let data = clustered(200, 12, 4);
    let mut d = DurableIndex::open(dir.path(), cfg(), Durability::Flush).unwrap();
    for (i, v) in data.iter().enumerate() {
        d.insert(i as u64, v, Payload::new()).unwrap();
    }
    for i in 0..150u64 {
        d.delete(i).unwrap();
    }
    assert!(d.tombstone_ratio() > 0.5);
    d.compact(2).unwrap();
    d.insert(9000, &data[0], Payload::new()).unwrap(); // log still works after the swap
    drop(d);

    let d = DurableIndex::open(dir.path(), cfg(), Durability::Flush).unwrap();
    assert_eq!(d.len(), 51);
    assert_eq!(d.tombstone_ratio(), 0.0);
    assert!(d.contains(9000) && !d.contains(3));
    assert_eq!(d.search(&data[199], 1).unwrap()[0].id, 199);
}

#[test]
fn concurrent_writers_share_fsyncs() {
    let dir = tempfile::tempdir().unwrap();
    let (threads, per) = (8u64, 60u64);
    let mut c = cfg();
    c.max_elements = 1000;
    let d = DurableIndex::open(dir.path(), c.clone(), Durability::Fsync).unwrap();
    std::thread::scope(|s| {
        for t in 0..threads {
            let d = &d;
            s.spawn(move || {
                for i in 0..per {
                    let id = t * 1000 + i;
                    d.insert(id, &[id as f32; 12], Payload::new()).unwrap();
                }
            });
        }
    });
    let writes = threads * per;
    let batches = d.wal_batches();
    println!("{writes} acknowledged writes in {batches} log batches");
    assert!(batches < writes, "no batching happened: {batches} batches for {writes} writes");
    drop(d);

    let d = DurableIndex::open(dir.path(), c, Durability::Fsync).unwrap();
    assert_eq!(d.len() as u64, writes);
}

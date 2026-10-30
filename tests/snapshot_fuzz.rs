//! Mutation fuzzing of the on-disk formats. Whatever happens to the bytes, opening must return
//! an error or an index that still answers queries; it must never panic, and a snapshot that
//! does open must still report the vector count it was written with (the header is checksummed).
// plain % reads fine in the test data generators
#![allow(clippy::manual_is_multiple_of)]

mod common;

use std::path::Path;

use common::{Rng, clustered};
use vecdb::{Config, Durability, DurableIndex, Index, Item, Metric, Payload, Quantization, Value};

const DIM: usize = 8;
const N: usize = 150;

fn mutate(bytes: &[u8], r: &mut Rng) -> Vec<u8> {
    let mut b = bytes.to_vec();
    let at = |r: &mut Rng, len: usize| (r.next_u64() as usize) % len.max(1);
    match r.next_u64() % 6 {
        0 => {
            for _ in 0..1 + r.next_u64() % 8 {
                let p = at(r, b.len());
                b[p] ^= 1 << (r.next_u64() % 8);
            }
        }
        1 => {
            let (p, n) = (at(r, b.len()), 1 + at(r, 64));
            let fill = if r.next_u64() % 2 == 0 { 0x00 } else { 0xFF };
            for x in b.iter_mut().skip(p).take(n) {
                *x = fill;
            }
        }
        2 => {
            let p = at(r, b.len());
            b.truncate(p);
        }
        3 => {
            let (p, n) = (at(r, b.len()), 1 + at(r, 16));
            for _ in 0..n {
                b.insert(p.min(b.len()), (r.next_u64() & 0xff) as u8);
            }
        }
        4 => {
            let (p, n) = (at(r, b.len()), 1 + at(r, 32));
            let end = (p + n).min(b.len());
            b.drain(p..end);
        }
        _ => {
            // overwrite a few bytes with one run copied from elsewhere in the file
            let (src, dst, n) = (at(r, b.len()), at(r, b.len()), 1 + at(r, 48));
            let chunk: Vec<u8> = b.iter().skip(src).take(n).copied().collect();
            for (i, x) in chunk.into_iter().enumerate() {
                if let Some(slot) = b.get_mut(dst + i) {
                    *slot = x;
                }
            }
        }
    }
    b
}

fn build(quant: Quantization, rerank: bool) -> Index {
    let data = clustered(N, DIM, 5);
    let mut cfg = Config::new(DIM, Metric::L2);
    cfg.quantization = quant;
    cfg.rerank = rerank;
    cfg.max_elements = 400;
    let items: Vec<Item> = data
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut p = Payload::new();
            p.insert("n".into(), Value::Int(i as i64));
            Item { id: i as u64, vector: v.clone(), payload: p }
        })
        .collect();
    let idx = match quant {
        Quantization::Pq { .. } => {
            let flat: Vec<f32> = data.iter().flatten().copied().collect();
            let sample: Vec<f32> = flat.iter().cycle().take(DIM * 300).copied().collect();
            Index::new_pq(cfg, &sample, 2).unwrap()
        }
        _ => Index::new(cfg).unwrap(),
    };
    idx.insert_bulk(&items, 2).unwrap();
    idx.delete(7).unwrap();
    idx
}

fn fuzz_snapshot(quant: Quantization, rerank: bool, rounds: usize, seed: u64) {
    let idx = build(quant, rerank);
    let live = idx.len();
    let dir = tempfile::tempdir().unwrap();
    let good = dir.path().join("good.vdb");
    idx.save(&good).unwrap();
    let bytes = std::fs::read(&good).unwrap();
    let probe = dir.path().join("probe.vdb");
    let q = [0.5f32; DIM];

    let mut r = Rng::new(seed);
    let mut opened = 0;
    for _ in 0..rounds {
        std::fs::write(&probe, mutate(&bytes, &mut r)).unwrap();
        if let Ok(back) = Index::open(&probe) {
            opened += 1;
            assert_eq!(back.len(), live);
            // vector bytes aren't checksummed, so garbage values are allowed, panics aren't
            let _ = back.search(&q, 5);
            let _ = back.get(3);
            let _ = back.stats();
        }
    }
    println!("{quant:?} rerank={rerank}: {opened}/{rounds} mutations still opened");
}

#[test]
fn snapshot_f32() {
    fuzz_snapshot(Quantization::None, false, 400, 1);
}

#[test]
fn snapshot_sq8() {
    fuzz_snapshot(Quantization::Sq8, false, 300, 2);
}

#[test]
fn snapshot_binary_with_rerank() {
    fuzz_snapshot(Quantization::Binary, true, 300, 3);
}

#[test]
fn snapshot_pq() {
    fuzz_snapshot(Quantization::Pq { m: 2 }, false, 300, 4);
}

fn newest(dir: &Path, ext: &str) -> std::path::PathBuf {
    let mut v: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == ext))
        .collect();
    v.sort();
    v.pop().unwrap()
}

#[test]
fn wal_mutations_only_ever_lose_a_suffix() {
    let src = tempfile::tempdir().unwrap();
    let mut cfg = Config::new(DIM, Metric::L2);
    cfg.max_elements = 400;
    let data = clustered(60, DIM, 8);
    {
        let d = DurableIndex::open(src.path(), cfg.clone(), Durability::Flush).unwrap();
        for (i, v) in data.iter().enumerate() {
            let mut p = Payload::new();
            p.insert("n".into(), Value::Int(i as i64));
            d.insert(i as u64, v, p).unwrap();
        }
    }
    let wal = newest(src.path(), "log");
    let bytes = std::fs::read(&wal).unwrap();

    let mut r = Rng::new(99);
    for round in 0..150 {
        let work = tempfile::tempdir().unwrap();
        std::fs::write(work.path().join(wal.file_name().unwrap()), mutate(&bytes, &mut r)).unwrap();
        let d = DurableIndex::open(work.path(), cfg.clone(), Durability::Flush)
            .unwrap_or_else(|e| panic!("round {round}: open failed: {e}"));
        // CRC per record: whatever survived is byte-for-byte what was written
        for rec in d.scroll(None, 0, 1000, true).items {
            assert_eq!(rec.vector, data[rec.id as usize], "round {round}: id {}", rec.id);
            assert_eq!(rec.payload["n"], Value::Int(rec.id as i64));
        }
    }
}

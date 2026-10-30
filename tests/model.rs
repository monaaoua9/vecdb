//! Randomised model check: a long mix of every mutation, checkpoints and online compactions,
//! then a "crash" (drop) and a reopen. The reopened index must equal a plain HashMap that saw
//! the same operations.

mod common;

use std::collections::HashMap;

use common::Rng;
use vecdb::{Config, Durability, DurableIndex, Metric, Payload, Quantization, Value};

const DIM: usize = 4;

type Model = HashMap<u64, (Vec<f32>, Payload)>;

fn payload(r: &mut Rng) -> Payload {
    let mut p = Payload::new();
    for key in ["a", "b", "c"] {
        match r.next_u64() % 4 {
            0 => {}
            1 => drop(p.insert(key.into(), Value::Int((r.next_u64() % 5) as i64))),
            2 => drop(p.insert(key.into(), Value::Str(format!("s{}", r.next_u64() % 3)))),
            _ => drop(p.insert(key.into(), Value::Bool(r.next_u64() % 2 == 0))),
        }
    }
    p
}

fn vector(r: &mut Rng) -> Vec<f32> {
    (0..DIM).map(|_| (r.next_u64() % 1000) as f32 / 10.0).collect()
}

fn run(seed: u64, quant: Quantization, steps: usize) {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = Config::new(DIM, Metric::L2);
    cfg.quantization = quant;
    cfg.max_elements = steps * 2;
    let mut r = Rng::new(seed);
    let mut model = Model::new();

    let d = DurableIndex::open(dir.path(), cfg.clone(), Durability::Flush).unwrap();
    for step in 0..steps {
        let id = r.next_u64() % 60;
        match r.next_u64() % 10 {
            0..=2 => {
                let (v, p) = (vector(&mut r), payload(&mut r));
                if d.insert(id, &v, p.clone()).is_ok() {
                    assert!(model.insert(id, (v, p)).is_none(), "insert of a live id succeeded");
                } else {
                    assert!(model.contains_key(&id));
                }
            }
            3 | 4 => {
                let (v, p) = (vector(&mut r), payload(&mut r));
                d.upsert(id, &v, p.clone()).unwrap();
                model.insert(id, (v, p));
            }
            5 => {
                let ok = d.delete(id).is_ok();
                assert_eq!(ok, model.remove(&id).is_some());
            }
            6 => {
                let p = payload(&mut r);
                let ok = d.set_payload(id, p.clone()).is_ok();
                match model.get_mut(&id) {
                    Some(m) => {
                        assert!(ok);
                        m.1 = p;
                    }
                    None => assert!(!ok),
                }
            }
            7 => {
                let p = payload(&mut r);
                let ok = d.merge_payload(id, p.clone()).is_ok();
                match model.get_mut(&id) {
                    Some(m) => {
                        assert!(ok);
                        m.1.extend(p);
                    }
                    None => assert!(!ok),
                }
            }
            8 if step % 7 == 0 => {
                d.checkpoint().unwrap();
            }
            9 if step % 11 == 0 => {
                assert!(d.compact_online(2).unwrap());
            }
            _ => {}
        }
    }

    check(&d, &model, quant);
    drop(d);
    let d = DurableIndex::open(dir.path(), cfg, Durability::Flush).unwrap();
    check(&d, &model, quant);
}

fn check(d: &DurableIndex, model: &Model, quant: Quantization) {
    assert_eq!(d.len(), model.len());
    for (id, (v, p)) in model {
        let got = d.get(*id).unwrap_or_else(|| panic!("id {id} missing"));
        assert_eq!(&got.payload, p, "payload of {id}");
        match quant {
            Quantization::None => assert_eq!(&got.vector, v, "vector of {id}"),
            // quantized stores hand back an approximation
            _ => assert!(got.vector.iter().zip(v).all(|(a, b)| (a - b).abs() < 1.0), "vector of {id}"),
        }
    }
    // everything a scroll sees is exactly the live set
    let mut seen: Vec<u64> = d.scroll(None, 0, 10_000, false).items.iter().map(|r| r.id).collect();
    seen.sort();
    let mut want: Vec<u64> = model.keys().copied().collect();
    want.sort();
    assert_eq!(seen, want);
}

#[test]
fn matches_the_model_f32() {
    for seed in 1..=4 {
        run(seed, Quantization::None, 400);
    }
}

#[test]
fn matches_the_model_sq8() {
    run(77, Quantization::Sq8, 400);
}

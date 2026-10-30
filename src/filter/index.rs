use std::collections::HashMap;

use parking_lot::RwLock;
use roaring::RoaringBitmap;

use super::{Filter, Payload, Value};

#[derive(Hash, PartialEq, Eq)]
enum Key {
    Bool(bool),
    Int(i64),
    Str(String),
}

impl Key {
    fn to_value(&self) -> Value {
        match self {
            Key::Bool(b) => Value::Bool(*b),
            Key::Int(i) => Value::Int(*i),
            Key::Str(s) => Value::Str(s.clone()),
        }
    }

    // floats are not hashable, so they stay out of the inverted index
    fn of(v: &Value) -> Option<Key> {
        match v {
            Value::Bool(b) => Some(Key::Bool(*b)),
            Value::Int(i) => Some(Key::Int(*i)),
            Value::Str(s) => Some(Key::Str(s.clone())),
            Value::Float(_) => None,
        }
    }
}

type Map = HashMap<(String, Key), RoaringBitmap>;

#[derive(Default)]
pub(crate) struct PayloadIndex {
    map: RwLock<Map>,
}

impl PayloadIndex {
    pub fn add(&self, id: u32, payload: &Payload) {
        let mut m = self.map.write();
        for (k, v) in payload {
            if let Some(key) = Key::of(v) {
                m.entry((k.clone(), key)).or_default().insert(id);
            }
        }
    }

    pub fn remove(&self, id: u32, payload: &Payload) {
        let mut m = self.map.write();
        for (k, v) in payload {
            if let Some(b) = Key::of(v).and_then(|key| m.get_mut(&(k.clone(), key))) {
                b.remove(id);
            }
        }
    }

    /// every indexed value of `field` with the ids that hold it
    pub fn values_of(&self, field: &str) -> Vec<(Value, RoaringBitmap)> {
        let m = self.map.read();
        m.iter()
            .filter(|((f, _), ids)| f == field && !ids.is_empty())
            .map(|((_, k), ids)| (k.to_value(), ids.clone()))
            .collect()
    }

    /// Ids that can satisfy `f`, and whether that set is exactly the answer.
    /// `None` means the filter can't be answered from the inverted index.
    pub fn plan(&self, f: &Filter) -> Option<(RoaringBitmap, bool)> {
        plan(&self.map.read(), f)
    }
}

fn lookup(m: &Map, field: &str, v: &Value) -> Option<RoaringBitmap> {
    let key = Key::of(v)?;
    Some(m.get(&(field.to_owned(), key)).cloned().unwrap_or_default())
}

fn plan(m: &Map, f: &Filter) -> Option<(RoaringBitmap, bool)> {
    match f {
        Filter::Eq(k, v) => Some((lookup(m, k, v)?, true)),
        Filter::AnyOf(k, vs) => {
            let mut out = RoaringBitmap::new();
            for v in vs {
                out |= lookup(m, k, v)?;
            }
            Some((out, true))
        }
        Filter::And(fs) => {
            // children we can't plan only loosen the bound, they get re-checked later
            let mut acc: Option<RoaringBitmap> = None;
            let mut exact = true;
            for c in fs {
                match plan(m, c) {
                    Some((b, e)) => {
                        exact &= e;
                        acc = Some(match acc {
                            Some(a) => a & b,
                            None => b,
                        });
                    }
                    None => exact = false,
                }
            }
            acc.map(|a| (a, exact))
        }
        Filter::Or(fs) => {
            let mut out = RoaringBitmap::new();
            let mut exact = true;
            for c in fs {
                let (b, e) = plan(m, c)?;
                out |= b;
                exact &= e;
            }
            Some((out, exact))
        }
        Filter::Range { .. } | Filter::Not(_) => None,
    }
}

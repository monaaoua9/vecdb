use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

mod index;
pub(crate) use index::PayloadIndex;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

impl Value {
    fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::Str(s.to_owned())
    }
}
impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Value::Int(i)
    }
}
impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}
impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Value::Float(f)
    }
}

pub type Payload = BTreeMap<String, Value>;

/// Equality is strict on the variant: `Int(1)` does not match `Float(1.0)`.
/// `Range` is the exception and compares any numeric as f64.
#[derive(Clone, Debug, PartialEq)]
pub enum Filter {
    Eq(String, Value),
    AnyOf(String, Vec<Value>),
    Range { field: String, min: Option<f64>, max: Option<f64> },
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

impl Filter {
    pub fn eq(field: &str, v: impl Into<Value>) -> Filter {
        Filter::Eq(field.to_owned(), v.into())
    }

    pub fn range(field: &str, min: Option<f64>, max: Option<f64>) -> Filter {
        Filter::Range { field: field.to_owned(), min, max }
    }

    pub fn matches(&self, p: &Payload) -> bool {
        match self {
            Filter::Eq(k, v) => p.get(k) == Some(v),
            Filter::AnyOf(k, vs) => p.get(k).is_some_and(|x| vs.contains(x)),
            Filter::Range { field, min, max } => {
                let Some(x) = p.get(field).and_then(Value::as_f64) else {
                    return false;
                };
                min.is_none_or(|m| x >= m) && max.is_none_or(|m| x <= m)
            }
            Filter::And(fs) => fs.iter().all(|f| f.matches(p)),
            Filter::Or(fs) => fs.iter().any(|f| f.matches(p)),
            Filter::Not(f) => !f.matches(p),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FilterStrategy {
    /// planner picks: exact scan for selective filters, filtered graph walk otherwise
    #[default]
    Auto,
    /// resolve the filter first, then rank only the matching vectors exactly
    PreFilter,
    /// plain ANN search, drop non-matching hits, over-fetch until k survive
    PostFilter,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(pairs: &[(&str, Value)]) -> Payload {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn basic_matching() {
        let doc = p(&[("lang", "en".into()), ("year", 2021.into()), ("score", 0.8.into())]);
        assert!(Filter::eq("lang", "en").matches(&doc));
        assert!(!Filter::eq("lang", "fr").matches(&doc));
        assert!(Filter::range("year", Some(2020.0), None).matches(&doc));
        assert!(Filter::range("score", None, Some(0.9)).matches(&doc));
        assert!(!Filter::range("missing", None, None).matches(&doc));
        assert!(Filter::Not(Box::new(Filter::eq("lang", "fr"))).matches(&doc));
        let f = Filter::And(vec![
            Filter::eq("lang", "en"),
            Filter::Or(vec![Filter::eq("year", 1999), Filter::eq("year", 2021)]),
        ]);
        assert!(f.matches(&doc));
    }

    #[test]
    fn eq_is_variant_strict() {
        let doc = p(&[("n", 1.into())]);
        assert!(!Filter::Eq("n".into(), Value::Float(1.0)).matches(&doc));
    }
}

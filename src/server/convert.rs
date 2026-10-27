use tonic::Status;

use super::pb;
use crate::error::Error;
use crate::filter::{Filter, FilterStrategy, Payload, Value};

const MAX_FILTER_DEPTH: usize = 16;

impl From<Error> for Status {
    fn from(e: Error) -> Status {
        let msg = e.to_string();
        match e {
            Error::DimensionMismatch { .. } | Error::InvalidVector(_) => Status::invalid_argument(msg),
            Error::DuplicateId(_) => Status::already_exists(msg),
            Error::NotFound(_) => Status::not_found(msg),
            Error::CapacityExceeded(_) => Status::resource_exhausted(msg),
            Error::InvalidConfig(_) => Status::failed_precondition(msg),
            Error::Corrupt(_) | Error::Io(_) | Error::Codec(_) => Status::internal(msg),
        }
    }
}

pub fn value_in(v: pb::Value) -> Result<Value, Status> {
    use pb::value::Kind;
    match v.kind {
        Some(Kind::BoolValue(b)) => Ok(Value::Bool(b)),
        Some(Kind::IntValue(i)) => Ok(Value::Int(i)),
        Some(Kind::FloatValue(f)) => Ok(Value::Float(f)),
        Some(Kind::StringValue(s)) => Ok(Value::Str(s)),
        None => Err(Status::invalid_argument("value has no kind set")),
    }
}

pub fn value_out(v: &Value) -> pb::Value {
    use pb::value::Kind;
    pb::Value {
        kind: Some(match v {
            Value::Bool(b) => Kind::BoolValue(*b),
            Value::Int(i) => Kind::IntValue(*i),
            Value::Float(f) => Kind::FloatValue(*f),
            Value::Str(s) => Kind::StringValue(s.clone()),
        }),
    }
}

pub fn payload_in(p: Option<pb::Payload>) -> Result<Payload, Status> {
    p.map(|p| p.fields.into_iter().map(|(k, v)| Ok((k, value_in(v)?))).collect())
        .unwrap_or_else(|| Ok(Payload::new()))
}

pub fn payload_out(p: &Payload) -> pb::Payload {
    pb::Payload { fields: p.iter().map(|(k, v)| (k.clone(), value_out(v))).collect() }
}

pub fn filter_in(f: pb::Filter) -> Result<Filter, Status> {
    filter_depth(f, 0)
}

fn filter_depth(f: pb::Filter, depth: usize) -> Result<Filter, Status> {
    use pb::filter::Kind;
    if depth > MAX_FILTER_DEPTH {
        return Err(Status::invalid_argument("filter nesting is too deep"));
    }
    let list = |l: pb::FilterList| -> Result<Vec<Filter>, Status> {
        l.filters.into_iter().map(|c| filter_depth(c, depth + 1)).collect()
    };
    match f.kind.ok_or_else(|| Status::invalid_argument("filter has no kind set"))? {
        Kind::Match(m) => {
            let v = m.value.ok_or_else(|| Status::invalid_argument("match needs a value"))?;
            Ok(Filter::Eq(m.field, value_in(v)?))
        }
        Kind::MatchAny(m) => {
            let vs = m.values.into_iter().map(value_in).collect::<Result<_, _>>()?;
            Ok(Filter::AnyOf(m.field, vs))
        }
        Kind::Range(r) => Ok(Filter::Range { field: r.field, min: r.min, max: r.max }),
        Kind::And(l) => Ok(Filter::And(list(l)?)),
        Kind::Or(l) => Ok(Filter::Or(list(l)?)),
        Kind::Not(inner) => Ok(Filter::Not(Box::new(filter_depth(*inner, depth + 1)?))),
    }
}

pub fn strategy_in(raw: i32) -> FilterStrategy {
    match pb::FilterStrategy::try_from(raw) {
        Ok(pb::FilterStrategy::Pre) => FilterStrategy::PreFilter,
        Ok(pb::FilterStrategy::Post) => FilterStrategy::PostFilter,
        _ => FilterStrategy::Auto,
    }
}

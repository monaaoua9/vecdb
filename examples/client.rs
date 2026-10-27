//! Talks to a running server:
//!
//!   cargo run --release --bin vecdb-server -- --dim 8 --metric l2 --data-dir ./data
//!   cargo run --example client

use std::collections::HashMap;

use vecdb::server::pb::{self, vector_db_client::VectorDbClient};

fn text(s: &str) -> pb::Value {
    pb::Value { kind: Some(pb::value::Kind::StringValue(s.into())) }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut c = VectorDbClient::connect("http://127.0.0.1:50051").await?;

    for i in 0..100u64 {
        let lang = if i % 3 == 0 { "en" } else { "ar" };
        c.insert(pb::InsertRequest {
            id: i,
            vector: (0..8).map(|d| ((i * 7 + d) % 13) as f32).collect(),
            payload: Some(pb::Payload { fields: HashMap::from([("lang".to_string(), text(lang))]) }),
        })
        .await?;
    }

    let only_ar = pb::Filter {
        kind: Some(pb::filter::Kind::Match(pb::FieldMatch { field: "lang".into(), value: Some(text("ar")) })),
    };
    let r = c
        .search(pb::SearchRequest {
            vector: vec![3.0; 8],
            k: 5,
            filter: Some(only_ar),
            with_payload: true,
            ..Default::default()
        })
        .await?
        .into_inner();

    for h in r.hits {
        println!("id={:<3} distance={:.3}", h.id, h.distance);
    }
    println!("took {} us", r.took_micros);
    Ok(())
}

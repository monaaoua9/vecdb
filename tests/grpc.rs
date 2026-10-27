#![cfg(feature = "server")]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::oneshot;
use vecdb::server;
use vecdb::server::pb::{self, vector_db_client::VectorDbClient};
use vecdb::{Config, Durability, DurableIndex, Index, Metric, snapshots};

fn free_addr() -> std::net::SocketAddr {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap()
}

fn val_int(i: i64) -> pb::Value {
    pb::Value { kind: Some(pb::value::Kind::IntValue(i)) }
}

fn val_str(s: &str) -> pb::Value {
    pb::Value { kind: Some(pb::value::Kind::StringValue(s.into())) }
}

async fn connect(addr: std::net::SocketAddr) -> VectorDbClient<tonic::transport::Channel> {
    for _ in 0..100 {
        if let Ok(c) = VectorDbClient::connect(format!("http://{addr}")).await {
            return c;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("server did not come up");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = Config::new(4, Metric::L2);
    cfg.max_elements = 1000;
    let data_dir = dir.path().to_path_buf();
    let index = Arc::new(DurableIndex::open(&data_dir, cfg, Durability::Flush).unwrap());
    let addr = free_addr();
    let (stop_tx, stop_rx) = oneshot::channel::<()>();

    let srv = tokio::spawn(server::serve(index.clone(), addr, Duration::ZERO, async {
        let _ = stop_rx.await;
    }));
    let mut c = connect(addr).await;

    for i in 0..50u64 {
        let payload = pb::Payload {
            fields: HashMap::from([
                ("parity".to_string(), val_int((i % 2) as i64)),
                ("name".to_string(), val_str(&format!("doc-{i}"))),
            ]),
        };
        c.insert(pb::InsertRequest { id: i, vector: vec![i as f32, 0.0, 0.0, 1.0], payload: Some(payload) })
            .await
            .unwrap();
    }

    // streaming bulk load
    let extra = (100..300u64).map(|i| pb::InsertRequest {
        id: i,
        vector: vec![i as f32, 1.0, 0.0, 0.0],
        payload: None,
    });
    let resp = c.insert_stream(tokio_stream::iter(extra)).await.unwrap().into_inner();
    assert_eq!(resp.inserted, 200);

    let q = vec![7.2, 0.0, 0.0, 1.0];
    let r = c
        .search(pb::SearchRequest { vector: q.clone(), k: 3, with_payload: true, ..Default::default() })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(r.hits[0].id, 7);
    assert_eq!(r.hits.len(), 3);
    assert!(r.hits[0].payload.as_ref().unwrap().fields.contains_key("name"));

    // metadata filter: odd ids only
    let filter = pb::Filter {
        kind: Some(pb::filter::Kind::Match(pb::FieldMatch {
            field: "parity".into(),
            value: Some(val_int(1)),
        })),
    };
    let r = c
        .search(pb::SearchRequest { vector: q.clone(), k: 4, filter: Some(filter), ..Default::default() })
        .await
        .unwrap()
        .into_inner();
    assert!(!r.hits.is_empty());
    assert!(r.hits.iter().all(|h| h.id % 2 == 1 && h.id < 50));

    let status =
        c.search(pb::SearchRequest { vector: vec![1.0], k: 3, ..Default::default() }).await.unwrap_err();
    assert_eq!(status.code(), tonic::Code::InvalidArgument);

    let status =
        c.insert(pb::InsertRequest { id: 3, vector: vec![0.0; 4], payload: None }).await.unwrap_err();
    assert_eq!(status.code(), tonic::Code::AlreadyExists);

    assert_eq!(c.get(pb::GetRequest { id: 9 }).await.unwrap().into_inner().vector[0], 9.0);
    c.delete(pb::DeleteRequest { id: 9 }).await.unwrap();
    assert_eq!(c.get(pb::GetRequest { id: 9 }).await.unwrap_err().code(), tonic::Code::NotFound);

    let st = c.stats(pb::StatsRequest {}).await.unwrap().into_inner();
    assert_eq!(st.live, 249);
    assert_eq!(st.dim, 4);
    assert!(!st.simd_kernel.is_empty());

    let snap = c.snapshot(pb::SnapshotRequest {}).await.unwrap().into_inner();
    assert_eq!(snap.vectors, 249);
    assert!(std::path::Path::new(&snap.path).exists());

    stop_tx.send(()).unwrap();
    srv.await.unwrap().unwrap();

    // restart from what the server left on disk
    let (generation, path) = snapshots::latest(&data_dir).unwrap();
    assert!(generation >= 2, "shutdown should have written another snapshot");
    let back = Index::open(path).unwrap();
    assert_eq!(back.len(), 249);
    assert_eq!(back.search(&q, 1).unwrap()[0].id, 7);
}

#[tokio::test]
async fn snapshot_without_data_dir_is_rejected() {
    let index = Arc::new(DurableIndex::in_memory(Index::new(Config::new(2, Metric::L2)).unwrap()));
    let addr = free_addr();
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let srv = tokio::spawn(server::serve(index, addr, Duration::ZERO, async {
        let _ = stop_rx.await;
    }));
    let mut c = connect(addr).await;
    let err = c.snapshot(pb::SnapshotRequest {}).await.unwrap_err();
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    stop_tx.send(()).unwrap();
    srv.await.unwrap().unwrap();
}

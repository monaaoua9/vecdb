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

#[tokio::test]
async fn upsert_and_payload_updates_over_grpc() {
    let index = Arc::new(DurableIndex::in_memory(Index::new(Config::new(2, Metric::L2)).unwrap()));
    let addr = free_addr();
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let srv = tokio::spawn(server::serve(index, addr, Duration::ZERO, async {
        let _ = stop_rx.await;
    }));
    let mut c = connect(addr).await;
    let fields = |k: &str, v: i64| pb::Payload { fields: HashMap::from([(k.to_string(), val_int(v))]) };

    c.upsert(pb::InsertRequest { id: 1, vector: vec![1.0, 1.0], payload: Some(fields("a", 1)) }).await.unwrap();
    c.upsert(pb::InsertRequest { id: 1, vector: vec![2.0, 2.0], payload: Some(fields("a", 2)) }).await.unwrap();
    let g = c.get(pb::GetRequest { id: 1 }).await.unwrap().into_inner();
    assert_eq!(g.vector, vec![2.0, 2.0]);

    c.update_payload(pb::UpdatePayloadRequest { id: 1, payload: Some(fields("b", 3)), merge: true }).await.unwrap();
    let p = c.get(pb::GetRequest { id: 1 }).await.unwrap().into_inner().payload.unwrap();
    assert_eq!(p.fields.len(), 2);

    c.update_payload(pb::UpdatePayloadRequest { id: 1, payload: Some(fields("c", 4)), merge: false }).await.unwrap();
    let p = c.get(pb::GetRequest { id: 1 }).await.unwrap().into_inner().payload.unwrap();
    assert_eq!(p.fields.len(), 1);

    let err = c
        .update_payload(pb::UpdatePayloadRequest { id: 99, payload: None, merge: true })
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::NotFound);

    stop_tx.send(()).unwrap();
    srv.await.unwrap().unwrap();
}

fn to<T>(msg: T, collection: &str) -> tonic::Request<T> {
    let mut r = tonic::Request::new(msg);
    r.metadata_mut().insert("x-collection", collection.parse().unwrap());
    r
}

#[tokio::test]
async fn collections_over_grpc() {
    let dir = tempfile::tempdir().unwrap();
    let index = Arc::new(DurableIndex::open(dir.path(), Config::new(4, Metric::L2), Durability::Flush).unwrap());
    let addr = free_addr();
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let srv = tokio::spawn(server::serve(index, addr, Duration::ZERO, async {
        let _ = stop_rx.await;
    }));
    let mut c = connect(addr).await;

    let cfg = pb::CollectionConfig {
        dim: 2,
        metric: pb::MetricKind::Cosine as i32,
        max_elements: 100,
        ..Default::default()
    };
    c.create_collection(pb::CreateCollectionRequest { name: "docs".into(), config: Some(cfg.clone()) })
        .await
        .unwrap();
    let dup = c
        .create_collection(pb::CreateCollectionRequest { name: "docs".into(), config: Some(cfg.clone()) })
        .await
        .unwrap_err();
    assert_eq!(dup.code(), tonic::Code::AlreadyExists);
    let bad = c
        .create_collection(pb::CreateCollectionRequest { name: "no way".into(), config: Some(cfg) })
        .await
        .unwrap_err();
    assert_eq!(bad.code(), tonic::Code::InvalidArgument);

    // same id in both collections, different dimensions
    c.insert(pb::InsertRequest { id: 1, vector: vec![1.0, 0.0], payload: None }).await.unwrap_err();
    c.insert(to(pb::InsertRequest { id: 1, vector: vec![1.0, 0.0], payload: None }, "docs")).await.unwrap();
    c.insert(pb::InsertRequest { id: 1, vector: vec![0.0; 4], payload: None }).await.unwrap();

    let a = c.get(to(pb::GetRequest { id: 1 }, "docs")).await.unwrap().into_inner();
    let b = c.get(pb::GetRequest { id: 1 }).await.unwrap().into_inner();
    assert_eq!((a.vector.len(), b.vector.len()), (2, 4));

    let missing = c.get(to(pb::GetRequest { id: 1 }, "nope")).await.unwrap_err();
    assert_eq!(missing.code(), tonic::Code::NotFound);

    let list = c.list_collections(pb::ListCollectionsRequest {}).await.unwrap().into_inner().collections;
    let names: Vec<_> = list.iter().map(|i| (i.name.as_str(), i.dim, i.live)).collect();
    assert_eq!(names, [("default", 4, 1), ("docs", 2, 1)]);

    c.drop_collection(pb::DropCollectionRequest { name: "docs".into() }).await.unwrap();
    assert_eq!(
        c.get(to(pb::GetRequest { id: 1 }, "docs")).await.unwrap_err().code(),
        tonic::Code::NotFound
    );
    let default = c.drop_collection(pb::DropCollectionRequest { name: "default".into() }).await.unwrap_err();
    assert_eq!(default.code(), tonic::Code::InvalidArgument);

    stop_tx.send(()).unwrap();
    srv.await.unwrap().unwrap();
}

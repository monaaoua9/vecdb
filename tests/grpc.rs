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

    c.upsert(pb::InsertRequest { id: 1, vector: vec![1.0, 1.0], payload: Some(fields("a", 1)) })
        .await
        .unwrap();
    c.upsert(pb::InsertRequest { id: 1, vector: vec![2.0, 2.0], payload: Some(fields("a", 2)) })
        .await
        .unwrap();
    let g = c.get(pb::GetRequest { id: 1 }).await.unwrap().into_inner();
    assert_eq!(g.vector, vec![2.0, 2.0]);

    c.update_payload(pb::UpdatePayloadRequest { id: 1, payload: Some(fields("b", 3)), merge: true })
        .await
        .unwrap();
    let p = c.get(pb::GetRequest { id: 1 }).await.unwrap().into_inner().payload.unwrap();
    assert_eq!(p.fields.len(), 2);

    c.update_payload(pb::UpdatePayloadRequest { id: 1, payload: Some(fields("c", 4)), merge: false })
        .await
        .unwrap();
    let p = c.get(pb::GetRequest { id: 1 }).await.unwrap().into_inner().payload.unwrap();
    assert_eq!(p.fields.len(), 1);

    let err =
        c.update_payload(pb::UpdatePayloadRequest { id: 99, payload: None, merge: true }).await.unwrap_err();
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
    let index =
        Arc::new(DurableIndex::open(dir.path(), Config::new(4, Metric::L2), Durability::Flush).unwrap());
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
    assert_eq!(c.get(to(pb::GetRequest { id: 1 }, "docs")).await.unwrap_err().code(), tonic::Code::NotFound);
    let default = c.drop_collection(pb::DropCollectionRequest { name: "default".into() }).await.unwrap_err();
    assert_eq!(default.code(), tonic::Code::InvalidArgument);

    stop_tx.send(()).unwrap();
    srv.await.unwrap().unwrap();
}

#[tokio::test]
async fn batch_range_mmr_and_recommend_over_grpc() {
    let mut cfg = Config::new(2, Metric::L2);
    cfg.max_elements = 500;
    let index = Arc::new(DurableIndex::in_memory(Index::new(cfg).unwrap()));
    for i in 0..100u64 {
        index.insert(i, &[i as f32, 0.0], Default::default()).unwrap();
    }
    let addr = free_addr();
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let srv = tokio::spawn(server::serve(index, addr, Duration::ZERO, async {
        let _ = stop_rx.await;
    }));
    let mut c = connect(addr).await;
    let q = |x: f32| pb::SearchRequest { vector: vec![x, 0.0], k: 2, ..Default::default() };

    let batch = c
        .search_batch(pb::BatchSearchRequest { queries: vec![q(10.0), q(50.2), q(90.0)] })
        .await
        .unwrap()
        .into_inner();
    let firsts: Vec<u64> = batch.results.iter().map(|r| r.hits[0].id).collect();
    assert_eq!(firsts, [10, 50, 90]);

    let too_many = pb::BatchSearchRequest { queries: vec![q(1.0); 257] };
    assert_eq!(c.search_batch(too_many).await.unwrap_err().code(), tonic::Code::InvalidArgument);

    let ranged =
        pb::SearchRequest { vector: vec![40.0, 0.0], k: 50, radius: Some(4.0), ..Default::default() };
    assert_eq!(c.search(ranged).await.unwrap().into_inner().hits.len(), 5);

    let both = pb::SearchRequest { radius: Some(1.0), mmr_lambda: Some(0.5), ..q(1.0) };
    assert_eq!(c.search(both).await.unwrap_err().code(), tonic::Code::InvalidArgument);

    let mmr = pb::SearchRequest { mmr_lambda: Some(0.2), mmr_fetch: 20, k: 3, ..q(10.0) };
    let ids: Vec<u64> = c.search(mmr).await.unwrap().into_inner().hits.iter().map(|h| h.id).collect();
    assert_eq!(ids[0], 10);
    assert!(ids.iter().any(|&i| i.abs_diff(10) >= 3), "{ids:?}");

    let rec = pb::RecommendRequest { positive: vec![50], negative: vec![40], k: 1, ..Default::default() };
    assert_eq!(c.recommend(rec).await.unwrap().into_inner().hits[0].id, 60);

    stop_tx.send(()).unwrap();
    srv.await.unwrap().unwrap();
}

#[tokio::test]
async fn count_scroll_and_facets_over_grpc() {
    let mut cfg = Config::new(2, Metric::L2);
    cfg.max_elements = 500;
    let index = Arc::new(DurableIndex::in_memory(Index::new(cfg).unwrap()));
    for i in 0..60u64 {
        let mut p = vecdb::Payload::new();
        p.insert("kind".into(), vecdb::Value::Str(if i % 3 == 0 { "a" } else { "b" }.into()));
        index.insert(i, &[i as f32, 0.0], p).unwrap();
    }
    let addr = free_addr();
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let srv = tokio::spawn(server::serve(index, addr, Duration::ZERO, async {
        let _ = stop_rx.await;
    }));
    let mut c = connect(addr).await;
    let only_a = || pb::Filter {
        kind: Some(pb::filter::Kind::Match(pb::FieldMatch {
            field: "kind".into(),
            value: Some(val_str("a")),
        })),
    };

    assert_eq!(c.count(pb::CountRequest { filter: None }).await.unwrap().into_inner().count, 60);
    assert_eq!(c.count(pb::CountRequest { filter: Some(only_a()) }).await.unwrap().into_inner().count, 20);

    let (mut seen, mut cursor) = (vec![], 0);
    loop {
        let r = c
            .scroll(pb::ScrollRequest {
                filter: Some(only_a()),
                cursor,
                limit: 7,
                with_payload: true,
                ..Default::default()
            })
            .await
            .unwrap()
            .into_inner();
        assert!(r.items.iter().all(|i| i.payload.is_some() && i.vector.is_empty()));
        seen.extend(r.items.iter().map(|i| i.id));
        match r.next_cursor {
            Some(n) => cursor = n,
            None => break,
        }
    }
    assert_eq!(seen, (0..60).step_by(3).collect::<Vec<_>>());

    let too_big = c.scroll(pb::ScrollRequest { limit: 10_001, ..Default::default() }).await.unwrap_err();
    assert_eq!(too_big.code(), tonic::Code::InvalidArgument);

    let f = c
        .facets(pb::FacetsRequest { field: "kind".into(), ..Default::default() })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(f.facets.len(), 2);
    assert_eq!(f.facets[0].count, 40);
    assert_eq!(f.facets[1].count, 20);

    stop_tx.send(()).unwrap();
    srv.await.unwrap().unwrap();
}

#[tokio::test]
async fn text_and_hybrid_search_over_grpc() {
    let mut cfg = Config::new(2, Metric::L2);
    cfg.max_elements = 50;
    let index = Arc::new(DurableIndex::in_memory(Index::new(cfg).unwrap()));
    let doc = |t: &str| {
        let mut p = vecdb::Payload::new();
        p.insert("body".into(), vecdb::Value::Str(t.into()));
        p
    };
    index.insert(0, &[0.0, 0.0], doc("alpha beta")).unwrap();
    index.insert(1, &[1.0, 0.0], doc("gamma")).unwrap();
    index.insert(2, &[90.0, 0.0], doc("delta epsilon")).unwrap();

    let cols = Arc::new(vecdb::Collections::new(index));
    cols.set_text_field("body");
    let addr = free_addr();
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let srv = tokio::spawn(server::serve_collections(cols, addr, Duration::ZERO, async {
        let _ = stop_rx.await;
    }));
    let mut c = connect(addr).await;

    let text_only = pb::SearchRequest { text: Some("epsilon".into()), k: 3, ..Default::default() };
    let hits = c.search(text_only).await.unwrap().into_inner().hits;
    assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), [2]);

    let hybrid = pb::SearchRequest {
        vector: vec![0.0, 0.0],
        text: Some("epsilon".into()),
        k: 3,
        ..Default::default()
    };
    let ids: Vec<u64> = c.search(hybrid).await.unwrap().into_inner().hits.iter().map(|h| h.id).collect();
    assert!(ids.contains(&2) && ids.contains(&0));

    let bad = pb::SearchRequest { text: Some("x".into()), radius: Some(1.0), k: 1, ..Default::default() };
    assert_eq!(c.search(bad).await.unwrap_err().code(), tonic::Code::InvalidArgument);

    stop_tx.send(()).unwrap();
    srv.await.unwrap().unwrap();
}

async fn plain_server(opts: server::ServeOptions) -> (std::net::SocketAddr, oneshot::Sender<()>) {
    let mut cfg = Config::new(2, Metric::L2);
    cfg.max_elements = 50;
    let index = Arc::new(DurableIndex::in_memory(Index::new(cfg).unwrap()));
    let cols = Arc::new(vecdb::Collections::new(index));
    let addr = free_addr();
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(server::serve_with(cols, addr, Duration::ZERO, opts, async {
        let _ = rx.await;
    }));
    (addr, tx)
}

#[tokio::test]
async fn api_keys_gate_the_data_rpcs() {
    let keys = server::ApiKeys::new(["s3cret".to_string()]);
    let (addr, stop) =
        plain_server(server::ServeOptions { api_keys: Some(keys), ..Default::default() }).await;
    let mut c = connect(addr).await;
    let ins = || pb::InsertRequest { id: 1, vector: vec![1.0, 2.0], payload: None };

    assert_eq!(c.insert(ins()).await.unwrap_err().code(), tonic::Code::Unauthenticated);

    let mut bad = tonic::Request::new(ins());
    bad.metadata_mut().insert("authorization", "Bearer nope".parse().unwrap());
    assert_eq!(c.insert(bad).await.unwrap_err().code(), tonic::Code::Unauthenticated);

    let mut good = tonic::Request::new(ins());
    good.metadata_mut().insert("authorization", "Bearer s3cret".parse().unwrap());
    c.insert(good).await.unwrap();

    // health checks stay reachable for load balancers
    let chan =
        tonic::transport::Channel::from_shared(format!("http://{addr}")).unwrap().connect().await.unwrap();
    let mut h = tonic_health::pb::health_client::HealthClient::new(chan);
    h.check(tonic_health::pb::HealthCheckRequest { service: String::new() }).await.unwrap();

    let _ = stop.send(());
}

#[tokio::test]
async fn metrics_endpoint_counts_requests() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let maddr = free_addr();
    let (addr, stop) =
        plain_server(server::ServeOptions { metrics_addr: Some(maddr), ..Default::default() }).await;
    let mut c = connect(addr).await;
    for i in 0..3u64 {
        c.insert(pb::InsertRequest { id: i, vector: vec![i as f32, 0.0], payload: None }).await.unwrap();
    }
    c.insert(pb::InsertRequest { id: 1, vector: vec![0.0, 0.0], payload: None }).await.unwrap_err();
    c.search(pb::SearchRequest { vector: vec![0.0, 0.0], k: 1, ..Default::default() }).await.unwrap();

    let mut body = String::new();
    for _ in 0..100 {
        if let Ok(mut s) = tokio::net::TcpStream::connect(maddr).await {
            s.write_all(b"GET /metrics HTTP/1.1\r\nHost: x\r\n\r\n").await.unwrap();
            s.read_to_string(&mut body).await.unwrap();
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(body.starts_with("HTTP/1.1 200"), "{body}");
    assert!(body.contains("vecdb_rpc_requests_total{rpc=\"Insert\",result=\"ok\"} 3"), "{body}");
    assert!(body.contains("vecdb_rpc_requests_total{rpc=\"Insert\",result=\"error\"} 1"), "{body}");
    assert!(body.contains("vecdb_rpc_duration_seconds_count{rpc=\"Search\"} 1"), "{body}");
    assert!(body.contains("vecdb_vectors{collection=\"default\"} 3"), "{body}");

    let _ = stop.send(());
}

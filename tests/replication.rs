#![cfg(feature = "server")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::oneshot;
use vecdb::server::pb::{self, replication_message::Kind, vector_db_client::VectorDbClient};
use vecdb::server::{self, ServeOptions};
use vecdb::{Collections, Config, Durability, DurableIndex, Index, Metric, Payload, Value};

fn free_addr() -> std::net::SocketAddr {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap()
}

fn cfg() -> Config {
    let mut c = Config::new(3, Metric::L2);
    c.max_elements = 20_000;
    c
}

fn doc(n: i64) -> Payload {
    let mut p = Payload::new();
    p.insert("n".into(), Value::Int(n));
    p.insert("tag".into(), Value::Str(format!("t{}", n % 4)));
    p
}

struct Leader {
    cols: Arc<Collections>,
    default: Arc<DurableIndex>,
    addr: std::net::SocketAddr,
    stop: oneshot::Sender<()>,
    _dir: tempfile::TempDir,
}

fn start_leader() -> Leader {
    let dir = tempfile::tempdir().unwrap();
    let default = Arc::new(DurableIndex::open(dir.path(), cfg(), Durability::Flush).unwrap());
    let cols = Arc::new(Collections::new(default.clone()));
    let addr = free_addr();
    let (stop, rx) = oneshot::channel::<()>();
    tokio::spawn(server::serve_with(cols.clone(), addr, Duration::ZERO, ServeOptions::default(), async {
        let _ = rx.await;
    }));
    Leader { cols, default, addr, stop, _dir: dir }
}

fn start_replica(leader: std::net::SocketAddr) -> (Arc<Collections>, Arc<AtomicBool>) {
    let placeholder = Arc::new(DurableIndex::in_memory(Index::new(Config::new(1, Metric::L2)).unwrap()));
    let cols = Arc::new(Collections::new(placeholder));
    let stop = Arc::new(AtomicBool::new(false));
    tokio::spawn(server::follow_every(
        cols.clone(),
        format!("http://{leader}"),
        None,
        stop.clone(),
        Duration::from_millis(200),
    ));
    (cols, stop)
}

fn same(a: &DurableIndex, b: &DurableIndex, ids: std::ops::Range<u64>) -> bool {
    a.len() == b.len()
        && ids.into_iter().all(|id| match (a.get(id), b.get(id)) {
            (None, None) => true,
            (Some(x), Some(y)) => x.vector == y.vector && x.payload == y.payload,
            _ => false,
        })
}

async fn until(what: &str, mut ok: impl FnMut() -> bool) {
    let t0 = Instant::now();
    while !ok() {
        assert!(t0.elapsed() < Duration::from_secs(20), "timed out waiting for: {what}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn client(addr: std::net::SocketAddr) -> VectorDbClient<tonic::transport::Channel> {
    loop {
        if let Ok(c) = VectorDbClient::connect(format!("http://{addr}")).await {
            return c;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replica_catches_up_and_keeps_following() {
    let l = start_leader();
    let d = &l.default;
    for i in 0..100u64 {
        d.insert(i, &[i as f32, 1.0, 0.0], doc(i as i64)).unwrap();
    }
    d.checkpoint().unwrap();
    for i in 100..150u64 {
        d.insert(i, &[i as f32, 1.0, 0.0], doc(i as i64)).unwrap();
    }
    d.delete(3).unwrap();
    d.upsert(5, &[500.0, 5.0, 5.0], doc(555)).unwrap();
    d.set_payload(7, doc(777)).unwrap();
    d.merge_payload(8, doc(888)).unwrap();

    let (rcols, stop) = start_replica(l.addr);
    let r = rcols.get("default").unwrap();
    until("initial catch-up", || same(d, &r, 0..400)).await;
    assert_eq!(r.config().dim, 3, "replica adopts the leader's config");

    // live writes
    for i in 150..170u64 {
        d.insert(i, &[i as f32, 1.0, 0.0], doc(i as i64)).unwrap();
    }
    d.delete(10).unwrap();
    until("live writes", || same(d, &r, 0..400)).await;

    // a checkpoint rotates the log; the replica must carry on into the next one
    d.checkpoint().unwrap();
    for i in 170..190u64 {
        d.insert(i, &[i as f32, 1.0, 0.0], doc(i as i64)).unwrap();
    }
    until("writes after a log rotation", || same(d, &r, 0..400)).await;

    // searches work on the replica and agree with the leader
    let q = [42.0, 1.0, 0.0];
    assert_eq!(r.search(&q, 5).unwrap(), d.search(&q, 5).unwrap());

    // a collection created on the leader later shows up too
    let mut c2 = Config::new(2, Metric::Cosine);
    c2.max_elements = 100;
    let docs = l.cols.create("docs", c2).unwrap();
    docs.insert(1, &[1.0, 0.0], doc(1)).unwrap();
    until("new collection", || rcols.get("docs").is_some_and(|c| c.len() == 1)).await;
    docs.insert(2, &[0.0, 1.0], doc(2)).unwrap();
    until("write into it", || rcols.get("docs").is_some_and(|c| c.len() == 2)).await;

    stop.store(true, Ordering::Relaxed);
    let _ = l.stop.send(());
}

#[tokio::test]
async fn replica_server_refuses_writes_but_serves_reads() {
    let idx = Arc::new(DurableIndex::in_memory(Index::new(Config::new(2, Metric::L2)).unwrap()));
    idx.insert(1, &[1.0, 1.0], Payload::new()).unwrap();
    let cols = Arc::new(Collections::new(idx));
    let addr = free_addr();
    let (stop, rx) = oneshot::channel::<()>();
    let opts = ServeOptions { read_only: true, ..Default::default() };
    tokio::spawn(server::serve_with(cols, addr, Duration::ZERO, opts, async {
        let _ = rx.await;
    }));
    let mut c = client(addr).await;

    let ins = || pb::InsertRequest { id: 2, vector: vec![0.0, 0.0], payload: None };
    assert_eq!(c.insert(ins()).await.unwrap_err().code(), tonic::Code::FailedPrecondition);
    assert_eq!(c.upsert(ins()).await.unwrap_err().code(), tonic::Code::FailedPrecondition);
    let del = c.delete(pb::DeleteRequest { id: 1 }).await.unwrap_err();
    assert_eq!(del.code(), tonic::Code::FailedPrecondition);
    let make = pb::CreateCollectionRequest {
        name: "x".into(),
        config: Some(pb::CollectionConfig { dim: 2, ..Default::default() }),
    };
    assert_eq!(c.create_collection(make).await.unwrap_err().code(), tonic::Code::FailedPrecondition);

    assert_eq!(c.get(pb::GetRequest { id: 1 }).await.unwrap().into_inner().vector, vec![1.0, 1.0]);
    let hits = c.search(pb::SearchRequest { vector: vec![1.0, 1.0], k: 1, ..Default::default() }).await.unwrap();
    assert_eq!(hits.into_inner().hits[0].id, 1);
    let _ = stop.send(());
}

#[tokio::test]
async fn memory_only_server_cannot_feed_replicas() {
    let idx = Arc::new(DurableIndex::in_memory(Index::new(Config::new(2, Metric::L2)).unwrap()));
    let cols = Arc::new(Collections::new(idx));
    let addr = free_addr();
    let (stop, rx) = oneshot::channel::<()>();
    tokio::spawn(server::serve_with(cols, addr, Duration::ZERO, ServeOptions::default(), async {
        let _ = rx.await;
    }));
    let mut c = client(addr).await;
    let err = c.replicate(pb::ReplicateRequest {}).await.unwrap_err();
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    let _ = stop.send(());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_replica_that_falls_behind_the_log_is_told_to_resync() {
    let l = start_leader();
    let d = l.default.clone();
    d.insert(0, &[0.0; 3], Payload::new()).unwrap();

    // connect, read the bootstrap, then stop reading so the leader's sender backs up
    let mut c = client(l.addr).await;
    let mut stream = c.replicate(pb::ReplicateRequest {}).await.unwrap().into_inner();
    let first = stream.message().await.unwrap().unwrap();
    assert!(matches!(first.kind, Some(Kind::Bootstrap(_))));

    // several MB of log, then two checkpoints: the log being read is deleted
    let big: String = "x".repeat(2000);
    tokio::task::spawn_blocking(move || {
        for i in 1..3000u64 {
            let mut p = Payload::new();
            p.insert("pad".into(), Value::Str(big.clone()));
            d.insert(i, &[i as f32, 0.0, 0.0], p).unwrap();
        }
        d.checkpoint().unwrap();
        d.checkpoint().unwrap();
    })
    .await
    .unwrap();

    let (mut chunks, t0) = (0, Instant::now());
    loop {
        assert!(t0.elapsed() < Duration::from_secs(30), "never got a resync after {chunks} chunks");
        match stream.message().await.unwrap() {
            Some(m) => match m.kind {
                Some(Kind::WalChunk(_)) => chunks += 1,
                Some(Kind::Resync(_)) => break,
                other => panic!("unexpected {other:?}"),
            },
            None => panic!("stream ended without a resync after {chunks} chunks"),
        }
    }
    let _ = l.stop.send(());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_caught_up_replica_crosses_a_checkpoint_without_resyncing() {
    let l = start_leader();
    let d = l.default.clone();
    for i in 0..20u64 {
        d.insert(i, &[i as f32, 0.0, 0.0], Payload::new()).unwrap();
    }
    let mut c = client(l.addr).await;
    let mut stream = c.replicate(pb::ReplicateRequest {}).await.unwrap().into_inner();
    assert!(matches!(stream.message().await.unwrap().unwrap().kind, Some(Kind::Bootstrap(_))));

    // drain everything so far: once nothing arrives for a moment the replica is caught up
    let mut seen = 0;
    while let Ok(m) = tokio::time::timeout(Duration::from_millis(500), stream.message()).await {
        if let Some(Kind::WalChunk(w)) = m.unwrap().unwrap().kind {
            seen += w.data.len();
        }
    }
    assert!(seen > 0);

    let d2 = d.clone();
    tokio::task::spawn_blocking(move || {
        d2.checkpoint().unwrap();
        d2.insert(100, &[1.0, 1.0, 1.0], Payload::new()).unwrap();
    })
    .await
    .unwrap();

    let t0 = Instant::now();
    loop {
        assert!(t0.elapsed() < Duration::from_secs(10), "never saw the next log");
        let m = tokio::time::timeout(Duration::from_secs(5), stream.message()).await.expect("stream stalled");
        match m.unwrap().unwrap().kind {
            Some(Kind::WalChunk(w)) if w.generation == 1 => break,
            Some(Kind::WalChunk(_)) => {}
            Some(Kind::Resync(_)) => panic!("a caught-up replica was told to resync"),
            other => panic!("unexpected {other:?}"),
        }
    }
    let _ = l.stop.send(());
}

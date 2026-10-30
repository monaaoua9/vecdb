//! Single-leader read replicas.
//!
//! The leader streams its newest snapshot, then tails the write-ahead log from the start of
//! that snapshot's generation. A replica keeps everything in memory: it loads the snapshot,
//! applies complete log records as they arrive, and starts over from a fresh snapshot when the
//! leader tells it the log it was reading is gone (a checkpoint retired it). No failover, no
//! persistence on the replica, replicas can't be chained.

use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::sync::mpsc::Sender;
use tonic::transport::Channel;
use tonic::{Request, Status};

use super::pb::{self, replication_message::Kind, vector_db_client::VectorDbClient};
use crate::collections::Collections;
use crate::config::Config;
use crate::durable::{DurableIndex, wal_path};
use crate::error::Error;
use crate::hnsw::Index;
use crate::quant::Quantization;
use crate::snapshots;

const WAL_CHUNK: usize = 256 << 10;
const SNAPSHOT_CHUNK: usize = 1 << 20;
const IDLE: Duration = Duration::from_millis(50);
const DISCOVER_EVERY: Duration = Duration::from_secs(10);

pub(super) struct WalTail {
    pub dir: PathBuf,
    pub generation: u64,
    pub offset: u64,
}

pub(super) enum Tail {
    Data {
        generation: u64,
        offset: u64,
        bytes: Vec<u8>,
    },
    Idle,
    /// the log we were reading was deleted by a checkpoint
    Lost,
}

impl WalTail {
    pub fn poll(&mut self) -> std::io::Result<Tail> {
        loop {
            let mut f = match File::open(wal_path(&self.dir, self.generation)) {
                Ok(f) => f,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Tail::Lost),
                Err(e) => return Err(e),
            };
            let len = f.metadata()?.len();
            if len > self.offset {
                f.seek(SeekFrom::Start(self.offset))?;
                let mut bytes = vec![0; ((len - self.offset) as usize).min(WAL_CHUNK)];
                f.read_exact(&mut bytes)?;
                let at = self.offset;
                self.offset += bytes.len() as u64;
                return Ok(Tail::Data { generation: self.generation, offset: at, bytes });
            }
            // Caught up. A newer log means this one is final: writers are held off before the
            // next file is created, so the length can't grow any more, but look once more.
            if wal_path(&self.dir, self.generation + 1).exists() {
                if f.metadata()?.len() > self.offset {
                    continue;
                }
                self.generation += 1;
                self.offset = 0;
                continue;
            }
            return Ok(Tail::Idle);
        }
    }
}

type Out = Sender<Result<pb::ReplicationMessage, Status>>;

fn msg(kind: Kind) -> Result<pb::ReplicationMessage, Status> {
    Ok(pb::ReplicationMessage { kind: Some(kind) })
}

/// Runs on a blocking thread, one per connected replica, until the replica goes away,
/// the server shuts down, or something fails.
pub(super) fn stream_to_replica(dir: PathBuf, config: Vec<u8>, tx: Out, stop: Arc<AtomicBool>) {
    let fail = |e: std::io::Error| {
        let _ = tx.blocking_send(Err(Status::internal(e.to_string())));
    };
    let snap = snapshots::latest(&dir);
    let (generation, mut file, snapshot_bytes) = match &snap {
        Some((g, p)) => match File::open(p).and_then(|f| Ok((f.metadata()?.len(), f))) {
            Ok((len, f)) => (*g, Some(f), len),
            Err(e) => return fail(e),
        },
        None => (crate::durable::wal_files(&dir).first().map_or(0, |w| w.0), None, 0),
    };

    let boot = pb::Bootstrap { config, generation, has_snapshot: file.is_some(), snapshot_bytes };
    if tx.blocking_send(msg(Kind::Bootstrap(boot))).is_err() {
        return;
    }
    if let Some(f) = file.as_mut() {
        let mut left = snapshot_bytes as usize;
        while left > 0 {
            let mut chunk = vec![0; left.min(SNAPSHOT_CHUNK)];
            if let Err(e) = f.read_exact(&mut chunk) {
                return fail(e);
            }
            left -= chunk.len();
            if tx.blocking_send(msg(Kind::SnapshotChunk(chunk))).is_err() {
                return;
            }
        }
    }

    let mut tail = WalTail { dir, generation, offset: 0 };
    while !stop.load(Ordering::Relaxed) {
        match tail.poll() {
            Ok(Tail::Data { generation, offset, bytes }) => {
                let chunk = pb::WalChunk { generation, offset, data: bytes };
                if tx.blocking_send(msg(Kind::WalChunk(chunk))).is_err() {
                    return;
                }
            }
            Ok(Tail::Idle) => std::thread::sleep(IDLE),
            Ok(Tail::Lost) => {
                let _ = tx.blocking_send(msg(Kind::Resync(pb::Resync {})));
                return;
            }
            Err(e) => return fail(e),
        }
    }
}

fn authed<T>(body: T, collection: Option<&str>, key: &Option<String>) -> Request<T> {
    let mut r = Request::new(body);
    if let Some(Ok(v)) = collection.map(|c| c.parse()) {
        r.metadata_mut().insert("x-collection", v);
    }
    if let Some(Ok(v)) = key.as_ref().map(|k| format!("Bearer {k}").parse()) {
        r.metadata_mut().insert("authorization", v);
    }
    r
}

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

/// Keeps `cols` in step with the leader at `leader` (e.g. `http://10.0.0.5:50051`):
/// every collection it has, discovered again every few seconds, until `stop` is set.
/// The leader doesn't have to be up yet; failures are retried with a growing pause.
pub async fn follow(
    cols: Arc<Collections>,
    leader: String,
    api_key: Option<String>,
    stop: Arc<AtomicBool>,
) -> Result<(), BoxErr> {
    follow_every(cols, leader, api_key, stop, DISCOVER_EVERY).await
}

/// Same as `follow`, with the pause between looking for new leader collections spelled out.
pub async fn follow_every(
    cols: Arc<Collections>,
    leader: String,
    api_key: Option<String>,
    stop: Arc<AtomicBool>,
    discover_every: Duration,
) -> Result<(), BoxErr> {
    let channel = Channel::from_shared(leader)?.connect_lazy();
    let tmp = std::env::temp_dir().join(format!("vecdb-replica-{}", std::process::id()));
    std::fs::create_dir_all(&tmp)?;

    let mut client = VectorDbClient::new(channel.clone());
    let mut started: HashSet<String> = HashSet::new();
    while !stop.load(Ordering::Relaxed) {
        match client.list_collections(authed(pb::ListCollectionsRequest {}, None, &api_key)).await {
            Ok(r) => {
                for c in r.into_inner().collections {
                    if started.insert(c.name.clone()) {
                        tracing::info!(collection = %c.name, "following");
                        tokio::spawn(follow_collection(
                            c.name,
                            channel.clone(),
                            api_key.clone(),
                            cols.clone(),
                            stop.clone(),
                            tmp.clone(),
                        ));
                    }
                }
            }
            Err(e) => tracing::warn!(error = %e, "can't list the leader's collections"),
        }
        let mut waited = Duration::ZERO;
        while waited < discover_every && !stop.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_millis(100)).await;
            waited += Duration::from_millis(100);
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
    Ok(())
}

async fn follow_collection(
    name: String,
    channel: Channel,
    key: Option<String>,
    cols: Arc<Collections>,
    stop: Arc<AtomicBool>,
    tmp: PathBuf,
) {
    let mut pause = Duration::from_millis(200);
    while !stop.load(Ordering::Relaxed) {
        match sync_once(&name, &channel, &key, &cols, &stop, &tmp).await {
            Ok(()) => pause = Duration::from_millis(200),
            Err(e) => {
                tracing::warn!(collection = %name, error = %e, "replication interrupted, retrying");
                pause = (pause * 2).min(Duration::from_secs(5));
            }
        }
        tokio::time::sleep(pause).await;
    }
}

async fn sync_once(
    name: &str,
    channel: &Channel,
    key: &Option<String>,
    cols: &Arc<Collections>,
    stop: &AtomicBool,
    tmp: &std::path::Path,
) -> Result<(), BoxErr> {
    let mut client = VectorDbClient::new(channel.clone());
    let mut stream = client.replicate(authed(pb::ReplicateRequest {}, Some(name), key)).await?.into_inner();

    let Some(Kind::Bootstrap(b)) = stream.message().await?.and_then(|m| m.kind) else {
        return Err("leader didn't start with a bootstrap message".into());
    };
    let cfg: Config = postcard::from_bytes(&b.config)?;
    let index = if b.has_snapshot {
        let path = tmp.join(format!("{name}-{}.vdb", b.generation));
        let mut f = File::create(&path)?;
        let mut got = 0u64;
        while got < b.snapshot_bytes {
            match stream.message().await?.and_then(|m| m.kind) {
                Some(Kind::SnapshotChunk(c)) => {
                    f.write_all(&c)?;
                    got += c.len() as u64;
                }
                _ => return Err("snapshot transfer was cut short".into()),
            }
        }
        drop(f);
        Index::open(&path)?
    } else if matches!(cfg.quantization, Quantization::Pq { .. }) {
        return Err("leader has no snapshot yet and uses pq; the codebook only travels in a snapshot".into());
    } else {
        Index::new(cfg)?
    };
    let target = cols.install(name, index);
    tracing::info!(
        collection = name,
        generation = b.generation,
        vectors = target.len(),
        "replica loaded snapshot"
    );

    let (mut generation, mut consumed, mut buf) = (b.generation, 0u64, Vec::<u8>::new());
    while !stop.load(Ordering::Relaxed) {
        let Some(m) = stream.message().await? else {
            return Ok(());
        };
        match m.kind {
            Some(Kind::WalChunk(w)) => {
                if w.generation != generation {
                    if !buf.is_empty() {
                        return Err("a log ended in the middle of a record".into());
                    }
                    (generation, consumed) = (w.generation, 0);
                }
                if w.offset != consumed + buf.len() as u64 {
                    return Err("gap in the log stream".into());
                }
                buf.extend_from_slice(&w.data);
                let (ops, used) = DurableIndex::parse_records(&buf);
                buf.drain(..used);
                consumed += used as u64;
                let t = target.clone();
                tokio::task::spawn_blocking(move || -> Result<(), Error> {
                    ops.into_iter().try_for_each(|op| t.apply_op(op))
                })
                .await??;
            }
            Some(Kind::Resync(_)) => {
                return Err("fell behind the leader's log, loading a new snapshot".into());
            }
            _ => return Err("unexpected message from the leader".into()),
        }
    }
    Ok(())
}

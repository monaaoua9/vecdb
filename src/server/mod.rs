use std::future::Future;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::task::spawn_blocking;
use tonic::{Request, Response, Status, Streaming};

use crate::durable::DurableIndex;
use crate::hnsw::{Item, SearchParams};

mod convert;

#[allow(clippy::all)]
pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/vecdb.v1.rs"));
}

use pb::vector_db_server::{VectorDb, VectorDbServer};

const MAX_K: usize = 10_000;
const STREAM_BATCH: usize = 512;
const MAX_MESSAGE_BYTES: usize = 64 << 20;

pub struct VectorDbService {
    index: Arc<DurableIndex>,
}

impl VectorDbService {
    pub fn new(index: Arc<DurableIndex>) -> Self {
        VectorDbService { index }
    }

    pub async fn checkpoint(&self) -> Result<Option<(PathBuf, u64)>, Status> {
        let index = self.index.clone();
        spawn_blocking(move || index.checkpoint())
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .map_err(Status::from)
    }
}

fn blocking_err(e: tokio::task::JoinError) -> Status {
    Status::internal(format!("worker failed: {e}"))
}

#[tonic::async_trait]
impl VectorDb for VectorDbService {
    async fn insert(&self, req: Request<pb::InsertRequest>) -> Result<Response<pb::InsertResponse>, Status> {
        let r = req.into_inner();
        let payload = convert::payload_in(r.payload)?;
        let index = self.index.clone();
        spawn_blocking(move || index.insert(r.id, &r.vector, payload)).await.map_err(blocking_err)??;
        Ok(Response::new(pb::InsertResponse {}))
    }

    async fn upsert(&self, req: Request<pb::InsertRequest>) -> Result<Response<pb::InsertResponse>, Status> {
        let r = req.into_inner();
        let payload = convert::payload_in(r.payload)?;
        let index = self.index.clone();
        spawn_blocking(move || index.upsert(r.id, &r.vector, payload)).await.map_err(blocking_err)??;
        Ok(Response::new(pb::InsertResponse {}))
    }

    async fn update_payload(
        &self,
        req: Request<pb::UpdatePayloadRequest>,
    ) -> Result<Response<pb::UpdatePayloadResponse>, Status> {
        let r = req.into_inner();
        let payload = convert::payload_in(r.payload)?;
        let index = self.index.clone();
        spawn_blocking(move || {
            if r.merge { index.merge_payload(r.id, payload) } else { index.set_payload(r.id, payload) }
        })
        .await
        .map_err(blocking_err)??;
        Ok(Response::new(pb::UpdatePayloadResponse {}))
    }

    async fn insert_stream(
        &self,
        req: Request<Streaming<pb::InsertRequest>>,
    ) -> Result<Response<pb::InsertStreamResponse>, Status> {
        let mut stream = req.into_inner();
        let threads = std::thread::available_parallelism().map_or(2, |n| n.get());
        let mut batch: Vec<Item> = Vec::with_capacity(STREAM_BATCH);
        let mut inserted = 0u64;

        loop {
            let msg = stream.message().await?;
            let done = msg.is_none();
            if let Some(m) = msg {
                batch.push(Item { id: m.id, vector: m.vector, payload: convert::payload_in(m.payload)? });
            }
            if batch.len() >= STREAM_BATCH || (done && !batch.is_empty()) {
                let items = std::mem::take(&mut batch);
                let n = items.len() as u64;
                let index = self.index.clone();
                spawn_blocking(move || index.insert_bulk(&items, threads)).await.map_err(blocking_err)??;
                inserted += n;
            }
            if done {
                break;
            }
        }
        Ok(Response::new(pb::InsertStreamResponse { inserted }))
    }

    async fn search(&self, req: Request<pb::SearchRequest>) -> Result<Response<pb::SearchResponse>, Status> {
        let r = req.into_inner();
        let k = r.k as usize;
        if k == 0 || k > MAX_K {
            return Err(Status::invalid_argument(format!("k must be in 1..={MAX_K}")));
        }
        let mut params = SearchParams::new(k).strategy(convert::strategy_in(r.strategy));
        if r.ef > 0 {
            params = params.ef(r.ef as usize);
        }
        if let Some(f) = r.filter {
            params = params.filter(convert::filter_in(f)?);
        }
        let with_payload = r.with_payload;
        let index = self.index.clone();

        let t0 = Instant::now();
        let hits = spawn_blocking(move || {
            let hits = index.search_with(&r.vector, &params)?;
            let out: Vec<pb::Hit> = hits
                .into_iter()
                .map(|h| pb::Hit {
                    id: h.id,
                    distance: h.distance,
                    payload: with_payload
                        .then(|| index.get(h.id).map(|rec| convert::payload_out(&rec.payload)))
                        .flatten(),
                })
                .collect();
            Ok::<_, crate::Error>(out)
        })
        .await
        .map_err(blocking_err)??;

        Ok(Response::new(pb::SearchResponse { hits, took_micros: t0.elapsed().as_micros() as u64 }))
    }

    async fn get(&self, req: Request<pb::GetRequest>) -> Result<Response<pb::GetResponse>, Status> {
        let id = req.into_inner().id;
        let rec = self.index.get(id).ok_or_else(|| Status::not_found(format!("id {id} not found")))?;
        Ok(Response::new(pb::GetResponse {
            id,
            vector: rec.vector,
            payload: Some(convert::payload_out(&rec.payload)),
        }))
    }

    async fn delete(&self, req: Request<pb::DeleteRequest>) -> Result<Response<pb::DeleteResponse>, Status> {
        let id = req.into_inner().id;
        let index = self.index.clone();
        spawn_blocking(move || index.delete(id)).await.map_err(blocking_err)??;
        Ok(Response::new(pb::DeleteResponse {}))
    }

    async fn stats(&self, _: Request<pb::StatsRequest>) -> Result<Response<pb::StatsResponse>, Status> {
        let index = self.index.clone();
        let s = spawn_blocking(move || index.stats()).await.map_err(blocking_err)?;
        let cfg = self.index.config();
        Ok(Response::new(pb::StatsResponse {
            live: s.live as u64,
            deleted: s.deleted as u64,
            capacity: s.capacity as u64,
            max_level: s.max_level as u32,
            layer0_edges: s.layer0_edges as u64,
            dim: cfg.dim as u32,
            metric: format!("{:?}", cfg.metric),
            quantization: format!("{:?}", cfg.quantization),
            simd_kernel: s.kernel.to_owned(),
            heap_vector_bytes: s.heap_vector_bytes as u64,
            mapped_bytes: s.mapped_bytes as u64,
        }))
    }

    async fn snapshot(
        &self,
        _: Request<pb::SnapshotRequest>,
    ) -> Result<Response<pb::SnapshotResponse>, Status> {
        match self.checkpoint().await? {
            Some((path, vectors)) => {
                Ok(Response::new(pb::SnapshotResponse { path: path.display().to_string(), vectors }))
            }
            None => Err(Status::failed_precondition("server was started without a data directory")),
        }
    }
}

/// Serves until `shutdown` resolves, then writes a final snapshot if a data dir is configured.
pub async fn serve(
    index: Arc<DurableIndex>,
    addr: SocketAddr,
    snapshot_every: Duration,
    shutdown: impl Future<Output = ()>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let svc = Arc::new(VectorDbService::new(index));

    let (health, health_svc) = tonic_health::server::health_reporter();
    health.set_serving::<VectorDbServer<VectorDbService>>().await;

    let ticker = (!snapshot_every.is_zero() && svc.index.is_durable()).then(|| {
        let svc = svc.clone();
        tokio::spawn(async move {
            let mut t = tokio::time::interval(snapshot_every);
            t.tick().await; // first tick fires immediately
            loop {
                t.tick().await;
                match svc.checkpoint().await {
                    Ok(Some((p, n))) => tracing::info!(path = %p.display(), vectors = n, "periodic snapshot"),
                    Ok(None) => {}
                    Err(e) => tracing::error!(error = %e, "periodic snapshot failed"),
                }
            }
        })
    });

    tracing::info!(%addr, "listening");
    tonic::transport::Server::builder()
        .add_service(health_svc)
        .add_service(
            VectorDbServer::from_arc(svc.clone())
                .max_decoding_message_size(MAX_MESSAGE_BYTES)
                .max_encoding_message_size(MAX_MESSAGE_BYTES),
        )
        .serve_with_shutdown(addr, shutdown)
        .await?;

    if let Some(t) = ticker {
        t.abort();
    }
    if let Some((path, n)) = svc.checkpoint().await? {
        tracing::info!(path = %path.display(), vectors = n, "final snapshot written");
    }
    Ok(())
}

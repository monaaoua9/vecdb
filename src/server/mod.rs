use std::future::Future;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::task::spawn_blocking;
use tonic::{Request, Response, Status, Streaming};

use crate::collections::{Collections, DEFAULT};
use crate::config::Config;
use crate::distance::Metric;
use crate::durable::DurableIndex;
use crate::quant::Quantization;
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

const COLLECTION_HEADER: &str = "x-collection";

pub struct VectorDbService {
    cols: Arc<Collections>,
}

impl VectorDbService {
    pub fn new(cols: Arc<Collections>) -> Self {
        VectorDbService { cols }
    }

    fn col<T>(&self, req: &Request<T>) -> Result<Arc<DurableIndex>, Status> {
        let name = match req.metadata().get(COLLECTION_HEADER) {
            Some(v) => v.to_str().map_err(|_| Status::invalid_argument("x-collection is not ascii"))?,
            None => DEFAULT,
        };
        self.cols
            .get(name)
            .ok_or_else(|| Status::not_found(format!("collection {name} not found")))
    }

    /// Snapshots every durable collection, returns (name, path, vectors) for each.
    pub async fn checkpoint_all(&self) -> Result<Vec<(String, PathBuf, u64)>, Status> {
        let mut out = Vec::new();
        for (name, idx) in self.cols.list() {
            let done = spawn_blocking(move || idx.checkpoint())
                .await
                .map_err(|e| Status::internal(e.to_string()))?
                .map_err(Status::from)?;
            if let Some((p, n)) = done {
                out.push((name, p, n));
            }
        }
        Ok(out)
    }
}

fn collection_config(c: pb::CollectionConfig) -> Result<Config, Status> {
    let metric = match pb::MetricKind::try_from(c.metric) {
        Ok(pb::MetricKind::L2) => Metric::L2,
        Ok(pb::MetricKind::Cosine) => Metric::Cosine,
        Ok(pb::MetricKind::InnerProduct) => Metric::InnerProduct,
        Err(_) => return Err(Status::invalid_argument("unknown metric")),
    };
    let mut cfg = Config::new(c.dim as usize, metric);
    cfg.quantization = match pb::QuantizationKind::try_from(c.quantization) {
        Ok(pb::QuantizationKind::None) => Quantization::None,
        Ok(pb::QuantizationKind::Sq8) => Quantization::Sq8,
        Ok(pb::QuantizationKind::Binary) => Quantization::Binary,
        Err(_) => return Err(Status::invalid_argument("unknown quantization")),
    };
    cfg.rerank = c.rerank;
    if c.m > 0 {
        cfg.m = c.m as usize;
    }
    if c.ef_construction > 0 {
        cfg.ef_construction = c.ef_construction as usize;
    }
    if c.ef_search > 0 {
        cfg.ef_search = c.ef_search as usize;
    }
    if c.max_elements > 0 {
        cfg.max_elements = c.max_elements as usize;
    }
    Ok(cfg)
}

fn blocking_err(e: tokio::task::JoinError) -> Status {
    Status::internal(format!("worker failed: {e}"))
}

#[tonic::async_trait]
impl VectorDb for VectorDbService {
    async fn insert(&self, req: Request<pb::InsertRequest>) -> Result<Response<pb::InsertResponse>, Status> {
        let index = self.col(&req)?;
        let r = req.into_inner();
        let payload = convert::payload_in(r.payload)?;
        spawn_blocking(move || index.insert(r.id, &r.vector, payload)).await.map_err(blocking_err)??;
        Ok(Response::new(pb::InsertResponse {}))
    }

    async fn upsert(&self, req: Request<pb::InsertRequest>) -> Result<Response<pb::InsertResponse>, Status> {
        let index = self.col(&req)?;
        let r = req.into_inner();
        let payload = convert::payload_in(r.payload)?;
        spawn_blocking(move || index.upsert(r.id, &r.vector, payload)).await.map_err(blocking_err)??;
        Ok(Response::new(pb::InsertResponse {}))
    }

    async fn update_payload(
        &self,
        req: Request<pb::UpdatePayloadRequest>,
    ) -> Result<Response<pb::UpdatePayloadResponse>, Status> {
        let index = self.col(&req)?;
        let r = req.into_inner();
        let payload = convert::payload_in(r.payload)?;
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
        let target = self.col(&req)?;
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
                let index = target.clone();
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
        let index = self.col(&req)?;
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
        let index = self.col(&req)?;
        let id = req.into_inner().id;
        let rec = index.get(id).ok_or_else(|| Status::not_found(format!("id {id} not found")))?;
        Ok(Response::new(pb::GetResponse {
            id,
            vector: rec.vector,
            payload: Some(convert::payload_out(&rec.payload)),
        }))
    }

    async fn delete(&self, req: Request<pb::DeleteRequest>) -> Result<Response<pb::DeleteResponse>, Status> {
        let index = self.col(&req)?;
        let id = req.into_inner().id;
        spawn_blocking(move || index.delete(id)).await.map_err(blocking_err)??;
        Ok(Response::new(pb::DeleteResponse {}))
    }

    async fn stats(&self, req: Request<pb::StatsRequest>) -> Result<Response<pb::StatsResponse>, Status> {
        let index = self.col(&req)?;
        let idx = index.clone();
        let s = spawn_blocking(move || idx.stats()).await.map_err(blocking_err)?;
        let cfg = index.config();
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
        req: Request<pb::SnapshotRequest>,
    ) -> Result<Response<pb::SnapshotResponse>, Status> {
        let index = self.col(&req)?;
        let done = spawn_blocking(move || index.checkpoint()).await.map_err(blocking_err)??;
        match done {
            Some((path, vectors)) => {
                Ok(Response::new(pb::SnapshotResponse { path: path.display().to_string(), vectors }))
            }
            None => Err(Status::failed_precondition("server was started without a data directory")),
        }
    }

    async fn create_collection(
        &self,
        req: Request<pb::CreateCollectionRequest>,
    ) -> Result<Response<pb::CreateCollectionResponse>, Status> {
        let r = req.into_inner();
        let cfg = collection_config(r.config.ok_or_else(|| Status::invalid_argument("config is required"))?)?;
        let cols = self.cols.clone();
        spawn_blocking(move || cols.create(&r.name, cfg)).await.map_err(blocking_err)??;
        Ok(Response::new(pb::CreateCollectionResponse {}))
    }

    async fn drop_collection(
        &self,
        req: Request<pb::DropCollectionRequest>,
    ) -> Result<Response<pb::DropCollectionResponse>, Status> {
        let name = req.into_inner().name;
        let cols = self.cols.clone();
        spawn_blocking(move || cols.drop_collection(&name)).await.map_err(blocking_err)??;
        Ok(Response::new(pb::DropCollectionResponse {}))
    }

    async fn list_collections(
        &self,
        _: Request<pb::ListCollectionsRequest>,
    ) -> Result<Response<pb::ListCollectionsResponse>, Status> {
        let collections = self
            .cols
            .list()
            .into_iter()
            .map(|(name, idx)| {
                let cfg = idx.config();
                pb::CollectionInfo {
                    name,
                    dim: cfg.dim as u32,
                    metric: format!("{:?}", cfg.metric),
                    quantization: format!("{:?}", cfg.quantization),
                    live: idx.len() as u64,
                }
            })
            .collect();
        Ok(Response::new(pb::ListCollectionsResponse { collections }))
    }
}

/// Serves a single index as the "default" collection; see `serve_collections`.
pub async fn serve(
    index: Arc<DurableIndex>,
    addr: SocketAddr,
    snapshot_every: Duration,
    shutdown: impl Future<Output = ()>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cols = Arc::new(Collections::new(index));
    cols.load_existing()?;
    serve_collections(cols, addr, snapshot_every, shutdown).await
}

/// Serves until `shutdown` resolves, then writes a final snapshot of every durable collection.
pub async fn serve_collections(
    cols: Arc<Collections>,
    addr: SocketAddr,
    snapshot_every: Duration,
    shutdown: impl Future<Output = ()>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let svc = Arc::new(VectorDbService::new(cols.clone()));
    let durable = cols.get(DEFAULT).is_some_and(|d| d.is_durable());

    let (health, health_svc) = tonic_health::server::health_reporter();
    health.set_serving::<VectorDbServer<VectorDbService>>().await;

    let ticker = (!snapshot_every.is_zero() && durable).then(|| {
        let svc = svc.clone();
        tokio::spawn(async move {
            let mut t = tokio::time::interval(snapshot_every);
            t.tick().await; // first tick fires immediately
            loop {
                t.tick().await;
                match svc.checkpoint_all().await {
                    Ok(done) => {
                        for (name, p, n) in done {
                            tracing::info!(collection = %name, path = %p.display(), vectors = n, "periodic snapshot");
                        }
                    }
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
    for (name, path, n) in svc.checkpoint_all().await? {
        tracing::info!(collection = %name, path = %path.display(), vectors = n, "final snapshot written");
    }
    Ok(())
}

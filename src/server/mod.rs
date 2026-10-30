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

mod auth;
mod convert;
mod metrics;

pub use auth::ApiKeys;
pub use metrics::Metrics;

#[allow(clippy::all)]
pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/vecdb.v1.rs"));
}

use pb::vector_db_server::{VectorDb, VectorDbServer};

const MAX_K: usize = 10_000;
const MAX_BATCH: usize = 256;
const MAX_SCROLL: usize = 10_000;
const MAX_FACETS: usize = 1000;
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

fn search_params(
    k: u32,
    ef: u32,
    filter: Option<pb::Filter>,
    strategy: i32,
) -> Result<SearchParams, Status> {
    let k = k as usize;
    if k == 0 || k > MAX_K {
        return Err(Status::invalid_argument(format!("k must be in 1..={MAX_K}")));
    }
    let mut params = SearchParams::new(k).strategy(convert::strategy_in(strategy));
    if ef > 0 {
        params = params.ef(ef as usize);
    }
    if let Some(f) = filter {
        params = params.filter(convert::filter_in(f)?);
    }
    Ok(params)
}

fn hits_out(index: &DurableIndex, hits: Vec<crate::Hit>, with_payload: bool) -> Vec<pb::Hit> {
    hits.into_iter()
        .map(|h| pb::Hit {
            id: h.id,
            distance: h.distance,
            payload: with_payload
                .then(|| index.get(h.id).map(|rec| convert::payload_out(&rec.payload)))
                .flatten(),
        })
        .collect()
}

async fn run_search(index: Arc<DurableIndex>, r: pb::SearchRequest) -> Result<pb::SearchResponse, Status> {
    if r.radius.is_some() && r.mmr_lambda.is_some() {
        return Err(Status::invalid_argument("radius and mmr_lambda can't be combined"));
    }
    let params = search_params(r.k, r.ef, r.filter, r.strategy)?;
    let with_payload = r.with_payload;
    if r.text.is_some() && (r.radius.is_some() || r.mmr_lambda.is_some()) {
        return Err(Status::invalid_argument("text can't be combined with radius or mmr_lambda"));
    }
    let fetch = if r.mmr_fetch > 0 { r.mmr_fetch as usize } else { params.k * 4 };

    let t0 = Instant::now();
    let hits = spawn_blocking(move || {
        let hits = match (r.radius, r.mmr_lambda) {
            _ if r.text.is_some() => match (r.vector.is_empty(), r.text.as_deref().unwrap()) {
                (true, t) => index.search_text(t, &params)?,
                (false, t) => index.search_hybrid(&r.vector, t, &params)?,
            },
            (Some(radius), _) => index.search_range(&r.vector, radius, &params)?,
            (_, Some(lambda)) => index.search_mmr(&r.vector, &params, lambda, fetch)?,
            _ => index.search_with(&r.vector, &params)?,
        };
        Ok::<_, crate::Error>(hits_out(&index, hits, with_payload))
    })
    .await
    .map_err(blocking_err)??;
    Ok(pb::SearchResponse { hits, took_micros: t0.elapsed().as_micros() as u64 })
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
        run_search(index, req.into_inner()).await.map(Response::new)
    }

    async fn search_batch(
        &self,
        req: Request<pb::BatchSearchRequest>,
    ) -> Result<Response<pb::BatchSearchResponse>, Status> {
        let index = self.col(&req)?;
        let queries = req.into_inner().queries;
        if queries.len() > MAX_BATCH {
            return Err(Status::invalid_argument(format!("at most {MAX_BATCH} queries per batch")));
        }
        let jobs: Vec<_> = queries.into_iter().map(|q| tokio::spawn(run_search(index.clone(), q))).collect();
        let mut results = Vec::with_capacity(jobs.len());
        for j in jobs {
            results.push(j.await.map_err(blocking_err)??);
        }
        Ok(Response::new(pb::BatchSearchResponse { results }))
    }

    async fn recommend(&self, req: Request<pb::RecommendRequest>) -> Result<Response<pb::SearchResponse>, Status> {
        let index = self.col(&req)?;
        let r = req.into_inner();
        let params = search_params(r.k, r.ef, r.filter, r.strategy)?;
        let with_payload = r.with_payload;

        let t0 = Instant::now();
        let hits = spawn_blocking(move || {
            let hits = index.recommend(&r.positive, &r.negative, &params)?;
            Ok::<_, crate::Error>(hits_out(&index, hits, with_payload))
        })
        .await
        .map_err(blocking_err)??;
        Ok(Response::new(pb::SearchResponse { hits, took_micros: t0.elapsed().as_micros() as u64 }))
    }

    async fn count(&self, req: Request<pb::CountRequest>) -> Result<Response<pb::CountResponse>, Status> {
        let index = self.col(&req)?;
        let filter = req.into_inner().filter.map(convert::filter_in).transpose()?;
        let count = spawn_blocking(move || index.count(filter.as_ref())).await.map_err(blocking_err)?;
        Ok(Response::new(pb::CountResponse { count: count as u64 }))
    }

    async fn scroll(&self, req: Request<pb::ScrollRequest>) -> Result<Response<pb::ScrollResponse>, Status> {
        let index = self.col(&req)?;
        let r = req.into_inner();
        let limit = match r.limit as usize {
            0 => 100,
            n if n > MAX_SCROLL => return Err(Status::invalid_argument(format!("limit is at most {MAX_SCROLL}"))),
            n => n,
        };
        let filter = r.filter.map(convert::filter_in).transpose()?;
        let (with_vector, with_payload) = (r.with_vector, r.with_payload);
        let page = spawn_blocking(move || index.scroll(filter.as_ref(), r.cursor, limit, with_vector))
            .await
            .map_err(blocking_err)?;
        let items = page
            .items
            .into_iter()
            .map(|rec| pb::ScrollItem {
                id: rec.id,
                vector: rec.vector,
                payload: with_payload.then(|| convert::payload_out(&rec.payload)),
            })
            .collect();
        Ok(Response::new(pb::ScrollResponse { items, next_cursor: page.next }))
    }

    async fn facets(&self, req: Request<pb::FacetsRequest>) -> Result<Response<pb::FacetsResponse>, Status> {
        let index = self.col(&req)?;
        let r = req.into_inner();
        let limit = match r.limit as usize {
            0 => 20,
            n => n.min(MAX_FACETS),
        };
        let filter = r.filter.map(convert::filter_in).transpose()?;
        let found = spawn_blocking(move || index.facets(&r.field, filter.as_ref(), limit))
            .await
            .map_err(blocking_err)?;
        let facets = found
            .into_iter()
            .map(|(v, n)| pb::FacetCount { value: Some(convert::value_out(&v)), count: n as u64 })
            .collect();
        Ok(Response::new(pb::FacetsResponse { facets }))
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

#[derive(Default)]
pub struct ServeOptions {
    /// when set, every data RPC needs `authorization: Bearer <key>` (the health service stays open)
    pub api_keys: Option<ApiKeys>,
    /// plain-HTTP Prometheus endpoint, `GET /metrics`
    pub metrics_addr: Option<SocketAddr>,
}

pub async fn serve_collections(
    cols: Arc<Collections>,
    addr: SocketAddr,
    snapshot_every: Duration,
    shutdown: impl Future<Output = ()>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    serve_with(cols, addr, snapshot_every, ServeOptions::default(), shutdown).await
}

/// Serves until `shutdown` resolves, then writes a final snapshot of every durable collection.
pub async fn serve_with(
    cols: Arc<Collections>,
    addr: SocketAddr,
    snapshot_every: Duration,
    opts: ServeOptions,
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

    let metrics = Metrics::default();
    let scraper = opts.metrics_addr.map(|a| {
        let (m, c) = (metrics.clone(), cols.clone());
        tokio::spawn(async move {
            if let Err(e) = metrics::serve_http(a, m, c).await {
                tracing::error!(error = %e, "metrics endpoint stopped");
            }
        })
    });

    let keys = opts.api_keys;
    let data = tonic::service::interceptor::InterceptedService::new(
        VectorDbServer::from_arc(svc.clone())
            .max_decoding_message_size(MAX_MESSAGE_BYTES)
            .max_encoding_message_size(MAX_MESSAGE_BYTES),
        move |req: Request<()>| match &keys {
            Some(k) => k.check(req),
            None => Ok(req),
        },
    );

    tracing::info!(%addr, "listening");
    tonic::transport::Server::builder()
        .layer(metrics::MetricsLayer(metrics))
        .add_service(health_svc)
        .add_service(data)
        .serve_with_shutdown(addr, shutdown)
        .await?;

    if let Some(t) = ticker {
        t.abort();
    }
    if let Some(t) = scraper {
        t.abort();
    }
    for (name, path, n) in svc.checkpoint_all().await? {
        tracing::info!(collection = %name, path = %path.display(), vectors = n, "final snapshot written");
    }
    Ok(())
}

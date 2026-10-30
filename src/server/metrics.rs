use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use parking_lot::Mutex;
use tower::{Layer, Service};

use crate::collections::Collections;

const BOUNDS: [f64; 12] = [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5];

#[derive(Default)]
struct Rpc {
    ok: u64,
    err: u64,
    sum: f64,
    // per-bucket counts, made cumulative at render time
    buckets: [u64; BOUNDS.len() + 1],
}

/// Per-RPC request counts and latency histograms, rendered in the Prometheus text format.
#[derive(Clone, Default)]
pub struct Metrics(Arc<Mutex<BTreeMap<String, Rpc>>>);

impl Metrics {
    pub fn observe(&self, rpc: &str, secs: f64, ok: bool) {
        let mut m = self.0.lock();
        let r = m.entry(rpc.to_owned()).or_default();
        if ok {
            r.ok += 1;
        } else {
            r.err += 1;
        }
        r.sum += secs;
        r.buckets[BOUNDS.iter().position(|&b| secs <= b).unwrap_or(BOUNDS.len())] += 1;
    }

    pub fn render(&self, cols: &Collections) -> String {
        let mut out = String::new();
        let m = self.0.lock();

        out.push_str("# TYPE vecdb_rpc_requests_total counter\n");
        for (name, r) in m.iter() {
            out.push_str(&format!("vecdb_rpc_requests_total{{rpc=\"{name}\",result=\"ok\"}} {}\n", r.ok));
            out.push_str(&format!("vecdb_rpc_requests_total{{rpc=\"{name}\",result=\"error\"}} {}\n", r.err));
        }
        out.push_str("# TYPE vecdb_rpc_duration_seconds histogram\n");
        for (name, r) in m.iter() {
            let mut cum = 0;
            for (i, b) in BOUNDS.iter().enumerate() {
                cum += r.buckets[i];
                out.push_str(&format!(
                    "vecdb_rpc_duration_seconds_bucket{{rpc=\"{name}\",le=\"{b}\"}} {cum}\n"
                ));
            }
            let total = cum + r.buckets[BOUNDS.len()];
            out.push_str(&format!(
                "vecdb_rpc_duration_seconds_bucket{{rpc=\"{name}\",le=\"+Inf\"}} {total}\n"
            ));
            out.push_str(&format!("vecdb_rpc_duration_seconds_sum{{rpc=\"{name}\"}} {}\n", r.sum));
            out.push_str(&format!("vecdb_rpc_duration_seconds_count{{rpc=\"{name}\"}} {total}\n"));
        }
        drop(m);

        out.push_str("# TYPE vecdb_vectors gauge\n# TYPE vecdb_tombstone_ratio gauge\n");
        for (name, idx) in cols.list() {
            out.push_str(&format!("vecdb_vectors{{collection=\"{name}\"}} {}\n", idx.len()));
            out.push_str(&format!(
                "vecdb_tombstone_ratio{{collection=\"{name}\"}} {}\n",
                idx.tombstone_ratio()
            ));
        }
        out
    }
}

#[derive(Clone)]
pub struct MetricsLayer(pub Metrics);

impl<S> Layer<S> for MetricsLayer {
    type Service = MetricsService<S>;
    fn layer(&self, inner: S) -> MetricsService<S> {
        MetricsService { inner, metrics: self.0.clone() }
    }
}

#[derive(Clone)]
pub struct MetricsService<S> {
    inner: S,
    metrics: Metrics,
}

impl<S, B, R> Service<http::Request<B>> for MetricsService<S>
where
    S: Service<http::Request<B>, Response = http::Response<R>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    B: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<S::Response, S::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: http::Request<B>) -> Self::Future {
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);
        let rpc = req.uri().path().rsplit('/').next().unwrap_or("unknown").to_owned();
        let metrics = self.metrics.clone();
        let t0 = Instant::now();
        Box::pin(async move {
            let res = inner.call(req).await;
            // failed unary calls answer "trailers-only", so grpc-status is in the headers;
            // a successful one carries it in the trailers and has no such header here
            let ok = match &res {
                Ok(r) => r.headers().get("grpc-status").is_none_or(|v| v.as_bytes() == b"0"),
                Err(_) => false,
            };
            metrics.observe(&rpc, t0.elapsed().as_secs_f64(), ok);
            res
        })
    }
}

/// Minimal HTTP/1.1 responder: `GET /metrics` gets the text, everything else a 404.
pub async fn serve_http(
    addr: std::net::SocketAddr,
    metrics: Metrics,
    cols: Arc<Collections>,
) -> std::io::Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "metrics endpoint");
    loop {
        let (mut sock, _) = listener.accept().await?;
        let (metrics, cols) = (metrics.clone(), cols.clone());
        tokio::spawn(async move {
            let mut buf = [0u8; 1024];
            let n = sock.read(&mut buf).await.unwrap_or(0);
            let line = String::from_utf8_lossy(&buf[..n]);
            let reply = if line.starts_with("GET /metrics ") {
                let body =
                    tokio::task::spawn_blocking(move || metrics.render(&cols)).await.unwrap_or_default();
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
            } else {
                "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_owned()
            };
            let _ = sock.write_all(reply.as_bytes()).await;
        });
    }
}

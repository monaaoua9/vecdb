use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, ValueEnum};
use vecdb::server;
use vecdb::{Config, Durability, DurableIndex, Index, Metric, Quantization};

#[derive(Clone, Copy, ValueEnum)]
enum MetricArg {
    L2,
    Cosine,
    Ip,
}

#[derive(Clone, Copy, ValueEnum)]
enum QuantArg {
    None,
    Sq8,
}

#[derive(Parser)]
#[command(name = "vecdb-server", version, about = "HNSW vector database over gRPC")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:50051")]
    listen: SocketAddr,

    /// snapshots and write-ahead log live here; without it the index is memory-only
    #[arg(long)]
    data_dir: Option<PathBuf>,

    /// only used when no snapshot exists yet
    #[arg(long, default_value_t = 768)]
    dim: usize,
    #[arg(long, value_enum, default_value = "cosine")]
    metric: MetricArg,
    #[arg(long, value_enum, default_value = "none")]
    quantization: QuantArg,
    #[arg(long, default_value_t = 16)]
    m: usize,
    #[arg(long, default_value_t = 200)]
    ef_construction: usize,
    #[arg(long, default_value_t = 64)]
    ef_search: usize,
    #[arg(long, default_value_t = 1_000_000)]
    max_elements: usize,

    /// at start-up, rebuild the graph if more than this share of nodes are deleted (0 = never)
    #[arg(long, default_value_t = 0.3)]
    compact_above: f64,

    /// fsync the write-ahead log before acknowledging each write
    #[arg(long)]
    wal_fsync: bool,

    /// 0 disables periodic snapshots
    #[arg(long, default_value_t = 0)]
    snapshot_every_secs: u64,
}

fn build_config(a: &Args) -> Config {
    let mut cfg = Config::new(
        a.dim,
        match a.metric {
            MetricArg::L2 => Metric::L2,
            MetricArg::Cosine => Metric::Cosine,
            MetricArg::Ip => Metric::InnerProduct,
        },
    );
    cfg.quantization = match a.quantization {
        QuantArg::None => Quantization::None,
        QuantArg::Sq8 => Quantization::Sq8,
    };
    cfg.m = a.m;
    cfg.ef_construction = a.ef_construction;
    cfg.ef_search = a.ef_search;
    cfg.max_elements = a.max_elements;
    cfg
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();
    let cfg = build_config(&args);
    let mode = if args.wal_fsync { Durability::Fsync } else { Durability::Flush };
    let mut index = match &args.data_dir {
        Some(dir) => DurableIndex::open(dir, cfg, mode)?,
        None => DurableIndex::in_memory(Index::new(cfg)?),
    };
    if index.is_durable() && args.compact_above > 0.0 && index.tombstone_ratio() > args.compact_above {
        let threads = std::thread::available_parallelism().map_or(2, |n| n.get());
        tracing::info!(ratio = index.tombstone_ratio(), "compacting before serving");
        index.compact(threads)?;
    }
    let index = Arc::new(index);
    tracing::info!(kernel = index.stats().kernel, dim = index.config().dim, "index ready");

    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutting down");
    };

    let every = Duration::from_secs(args.snapshot_every_secs);
    server::serve(index, args.listen, every, shutdown).await
}

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, ValueEnum};
use vecdb::server;
use vecdb::{Collections, Config, Durability, DurableIndex, Index, Metric, Quantization};

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
    /// 1 bit per component, pair it with --rerank
    Binary,
    /// needs a codebook in --data-dir, see DurableIndex::train_pq
    Pq,
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
    /// subspaces (bytes per vector) when --quantization pq
    #[arg(long, default_value_t = 16)]
    pq_m: u16,
    /// keep full-precision vectors and re-score the final candidates with them
    #[arg(long)]
    rerank: bool,
    #[arg(long, default_value_t = 16)]
    m: usize,
    #[arg(long, default_value_t = 200)]
    ef_construction: usize,
    #[arg(long, default_value_t = 64)]
    ef_search: usize,
    #[arg(long, default_value_t = 1_000_000)]
    max_elements: usize,

    /// rebuild the graph when more than this share of nodes are deleted, at start-up and
    /// every --compact-check-secs while serving (0 = never)
    #[arg(long, default_value_t = 0.3)]
    compact_above: f64,

    /// how often the running server checks the deleted share, 0 = start-up only
    #[arg(long, default_value_t = 60)]
    compact_check_secs: u64,

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
        QuantArg::Binary => Quantization::Binary,
        QuantArg::Pq => Quantization::Pq { m: a.pq_m },
    };
    cfg.rerank = a.rerank;
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
    let cols = Arc::new(Collections::new(index));
    let extra = cols.load_existing()?;
    if extra > 0 {
        tracing::info!(extra, "collections loaded");
    }

    if args.compact_above > 0.0 && args.compact_check_secs > 0 {
        let (cols, above) = (cols.clone(), args.compact_above);
        let every = Duration::from_secs(args.compact_check_secs);
        tokio::spawn(async move {
            let mut t = tokio::time::interval(every);
            t.tick().await;
            loop {
                t.tick().await;
                for (name, idx) in cols.list() {
                    let ran = tokio::task::spawn_blocking(move || {
                        if idx.tombstone_ratio() <= above {
                            return Ok(false);
                        }
                        let threads = std::thread::available_parallelism().map_or(2, |n| n.get());
                        idx.compact_online(threads)
                    })
                    .await;
                    match ran {
                        Ok(Ok(true)) => tracing::info!(collection = %name, "online compaction done"),
                        Ok(Ok(false)) => {}
                        Ok(Err(e)) => tracing::error!(collection = %name, error = %e, "online compaction failed"),
                        Err(e) => tracing::error!(error = %e, "compaction worker died"),
                    }
                }
            }
        });
    }

    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutting down");
    };

    let every = Duration::from_secs(args.snapshot_every_secs);
    server::serve_collections(cols, args.listen, every, shutdown).await
}

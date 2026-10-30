use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand, ValueEnum};
use vecdb::admin::{self, DirReport};
use vecdb::{Config, Metric, Quantization};

/// Offline tools for a vecdb data directory. Stop the server before anything that writes.
#[derive(Parser)]
#[command(name = "vecdb-cli")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// show snapshots, logs and collections of a data directory
    Inspect { dir: PathBuf },
    /// like inspect, but exits non-zero when something is damaged
    Verify { dir: PathBuf },
    /// copy a data directory (safe while the server runs, retries if a snapshot is cut meanwhile)
    Backup { dir: PathBuf, dest: PathBuf },
    /// copy a backup into an empty directory and verify it
    Restore { backup: PathBuf, dest: PathBuf },
    /// write every live vector as JSON lines
    Export {
        dir: PathBuf,
        /// output file, `-` for stdout
        out: PathBuf,
    },
    /// load JSON lines (as written by export) into a directory, creating it if needed
    Import {
        dir: PathBuf,
        input: PathBuf,
        #[command(flatten)]
        shape: Shape,
    },
}

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
    Binary,
}

/// only needed when the directory has no snapshot yet
#[derive(Args)]
struct Shape {
    #[arg(long)]
    dim: Option<usize>,
    #[arg(long, value_enum, default_value = "l2")]
    metric: MetricArg,
    #[arg(long, value_enum, default_value = "none")]
    quantization: QuantArg,
    #[arg(long)]
    rerank: bool,
    #[arg(long, default_value_t = 1_000_000)]
    max_elements: usize,
}

impl Shape {
    fn config(&self) -> Option<Config> {
        let mut c = Config::new(
            self.dim?,
            match self.metric {
                MetricArg::L2 => Metric::L2,
                MetricArg::Cosine => Metric::Cosine,
                MetricArg::Ip => Metric::InnerProduct,
            },
        );
        c.quantization = match self.quantization {
            QuantArg::None => Quantization::None,
            QuantArg::Sq8 => Quantization::Sq8,
            QuantArg::Binary => Quantization::Binary,
        };
        c.rerank = self.rerank;
        c.max_elements = self.max_elements;
        Some(c)
    }
}

fn print_report(r: &DirReport, indent: &str) {
    match (&r.snapshot, &r.snapshot_error) {
        (Some(s), _) => println!(
            "{indent}snapshot gen {}  {} live  {:.1}% deleted  {} bytes  dim {} {:?} {:?}{}",
            s.generation,
            s.live,
            s.tombstone_ratio * 100.0,
            s.bytes,
            s.config.dim,
            s.config.metric,
            s.config.quantization,
            if s.config.rerank { " +rerank" } else { "" },
        ),
        (None, Some(e)) => println!("{indent}snapshot UNREADABLE: {e}"),
        (None, None) => println!("{indent}no snapshot"),
    }
    for w in &r.wals {
        let name = w.path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        println!("{indent}{name}  {} records  {}/{} bytes intact", w.records, w.valid_bytes, w.file_bytes);
    }
    if r.has_codebook {
        println!("{indent}codebook.bin");
    }
    for (name, c) in &r.collections {
        println!("{indent}collection {name}:");
        print_report(c, &format!("{indent}  "));
    }
}

/// prints the verdict, true when nothing is wrong
fn verdict(r: &DirReport) -> bool {
    for w in r.warnings() {
        println!("warning: {w}");
    }
    let problems = r.problems();
    for p in &problems {
        println!("PROBLEM: {p}");
    }
    problems.is_empty()
}

fn run(cmd: Cmd) -> Result<bool, Box<dyn std::error::Error>> {
    Ok(match cmd {
        Cmd::Inspect { dir } => {
            let r = admin::inspect(&dir)?;
            print_report(&r, "");
            verdict(&r)
        }
        Cmd::Verify { dir } => {
            let r = admin::inspect(&dir)?;
            let ok = verdict(&r);
            if ok {
                println!("ok");
            }
            ok
        }
        Cmd::Backup { dir, dest } => {
            let rep = admin::backup(&dir, &dest)?;
            println!("copied {} files, {} bytes", rep.files, rep.bytes);
            true
        }
        Cmd::Restore { backup, dest } => {
            let (rep, report) = admin::restore(&backup, &dest)?;
            println!("copied {} files, {} bytes", rep.files, rep.bytes);
            verdict(&report)
        }
        Cmd::Export { dir, out } => {
            let index = admin::open_dir(&dir, None)?;
            let n = if out == Path::new("-") {
                admin::export_jsonl(&index, BufWriter::new(std::io::stdout().lock()))?
            } else {
                admin::export_jsonl(&index, BufWriter::new(File::create(&out)?))?
            };
            eprintln!("exported {n} vectors");
            true
        }
        Cmd::Import { dir, input, shape } => {
            let index = admin::open_dir(&dir, shape.config())?;
            let threads = std::thread::available_parallelism().map_or(2, |n| n.get());
            let n = admin::import_jsonl(&index, BufReader::new(File::open(&input)?), threads)?;
            index.checkpoint()?;
            println!("imported {n} vectors, {} live", index.len());
            true
        }
    })
}

fn main() -> ExitCode {
    match run(Cli::parse().cmd) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

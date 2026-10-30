//! Offline tools for a data directory: inspect, verify, back up, restore, JSONL export/import.
//! They work on files, so for anything that writes (restore, import) the server should be stopped.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::collections::{CONFIG_FILE, DROPPED, SUBDIR};
use crate::config::Config;
use crate::distance::Metric;
use crate::durable::{Durability, DurableIndex, scan_wal, wal_files};
use crate::error::{Error, Result};
use crate::filter::{Payload, Value};
use crate::hnsw::{Index, Item};
use crate::snapshots;

const CODEBOOK: &str = "codebook.bin";
const PAGE: usize = 1000;

pub struct SnapshotInfo {
    pub generation: u64,
    pub path: PathBuf,
    pub bytes: u64,
    pub config: Config,
    pub live: usize,
    pub tombstone_ratio: f64,
}

pub struct WalInfo {
    pub generation: u64,
    pub path: PathBuf,
    pub records: usize,
    pub valid_bytes: u64,
    pub file_bytes: u64,
}

#[derive(Default)]
pub struct DirReport {
    pub snapshot: Option<SnapshotInfo>,
    pub snapshot_error: Option<String>,
    pub wals: Vec<WalInfo>,
    pub has_codebook: bool,
    pub collections: Vec<(String, DirReport)>,
}

impl DirReport {
    /// Things that mean data is lost or unreadable.
    pub fn problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.collect(String::new(), &mut out, true);
        out
    }

    /// Things worth knowing that recovery handles on its own (a torn tail on the newest log).
    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.collect(String::new(), &mut out, false);
        out
    }

    fn collect(&self, prefix: String, out: &mut Vec<String>, problems: bool) {
        if problems && let Some(e) = &self.snapshot_error {
            out.push(format!("{prefix}snapshot: {e}"));
        }
        let newest = self.wals.iter().map(|w| w.generation).max();
        for w in &self.wals {
            if w.valid_bytes < w.file_bytes {
                let torn_tail = Some(w.generation) == newest;
                if problems != torn_tail {
                    out.push(format!(
                        "{prefix}{}: {} of {} bytes are intact ({} records)",
                        w.path.file_name().and_then(|n| n.to_str()).unwrap_or("wal"),
                        w.valid_bytes,
                        w.file_bytes,
                        w.records
                    ));
                }
            }
        }
        for (name, c) in &self.collections {
            c.collect(format!("{prefix}[{name}] "), out, problems);
        }
    }
}

fn subdirs(dir: &Path) -> Vec<(String, PathBuf)> {
    let Ok(rd) = std::fs::read_dir(dir.join(SUBDIR)) else {
        return vec![];
    };
    let mut v: Vec<_> = rd
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir() && !e.path().join(DROPPED).exists())
        .filter_map(|e| Some((e.file_name().to_str()?.to_owned(), e.path())))
        .collect();
    v.sort();
    v
}

/// Read-only: opens the newest snapshot (which checks its CRCs) and scans the logs
/// without applying them. Also looks into every collection.
pub fn inspect(dir: &Path) -> Result<DirReport> {
    let mut r = DirReport::default();
    if let Some((generation, path)) = snapshots::latest(dir) {
        match Index::open(&path) {
            Ok(idx) => {
                r.snapshot = Some(SnapshotInfo {
                    generation,
                    bytes: std::fs::metadata(&path)?.len(),
                    config: idx.config().clone(),
                    live: idx.len(),
                    tombstone_ratio: idx.tombstone_ratio(),
                    path,
                })
            }
            Err(e) => r.snapshot_error = Some(e.to_string()),
        }
    }
    for (generation, path) in wal_files(dir) {
        let data = std::fs::read(&path)?;
        let (records, valid_bytes) = scan_wal(&data);
        r.wals.push(WalInfo { generation, path, records, valid_bytes, file_bytes: data.len() as u64 });
    }
    r.has_codebook = dir.join(CODEBOOK).exists();
    for (name, path) in subdirs(dir) {
        r.collections.push((name, inspect(&path)?));
    }
    Ok(r)
}

#[derive(Debug, Default)]
pub struct CopyReport {
    pub files: usize,
    pub bytes: u64,
}

fn copy_into(src: &Path, dst: &Path, rep: &mut CopyReport) -> Result<()> {
    std::fs::copy(src, dst).map(|n| {
        rep.files += 1;
        rep.bytes += n;
    })?;
    Ok(())
}

// newest snapshot, the logs that sit on top of it, codebook and config; then every collection
fn copy_tree(src: &Path, dst: &Path, rep: &mut CopyReport) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    let base = match snapshots::latest(src) {
        Some((g, p)) => {
            copy_into(&p, &dst.join(p.file_name().unwrap()), rep)?;
            g
        }
        None => 0,
    };
    for (g, p) in wal_files(src) {
        if g >= base {
            copy_into(&p, &dst.join(p.file_name().unwrap()), rep)?;
        }
    }
    for name in [CODEBOOK, CONFIG_FILE] {
        if src.join(name).exists() {
            copy_into(&src.join(name), &dst.join(name), rep)?;
        }
    }
    for (name, path) in subdirs(src) {
        copy_tree(&path, &dst.join(SUBDIR).join(name), rep)?;
    }
    Ok(())
}

fn empty_or_missing(p: &Path) -> bool {
    std::fs::read_dir(p).map_or(true, |mut d| d.next().is_none())
}

/// Copies a data directory. Fine while the server runs: logs are append-only and a torn
/// tail is cut by recovery. If a snapshot is cut right while copying, a log disappears
/// under us; that is retried a few times.
pub fn backup(src: &Path, dst: &Path) -> Result<CopyReport> {
    if !empty_or_missing(dst) {
        return Err(Error::BadInput(format!("{} is not empty", dst.display())));
    }
    let mut last = None;
    for _ in 0..3 {
        let mut rep = CopyReport::default();
        match copy_tree(src, dst, &mut rep) {
            Ok(()) => return Ok(rep),
            Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                last = Some(Error::Io(e));
                let _ = std::fs::remove_dir_all(dst);
            }
            Err(e) => return Err(e),
        }
    }
    Err(last.unwrap())
}

/// A backup is a normal data directory, so restoring is copying it to an empty place
/// and checking the result.
pub fn restore(backup_dir: &Path, dst: &Path) -> Result<(CopyReport, DirReport)> {
    let rep = backup(backup_dir, dst)?;
    Ok((rep, inspect(dst)?))
}

/// Opens a data directory (or one collection directory) for offline use. The config comes
/// from the snapshot, or from `collection.cfg`, or else from `fallback`.
pub fn open_dir(dir: &Path, fallback: Option<Config>) -> Result<DurableIndex> {
    let from_file = std::fs::read(dir.join(CONFIG_FILE)).ok();
    let cfg = match (from_file, fallback) {
        (Some(raw), _) => postcard::from_bytes(&raw)?,
        (None, Some(c)) => c,
        (None, None) if snapshots::latest(dir).is_some() => Config::new(1, Metric::L2), // unused, snapshot wins
        (None, None) => {
            return Err(Error::InvalidConfig(
                "no snapshot in this directory, say what the index looks like".into(),
            ));
        }
    };
    DurableIndex::open(dir, cfg, Durability::Flush)
}

#[derive(Serialize)]
struct OutLine<'a> {
    id: u64,
    vector: &'a [f32],
    payload: serde_json::Map<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct InLine {
    id: u64,
    vector: Vec<f32>,
    #[serde(default)]
    payload: serde_json::Map<String, serde_json::Value>,
}

fn bad(e: impl std::fmt::Display) -> Error {
    Error::BadInput(e.to_string())
}

fn value_json(v: &Value) -> Result<serde_json::Value> {
    Ok(match v {
        Value::Bool(b) => (*b).into(),
        Value::Int(i) => (*i).into(),
        Value::Float(f) => {
            serde_json::Number::from_f64(*f).ok_or_else(|| bad("NaN or infinite payload float"))?.into()
        }
        Value::Str(s) => s.as_str().into(),
    })
}

/// One JSON object per line: `{"id":1,"vector":[..],"payload":{..}}`. Vectors are what `get`
/// returns (decoded for quantized indexes, normalised for cosine).
pub fn export_jsonl(index: &DurableIndex, mut w: impl Write) -> Result<usize> {
    let (mut cursor, mut n) = (0, 0);
    loop {
        let page = index.scroll(None, cursor, PAGE, true);
        for r in &page.items {
            let payload =
                r.payload.iter().map(|(k, v)| Ok((k.clone(), value_json(v)?))).collect::<Result<_>>()?;
            serde_json::to_writer(&mut w, &OutLine { id: r.id, vector: &r.vector, payload }).map_err(bad)?;
            w.write_all(b"\n")?;
            n += 1;
        }
        match page.next {
            Some(c) => cursor = c,
            None => return Ok(n),
        }
    }
}

fn payload_from(m: serde_json::Map<String, serde_json::Value>, line: usize) -> Result<Payload> {
    use serde_json::Value as J;
    m.into_iter()
        .map(|(k, v)| {
            let v = match v {
                J::Bool(b) => Value::Bool(b),
                J::String(s) => Value::Str(s),
                J::Number(n) => match n.as_i64() {
                    Some(i) => Value::Int(i),
                    None => Value::Float(
                        n.as_f64().ok_or_else(|| bad(format!("line {line}: unusable number in {k:?}")))?,
                    ),
                },
                _ => {
                    return Err(bad(format!(
                        "line {line}: payload field {k:?} must be a bool, number or string"
                    )));
                }
            };
            Ok((k, v))
        })
        .collect()
}

/// Reads what `export_jsonl` writes. Items go in with the normal insert path, in batches;
/// the first bad line or insert error stops the import (what came before stays).
pub fn import_jsonl(index: &DurableIndex, input: impl BufRead, threads: usize) -> Result<usize> {
    let (mut batch, mut total): (Vec<Item>, usize) = (Vec::with_capacity(PAGE), 0);
    for (i, line) in input.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let l: InLine = serde_json::from_str(&line).map_err(|e| bad(format!("line {}: {e}", i + 1)))?;
        batch.push(Item { id: l.id, vector: l.vector, payload: payload_from(l.payload, i + 1)? });
        if batch.len() == PAGE {
            index.insert_bulk(&batch, threads)?;
            total += batch.len();
            batch.clear();
        }
    }
    index.insert_bulk(&batch, threads)?;
    Ok(total + batch.len())
}

//! Snapshot + write-ahead log on top of `Index`.
//!
//! Layout of the data directory:
//!   snapshot-<gen>.vdb   full state as of the moment generation <gen> was cut
//!   wal-<gen>.log        every mutation applied after that snapshot
//!   codebook.bin         pq only: trained once, before the first insert
//!
//! Recovery = open the newest snapshot, replay every wal with gen >= snapshot gen.
//! Replay is idempotent (duplicate inserts / missing deletes are skipped), which
//! is what makes the "wal rotated but snapshot not finished" crash window harmless.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::{Condvar, Mutex, MutexGuard, RwLock};
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::error::{Error, Result};
use crate::filter::Payload;
use crate::filter::{Filter, Value};
use crate::hnsw::{Hit, Index, Item, Page, Record, SearchParams, Stats};
use crate::pq::Codebook;
use crate::quant::Quantization;
use crate::snapshots;

const CODEBOOK: &str = "codebook.bin";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Durability {
    /// hand records to the OS before acknowledging; survives a process crash
    #[default]
    Flush,
    /// fsync before acknowledging; survives power loss, costs latency
    Fsync,
}

#[derive(Serialize)]
enum OpRef<'a> {
    Insert { id: u64, vector: &'a [f32], payload: &'a Payload },
    Delete(u64),
    Upsert { id: u64, vector: &'a [f32], payload: &'a Payload },
    SetPayload { id: u64, payload: &'a Payload },
    MergePayload { id: u64, payload: &'a Payload },
}

#[derive(Deserialize, Clone)]
enum Op {
    Insert { id: u64, vector: Vec<f32>, payload: Payload },
    Delete(u64),
    Upsert { id: u64, vector: Vec<f32>, payload: Payload },
    SetPayload { id: u64, payload: Payload },
    MergePayload { id: u64, payload: Payload },
}

pub struct DurableIndex {
    // swapped wholesale by online compaction; readers clone the Arc and keep using the old one
    index: RwLock<Arc<Index>>,
    cfg: Config,
    dir: Option<PathBuf>,
    mode: Durability,
    wal: RwLock<Option<Arc<Wal>>>,
    generation: Mutex<u64>,
    // mutations hold it shared across apply+log, checkpoint and the compaction swap hold it exclusive
    gate: RwLock<()>,
    // ops applied while a compaction is rebuilding, replayed onto the new index before the swap
    tail: Mutex<Vec<Op>>,
    recording: AtomicBool,
    compacting: AtomicBool,
}

fn wal_path(dir: &Path, gen_no: u64) -> PathBuf {
    dir.join(format!("wal-{gen_no:012}.log"))
}

pub(crate) fn wal_files(dir: &Path) -> Vec<(u64, PathBuf)> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return vec![];
    };
    let mut v: Vec<_> = rd
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let p = e.path();
            let n = p.file_name()?.to_str()?;
            let g = n.strip_prefix("wal-")?.strip_suffix(".log")?.parse().ok()?;
            Some((g, p))
        })
        .collect();
    v.sort();
    v
}

struct WalState {
    pending: Vec<u8>,
    // each append call takes the next ticket; durable_seq is the highest ticket on disk
    next_seq: u64,
    durable_seq: u64,
    flushing: bool,
    failed: Option<String>,
}

/// Group commit: concurrent writers queue their records, one of them becomes the
/// leader and writes (and fsyncs) everything queued so far in a single go while
/// the others wait on the condvar.
struct Wal {
    st: Mutex<WalState>,
    // only the current leader touches it, so it is never contended
    file: Mutex<File>,
    cv: Condvar,
    syncs: AtomicU64,
}

impl Wal {
    fn open(path: &Path) -> Result<Wal> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Wal {
            file: Mutex::new(file),
            st: Mutex::new(WalState {
                pending: Vec::new(),
                next_seq: 0,
                durable_seq: 0,
                flushing: false,
                failed: None,
            }),
            cv: Condvar::new(),
            syncs: AtomicU64::new(0),
        })
    }

    fn append(&self, recs: &[Vec<u8>], fsync: bool) -> Result<()> {
        let mut st = self.st.lock();
        if let Some(e) = &st.failed {
            return Err(Error::Io(std::io::Error::other(e.clone())));
        }
        for r in recs {
            st.pending.extend_from_slice(r);
        }
        st.next_seq += 1;
        let mine = st.next_seq;

        loop {
            if st.durable_seq >= mine {
                return Ok(());
            }
            if let Some(e) = &st.failed {
                return Err(Error::Io(std::io::Error::other(e.clone())));
            }
            if st.flushing {
                self.cv.wait(&mut st);
                continue;
            }
            // become the leader for everything queued right now
            st.flushing = true;
            let batch = std::mem::take(&mut st.pending);
            let upto = st.next_seq;
            // I/O happens with the queue unlocked so new writers can keep joining
            let io = MutexGuard::unlocked(&mut st, || write_batch(&mut self.file.lock(), &batch, fsync));
            st.flushing = false;
            match io {
                Ok(()) => {
                    st.durable_seq = upto;
                    self.syncs.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => st.failed = Some(e.to_string()),
            }
            self.cv.notify_all();
        }
    }
}

fn write_batch(f: &mut File, batch: &[u8], fsync: bool) -> std::io::Result<()> {
    f.write_all(batch)?;
    if fsync {
        f.sync_data()?;
    }
    Ok(())
}

fn encode(op: &OpRef<'_>) -> Result<Vec<u8>> {
    let body = postcard::to_stdvec(op)?;
    let mut rec = Vec::with_capacity(body.len() + 8);
    rec.extend_from_slice(&(body.len() as u32).to_le_bytes());
    rec.extend_from_slice(&crc32fast::hash(&body).to_le_bytes());
    rec.extend_from_slice(&body);
    Ok(rec)
}

// ops that fail the same way they failed the first time are not a recovery problem
fn skippable(e: &Error) -> bool {
    matches!(
        e,
        Error::DuplicateId(_)
            | Error::NotFound(_)
            | Error::DimensionMismatch { .. }
            | Error::InvalidVector(_)
    )
}

fn apply(index: &Index, op: Op) -> Result<()> {
    match op {
        Op::Insert { id, vector, payload } => index.insert(id, &vector, payload),
        Op::Delete(id) => index.delete(id),
        Op::Upsert { id, vector, payload } => index.upsert(id, &vector, payload),
        Op::SetPayload { id, payload } => index.set_payload(id, payload),
        Op::MergePayload { id, payload } => index.merge_payload(id, payload),
    }
}

/// Walks the intact records of a log image; stops at the first torn or damaged one.
struct Records<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Iterator for Records<'_> {
    type Item = Op;

    fn next(&mut self) -> Option<Op> {
        let (d, pos) = (self.data, self.pos);
        if pos + 8 > d.len() {
            return None;
        }
        let n = u32::from_le_bytes(d[pos..pos + 4].try_into().unwrap()) as usize;
        let crc = u32::from_le_bytes(d[pos + 4..pos + 8].try_into().unwrap());
        let body = d.get(pos + 8..pos + 8 + n)?;
        if crc32fast::hash(body) != crc {
            return None;
        }
        let op = postcard::from_bytes::<Op>(body).ok()?;
        self.pos = pos + 8 + n;
        Some(op)
    }
}

/// (intact records, bytes they cover), without applying anything
pub(crate) fn scan_wal(data: &[u8]) -> (usize, u64) {
    let mut r = Records { data, pos: 0 };
    let n = (&mut r).count();
    (n, r.pos as u64)
}

/// Applies every intact record, returns (records applied, bytes that were valid).
fn replay(index: &Index, path: &Path) -> Result<(usize, u64)> {
    let data = std::fs::read(path)?;
    let mut recs = Records { data: &data, pos: 0 };
    let mut applied = 0;
    for op in &mut recs {
        match apply(index, op) {
            Err(e) if !skippable(&e) => return Err(e),
            _ => applied += 1,
        }
    }
    Ok((applied, recs.pos as u64))
}

impl DurableIndex {
    /// no files, no log: a plain index behind the same API
    pub fn in_memory(index: Index) -> Self {
        DurableIndex::assemble(index, None, Durability::Flush, None, 0)
    }

    /// First step for a PQ index: trains the codebook and stores it in `dir`, so that
    /// `open` (and wal replay after a crash) can rebuild the index before any snapshot exists.
    pub fn train_pq(dir: impl AsRef<Path>, cfg: &Config, sample: &[f32], threads: usize) -> Result<()> {
        let Quantization::Pq { m } = cfg.quantization else {
            return Err(Error::InvalidConfig("train_pq needs Quantization::Pq".into()));
        };
        cfg.validate()?;
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        let path = dir.join(CODEBOOK);
        if path.exists() {
            return Err(Error::InvalidConfig("codebook already exists in this directory".into()));
        }
        let book = Index::train_codebook(cfg, m, sample, threads)?;
        let tmp = path.with_extension("tmp");
        let mut f = File::create(&tmp)?;
        f.write_all(&book.to_bytes())?;
        f.sync_all()?;
        drop(f);
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    /// Recovers from `dir` (newest snapshot + wal replay), or starts empty with `cfg`.
    pub fn open(dir: impl AsRef<Path>, cfg: Config, mode: Durability) -> Result<Self> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;

        let (index, snap_gen) = match snapshots::latest(dir) {
            Some((g, p)) => (Index::open(&p)?, g),
            None if matches!(cfg.quantization, Quantization::Pq { .. }) => {
                let raw = std::fs::read(dir.join(CODEBOOK)).map_err(|_| {
                    Error::InvalidConfig(
                        "pq index: call DurableIndex::train_pq on this directory first".into(),
                    )
                })?;
                (Index::with_codebook(cfg, Codebook::from_bytes(&raw)?)?, 0)
            }
            None => (Index::new(cfg)?, 0),
        };

        let mut newest_wal = snap_gen;
        for (g, p) in wal_files(dir) {
            if g < snap_gen {
                let _ = std::fs::remove_file(&p);
                continue;
            }
            let (n, valid) = replay(&index, &p)?;
            let len = std::fs::metadata(&p)?.len();
            if valid < len {
                tracing::warn!(path = %p.display(), dropped_bytes = len - valid, "truncating torn wal tail");
                OpenOptions::new().write(true).open(&p)?.set_len(valid)?;
            }
            tracing::info!(path = %p.display(), records = n, "wal replayed");
            newest_wal = newest_wal.max(g);
        }

        let out = Arc::new(Wal::open(&wal_path(dir, newest_wal))?);
        Ok(DurableIndex::assemble(index, Some(dir.to_path_buf()), mode, Some(out), snap_gen))
    }

    fn assemble(index: Index, dir: Option<PathBuf>, mode: Durability, wal: Option<Arc<Wal>>, gen_no: u64) -> Self {
        DurableIndex {
            cfg: index.config().clone(),
            index: RwLock::new(Arc::new(index)),
            dir,
            mode,
            wal: RwLock::new(wal),
            generation: Mutex::new(gen_no),
            gate: RwLock::new(()),
            tail: Mutex::new(Vec::new()),
            recording: AtomicBool::new(false),
            compacting: AtomicBool::new(false),
        }
    }

    /// The index as it is right now. A compaction swap doesn't affect handles already taken.
    pub fn index(&self) -> Arc<Index> {
        self.index.read().clone()
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn len(&self) -> usize {
        self.index().len()
    }

    pub fn is_empty(&self) -> bool {
        self.index().is_empty()
    }

    pub fn contains(&self, id: u64) -> bool {
        self.index().contains(id)
    }

    pub fn get(&self, id: u64) -> Option<Record> {
        self.index().get(id)
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>> {
        self.index().search(query, k)
    }

    pub fn search_with(&self, query: &[f32], p: &SearchParams) -> Result<Vec<Hit>> {
        self.index().search_with(query, p)
    }

    pub fn search_batch(&self, queries: &[Vec<f32>], p: &SearchParams, threads: usize) -> Result<Vec<Vec<Hit>>> {
        self.index().search_batch(queries, p, threads)
    }

    pub fn search_range(&self, query: &[f32], radius: f32, p: &SearchParams) -> Result<Vec<Hit>> {
        self.index().search_range(query, radius, p)
    }

    pub fn recommend(&self, positive: &[u64], negative: &[u64], p: &SearchParams) -> Result<Vec<Hit>> {
        self.index().recommend(positive, negative, p)
    }

    pub fn search_mmr(&self, query: &[f32], p: &SearchParams, lambda: f32, fetch: usize) -> Result<Vec<Hit>> {
        self.index().search_mmr(query, p, lambda, fetch)
    }

    pub fn count(&self, filter: Option<&Filter>) -> usize {
        self.index().count(filter)
    }

    pub fn scroll(&self, filter: Option<&Filter>, cursor: u64, limit: usize, with_vector: bool) -> Page {
        self.index().scroll(filter, cursor, limit, with_vector)
    }

    pub fn facets(&self, field: &str, filter: Option<&Filter>, limit: usize) -> Vec<(Value, usize)> {
        self.index().facets(field, filter, limit)
    }

    pub fn enable_text_index(&self, field: &str) {
        self.index().enable_text_index(field)
    }

    pub fn search_text(&self, query: &str, p: &SearchParams) -> Result<Vec<Hit>> {
        self.index().search_text(query, p)
    }

    pub fn search_hybrid(&self, query: &[f32], text: &str, p: &SearchParams) -> Result<Vec<Hit>> {
        self.index().search_hybrid(query, text, p)
    }

    pub fn exact_search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>> {
        self.index().exact_search(query, k)
    }

    pub fn stats(&self) -> Stats {
        self.index().stats()
    }

    pub fn tombstone_ratio(&self) -> f64 {
        self.index().tombstone_ratio()
    }

    // the flag only flips under the exclusive gate, so it is stable for a whole mutation
    fn capture<T>(&self, f: impl FnOnce() -> T) -> Option<T> {
        self.recording.load(Ordering::Acquire).then(f)
    }

    fn push<I: IntoIterator<Item = Op>>(&self, ops: Option<I>) {
        if let Some(ops) = ops {
            self.tail.lock().extend(ops);
        }
    }

    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    pub fn durability(&self) -> Durability {
        self.mode
    }

    pub fn is_durable(&self) -> bool {
        self.dir.is_some()
    }

    fn append(&self, recs: &[Vec<u8>]) -> Result<()> {
        let wal = self.wal.read().clone();
        match wal {
            Some(w) => w.append(recs, self.mode == Durability::Fsync),
            None => Ok(()),
        }
    }

    /// number of write batches that reached the log; with concurrent writers
    /// this is smaller than the number of acknowledged writes
    pub fn wal_batches(&self) -> u64 {
        self.wal.read().as_ref().map_or(0, |w| w.syncs.load(Ordering::Relaxed))
    }

    pub fn insert(&self, id: u64, vector: &[f32], payload: Payload) -> Result<()> {
        let _g = self.gate.read();
        let rec = self.dir.is_some().then(|| encode(&OpRef::Insert { id, vector, payload: &payload }));
        let op = self.capture(|| Op::Insert { id, vector: vector.to_vec(), payload: payload.clone() });
        self.index().insert(id, vector, payload)?;
        self.push(op.map(|o| [o]));
        match rec {
            Some(r) => self.append(&[r?]),
            None => Ok(()),
        }
    }

    pub fn upsert(&self, id: u64, vector: &[f32], payload: Payload) -> Result<()> {
        let _g = self.gate.read();
        let rec = self.dir.is_some().then(|| encode(&OpRef::Upsert { id, vector, payload: &payload }));
        let op = self.capture(|| Op::Upsert { id, vector: vector.to_vec(), payload: payload.clone() });
        self.index().upsert(id, vector, payload)?;
        self.push(op.map(|o| [o]));
        match rec {
            Some(r) => self.append(&[r?]),
            None => Ok(()),
        }
    }

    pub fn set_payload(&self, id: u64, payload: Payload) -> Result<()> {
        let _g = self.gate.read();
        let rec = self.dir.is_some().then(|| encode(&OpRef::SetPayload { id, payload: &payload }));
        let op = self.capture(|| Op::SetPayload { id, payload: payload.clone() });
        self.index().set_payload(id, payload)?;
        self.push(op.map(|o| [o]));
        match rec {
            Some(r) => self.append(&[r?]),
            None => Ok(()),
        }
    }

    pub fn merge_payload(&self, id: u64, payload: Payload) -> Result<()> {
        let _g = self.gate.read();
        let rec = self.dir.is_some().then(|| encode(&OpRef::MergePayload { id, payload: &payload }));
        let op = self.capture(|| Op::MergePayload { id, payload: payload.clone() });
        self.index().merge_payload(id, payload)?;
        self.push(op.map(|o| [o]));
        match rec {
            Some(r) => self.append(&[r?]),
            None => Ok(()),
        }
    }

    pub fn insert_bulk(&self, items: &[Item], threads: usize) -> Result<()> {
        let _g = self.gate.read();
        if self.dir.is_some() {
            let recs = items
                .iter()
                .map(|it| encode(&OpRef::Insert { id: it.id, vector: &it.vector, payload: &it.payload }))
                .collect::<Result<Vec<_>>>()?;
            self.append(&recs)?;
        }
        let ops = self.capture(|| {
            items
                .iter()
                .map(|it| Op::Insert { id: it.id, vector: it.vector.clone(), payload: it.payload.clone() })
                .collect::<Vec<_>>()
        });
        self.index().insert_bulk(items, threads)?;
        self.push(ops);
        Ok(())
    }

    pub fn delete(&self, id: u64) -> Result<()> {
        let _g = self.gate.read();
        let op = self.capture(|| Op::Delete(id));
        self.index().delete(id)?;
        self.push(op.map(|o| [o]));
        if self.dir.is_some() {
            self.append(&[encode(&OpRef::Delete(id))?])?;
        }
        Ok(())
    }

    /// Compacts, then persists the result as a new snapshot generation.
    pub fn compact(&mut self, threads: usize) -> Result<()> {
        self.compact_online(threads)?;
        self.checkpoint()?;
        Ok(())
    }

    /// Rebuilds the graph without tombstones while reads and writes keep going.
    /// Writes that land during the rebuild are replayed onto the new index just before
    /// the swap; only that replay holds writers up. The swap is in memory only, the
    /// compacted state reaches disk with the next checkpoint (until then recovery
    /// replays the log onto the old snapshot, which gives the same data).
    /// Returns false if another compaction is already running.
    pub fn compact_online(&self, threads: usize) -> Result<bool> {
        if self.compacting.swap(true, Ordering::AcqRel) {
            return Ok(false);
        }
        let res = self.rebuild_and_swap(threads);
        self.compacting.store(false, Ordering::Release);
        res.map(|_| true)
    }

    fn rebuild_and_swap(&self, threads: usize) -> Result<()> {
        {
            let _w = self.gate.write();
            self.tail.lock().clear();
            self.recording.store(true, Ordering::Release);
        }
        let built = self.index().compacted(threads);

        let _w = self.gate.write();
        self.recording.store(false, Ordering::Release);
        let tail = std::mem::take(&mut *self.tail.lock());
        let fresh = built?;
        for op in tail {
            match apply(&fresh, op) {
                Err(e) if !skippable(&e) => return Err(e),
                _ => {}
            }
        }
        *self.index.write() = Arc::new(fresh);
        Ok(())
    }

    /// Cuts a new snapshot generation and retires the wal files it covers.
    /// Mutations wait for the duration; searches don't.
    pub fn checkpoint(&self) -> Result<Option<(PathBuf, u64)>> {
        let Some(dir) = &self.dir else {
            return Ok(None);
        };
        let _w = self.gate.write();
        let mut gen_no = self.generation.lock();
        let next = *gen_no + 1;

        let new_wal = Arc::new(Wal::open(&wal_path(dir, next))?);
        let path = snapshots::path_for(dir, next);
        self.index().save(&path)?;

        *self.wal.write() = Some(new_wal);
        *gen_no = next;
        snapshots::prune(dir, 2);
        for (g, p) in wal_files(dir) {
            if g < next {
                let _ = std::fs::remove_file(p);
            }
        }
        Ok(Some((path, self.len() as u64)))
    }
}

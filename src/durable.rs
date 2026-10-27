//! Snapshot + write-ahead log on top of `Index`.
//!
//! Layout of the data directory:
//!   snapshot-<gen>.vdb   full state as of the moment generation <gen> was cut
//!   wal-<gen>.log        every mutation applied after that snapshot
//!
//! Recovery = open the newest snapshot, replay every wal with gen >= snapshot gen.
//! Replay is idempotent (duplicate inserts / missing deletes are skipped), which
//! is what makes the "wal rotated but snapshot not finished" crash window harmless.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::{Condvar, Mutex, MutexGuard, RwLock};
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::error::{Error, Result};
use crate::filter::Payload;
use crate::hnsw::{Index, Item};
use crate::snapshots;

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
}

#[derive(Deserialize)]
enum Op {
    Insert { id: u64, vector: Vec<f32>, payload: Payload },
    Delete(u64),
}

pub struct DurableIndex {
    index: Index,
    dir: Option<PathBuf>,
    mode: Durability,
    wal: RwLock<Option<Arc<Wal>>>,
    generation: Mutex<u64>,
    // mutations hold it shared across apply+log, checkpoint holds it exclusive
    gate: RwLock<()>,
}

impl Deref for DurableIndex {
    type Target = Index;
    fn deref(&self) -> &Index {
        &self.index
    }
}

fn wal_path(dir: &Path, gen_no: u64) -> PathBuf {
    dir.join(format!("wal-{gen_no:012}.log"))
}

fn wal_files(dir: &Path) -> Vec<(u64, PathBuf)> {
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

/// Applies every intact record, returns (records applied, bytes that were valid).
fn replay(index: &Index, path: &Path) -> Result<(usize, u64)> {
    let data = std::fs::read(path)?;
    let (mut pos, mut applied) = (0usize, 0usize);
    while pos + 8 <= data.len() {
        let n = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        let crc = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap());
        let Some(body) = data.get(pos + 8..pos + 8 + n) else {
            break;
        };
        if crc32fast::hash(body) != crc {
            break;
        }
        let Ok(op) = postcard::from_bytes::<Op>(body) else {
            break;
        };
        let res = match op {
            Op::Insert { id, vector, payload } => index.insert(id, &vector, payload),
            Op::Delete(id) => index.delete(id),
        };
        match res {
            Err(e) if !skippable(&e) => return Err(e),
            _ => applied += 1,
        }
        pos += 8 + n;
    }
    Ok((applied, pos as u64))
}

impl DurableIndex {
    /// no files, no log: a plain index behind the same API
    pub fn in_memory(index: Index) -> Self {
        DurableIndex {
            index,
            dir: None,
            mode: Durability::Flush,
            wal: RwLock::new(None),
            generation: Mutex::new(0),
            gate: RwLock::new(()),
        }
    }

    /// Recovers from `dir` (newest snapshot + wal replay), or starts empty with `cfg`.
    pub fn open(dir: impl AsRef<Path>, cfg: Config, mode: Durability) -> Result<Self> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;

        let (index, snap_gen) = match snapshots::latest(dir) {
            Some((g, p)) => (Index::open(&p)?, g),
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
        Ok(DurableIndex {
            index,
            dir: Some(dir.to_path_buf()),
            mode,
            wal: RwLock::new(Some(out)),
            generation: Mutex::new(snap_gen),
            gate: RwLock::new(()),
        })
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
        self.index.insert(id, vector, payload)?;
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
        self.index.insert_bulk(items, threads)
    }

    pub fn delete(&self, id: u64) -> Result<()> {
        let _g = self.gate.read();
        self.index.delete(id)?;
        if self.dir.is_some() {
            self.append(&[encode(&OpRef::Delete(id))?])?;
        }
        Ok(())
    }

    /// Rebuilds the graph without tombstones and persists the result as a new
    /// snapshot generation. Needs exclusive access, so run it at start-up,
    /// before the index is shared.
    pub fn compact(&mut self, threads: usize) -> Result<()> {
        self.index = self.index.compacted(threads)?;
        self.checkpoint()?;
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
        self.index.save(&path)?;

        *self.wal.write() = Some(new_wal);
        *gen_no = next;
        snapshots::prune(dir, 2);
        for (g, p) in wal_files(dir) {
            if g < next {
                let _ = std::fs::remove_file(p);
            }
        }
        Ok(Some((path, self.index.len() as u64)))
    }
}

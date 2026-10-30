use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::config::Config;
use crate::durable::{Durability, DurableIndex};
use crate::error::{Error, Result};
use crate::hnsw::Index;
use crate::quant::Quantization;

pub const DEFAULT: &str = "default";

const SUBDIR: &str = "collections";
const CONFIG_FILE: &str = "collection.cfg";
// dropping while a request still maps the snapshot can't delete the files on Windows,
// so the marker is what keeps the collection dead until the next start-up cleans it
const DROPPED: &str = "dropped";

/// Named indexes behind one server. "default" is whatever index the process was started
/// with (so a single-index data directory keeps its old layout); the others live in
/// `<data dir>/collections/<name>/` and remember their own config.
pub struct Collections {
    root: Option<PathBuf>,
    mode: Durability,
    map: RwLock<BTreeMap<String, Arc<DurableIndex>>>,
    text_field: RwLock<Option<String>>,
}

fn valid_name(n: &str) -> bool {
    !n.is_empty() && n.len() <= 64 && n.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

impl Collections {
    pub fn new(default: Arc<DurableIndex>) -> Self {
        Collections {
            root: default.dir().map(|d| d.join(SUBDIR)),
            mode: default.durability(),
            map: RwLock::new(BTreeMap::from([(DEFAULT.to_owned(), default)])),
            text_field: RwLock::new(None),
        }
    }

    /// Enables BM25 on this payload field for every collection, existing and future.
    pub fn set_text_field(&self, field: &str) {
        *self.text_field.write() = Some(field.to_owned());
        for (_, idx) in self.list() {
            idx.enable_text_index(field);
        }
    }

    fn with_text(&self, idx: DurableIndex) -> Arc<DurableIndex> {
        if let Some(f) = self.text_field.read().as_deref() {
            idx.enable_text_index(f);
        }
        Arc::new(idx)
    }

    /// Opens every collection found on disk. Call once, before serving.
    pub fn load_existing(&self) -> Result<usize> {
        let Some(root) = &self.root else {
            return Ok(0);
        };
        let Ok(rd) = std::fs::read_dir(root) else {
            return Ok(0);
        };
        let mut n = 0;
        for e in rd.filter_map(|e| e.ok()) {
            let dir = e.path();
            let Some(name) = dir.file_name().and_then(|s| s.to_str()).map(str::to_owned) else {
                continue;
            };
            if dir.join(DROPPED).exists() {
                let _ = std::fs::remove_dir_all(&dir);
                continue;
            }
            let Ok(raw) = std::fs::read(dir.join(CONFIG_FILE)) else {
                continue;
            };
            let cfg: Config = postcard::from_bytes(&raw)?;
            let idx = DurableIndex::open(&dir, cfg, self.mode)?;
            self.map.write().insert(name, self.with_text(idx));
            n += 1;
        }
        Ok(n)
    }

    pub fn get(&self, name: &str) -> Option<Arc<DurableIndex>> {
        self.map.read().get(name).cloned()
    }

    pub fn list(&self) -> Vec<(String, Arc<DurableIndex>)> {
        self.map.read().iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }

    pub fn create(&self, name: &str, cfg: Config) -> Result<Arc<DurableIndex>> {
        if !valid_name(name) {
            return Err(Error::InvalidName(name.to_owned()));
        }
        if matches!(cfg.quantization, Quantization::Pq { .. }) {
            return Err(Error::InvalidConfig("pq collections need a trained codebook, create them from code".into()));
        }
        cfg.validate()?;

        let mut map = self.map.write();
        if map.contains_key(name) {
            return Err(Error::CollectionExists(name.to_owned()));
        }
        let idx = match &self.root {
            Some(root) => {
                let dir = root.join(name);
                if dir.join(DROPPED).exists() {
                    // a previous drop could not delete its files yet
                    std::fs::remove_dir_all(&dir)?;
                }
                std::fs::create_dir_all(&dir)?;
                let tmp = dir.join("collection.tmp");
                std::fs::write(&tmp, postcard::to_stdvec(&cfg)?)?;
                std::fs::rename(&tmp, dir.join(CONFIG_FILE))?;
                DurableIndex::open(&dir, cfg, self.mode)?
            }
            None => DurableIndex::in_memory(Index::new(cfg)?),
        };
        let idx = self.with_text(idx);
        map.insert(name.to_owned(), idx.clone());
        Ok(idx)
    }

    /// Removes the collection and its files. Requests already holding it finish normally.
    pub fn drop_collection(&self, name: &str) -> Result<()> {
        if name == DEFAULT {
            return Err(Error::InvalidName(format!("{DEFAULT} cannot be dropped")));
        }
        let mut map = self.map.write();
        let idx = map.remove(name).ok_or_else(|| Error::CollectionNotFound(name.to_owned()))?;
        if let Some(dir) = idx.dir() {
            let dir = dir.to_path_buf();
            std::fs::write(dir.join(DROPPED), b"")?;
            drop(idx);
            drop(map);
            if let Err(e) = std::fs::remove_dir_all(&dir) {
                tracing::warn!(collection = name, error = %e, "files stay until the next start-up");
            }
        }
        Ok(())
    }
}

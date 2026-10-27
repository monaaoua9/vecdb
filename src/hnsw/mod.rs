use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use parking_lot::RwLock;

use crate::config::Config;
use crate::distance::Space;
use crate::error::{Error, Result};
use crate::filter::{Filter, FilterStrategy, Payload, PayloadIndex};
use crate::store::{Segment, VectorStore};

mod compact;
mod graph;
mod insert;
mod scratch;
mod search;

pub(crate) use graph::{AppendVec, Node};

const MAX_LEVEL: usize = 16;

#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub id: u64,
    pub distance: f32,
}

#[derive(Clone, Debug)]
pub struct SearchParams {
    pub k: usize,
    /// beam width on layer 0, falls back to `Config::ef_search`
    pub ef: Option<usize>,
    pub filter: Option<Filter>,
    pub strategy: FilterStrategy,
}

impl SearchParams {
    pub fn new(k: usize) -> Self {
        SearchParams { k, ef: None, filter: None, strategy: FilterStrategy::Auto }
    }

    pub fn ef(mut self, ef: usize) -> Self {
        self.ef = Some(ef);
        self
    }

    pub fn filter(mut self, f: Filter) -> Self {
        self.filter = Some(f);
        self
    }

    pub fn strategy(mut self, s: FilterStrategy) -> Self {
        self.strategy = s;
        self
    }
}

#[derive(Clone, Debug)]
pub struct Item {
    pub id: u64,
    pub vector: Vec<f32>,
    pub payload: Payload,
}

#[derive(Clone, Debug)]
pub struct Record {
    pub id: u64,
    pub vector: Vec<f32>,
    pub payload: Payload,
}

#[derive(Clone, Debug)]
pub struct Stats {
    pub live: usize,
    pub deleted: usize,
    pub capacity: usize,
    pub max_level: usize,
    pub layer0_edges: usize,
    pub kernel: &'static str,
    pub heap_vector_bytes: usize,
    pub mapped_bytes: usize,
}

pub struct Index {
    pub(crate) cfg: Config,
    pub(crate) space: Space,
    pub(crate) store: VectorStore,
    pub(crate) nodes: AppendVec<Node>,
    pub(crate) ids: RwLock<HashMap<u64, u32>>,
    pub(crate) entry: RwLock<Option<(u32, u8)>>,
    pub(crate) next_id: AtomicUsize,
    pub(crate) live: AtomicUsize,
    // lets unfiltered searches skip the per-candidate node lookup while nothing is deleted
    pub(crate) tombstones: AtomicUsize,
    pub(crate) pidx: PayloadIndex,
    // writers (insert/delete) hold this shared, save() takes it exclusive
    pub(crate) gate: RwLock<()>,
    rng: AtomicU64,
    level_mult: f64,
}

impl Index {
    pub fn new(cfg: Config) -> Result<Index> {
        cfg.validate()?;
        Ok(Self::assemble(cfg, None))
    }

    pub(crate) fn assemble(cfg: Config, base: Option<Segment>) -> Index {
        let space = Space::new(cfg.metric, cfg.dim, cfg.quantization);
        let store = VectorStore::new(cfg.dim, cfg.quantization, cfg.max_elements, base);
        Index {
            space,
            store,
            nodes: AppendVec::with_capacity(cfg.max_elements),
            ids: RwLock::new(HashMap::new()),
            entry: RwLock::new(None),
            next_id: AtomicUsize::new(0),
            live: AtomicUsize::new(0),
            tombstones: AtomicUsize::new(0),
            pidx: PayloadIndex::default(),
            gate: RwLock::new(()),
            rng: AtomicU64::new(cfg.seed),
            level_mult: 1.0 / (cfg.m as f64).ln(),
            cfg,
        }
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn len(&self) -> usize {
        self.live.load(Ordering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, id: u64) -> bool {
        self.lookup(id).is_some()
    }

    fn lookup(&self, ext_id: u64) -> Option<(u32, &Node)> {
        let id = *self.ids.read().get(&ext_id)?;
        let n = self.nodes.get(id as usize)?;
        (!n.deleted.load(Ordering::Relaxed)).then_some((id, n))
    }

    pub fn get(&self, ext_id: u64) -> Option<Record> {
        let (id, n) = self.lookup(ext_id)?;
        let mut buf = Vec::new();
        let v = self.store.decode(id as usize, &mut buf)?.to_vec();
        Some(Record { id: ext_id, vector: v, payload: n.payload.clone() })
    }

    pub fn delete(&self, ext_id: u64) -> Result<()> {
        let _g = self.gate.read();
        let (id, node) = {
            let mut ids = self.ids.write();
            let id = *ids.get(&ext_id).ok_or(Error::NotFound(ext_id))?;
            let node = self.nodes.get(id as usize).ok_or(Error::NotFound(ext_id))?;
            ids.remove(&ext_id);
            (id, node)
        };
        // tombstone only: the node keeps routing queries, it just never shows up in results
        if !node.deleted.swap(true, Ordering::AcqRel) {
            self.pidx.remove(id, &node.payload);
            self.live.fetch_sub(1, Ordering::Relaxed);
            self.tombstones.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    pub fn stats(&self) -> Stats {
        let n = self.next_id.load(Ordering::Acquire);
        let mut edges = 0;
        for i in 0..n {
            if let Some(node) = self.nodes.get(i) {
                edges += node.links[0].read().len();
            }
        }
        let live = self.len();
        Stats {
            live,
            deleted: n.saturating_sub(live),
            capacity: self.cfg.max_elements,
            max_level: self.entry.read().map_or(0, |(_, l)| l as usize),
            layer0_edges: edges,
            kernel: self.space.kernel_name(),
            heap_vector_bytes: self.store.heap_bytes(),
            mapped_bytes: self.store.mapped_bytes(),
        }
    }

    fn random_level(&self) -> u8 {
        let mut z = self.rng.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let u = ((z >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        ((-u.ln() * self.level_mult) as usize).min(MAX_LEVEL) as u8
    }
}

const _: fn() = || {
    fn assert_sync<T: Send + Sync>() {}
    assert_sync::<Index>();
};

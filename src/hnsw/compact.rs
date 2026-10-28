use std::sync::atomic::Ordering;

use super::{Index, Item};
use crate::error::Result;

const CHUNK: usize = 8192;

impl Index {
    /// share of allocated nodes that are tombstones
    pub fn tombstone_ratio(&self) -> f64 {
        let total = self.next_id.load(Ordering::Acquire);
        if total == 0 {
            return 0.0;
        }
        total.saturating_sub(self.len()) as f64 / total as f64
    }

    /// Builds a fresh index holding only the live vectors, so deleted nodes stop
    /// costing memory and graph hops. Goes through the normal insert path in
    /// chunks, so peak extra memory is one chunk of vectors plus the new index.
    /// SQ8 and PQ vectors are re-quantised from their decoded values (PQ keeps the same codebook).
    pub fn compacted(&self, threads: usize) -> Result<Index> {
        let mut cfg = self.cfg.clone();
        cfg.max_elements = cfg.max_elements.max(self.len());
        let out = match self.store.codebook() {
            Some(b) => Index::assemble(cfg, None, Some(b.clone())),
            None => Index::new(cfg)?,
        };

        let total = self.next_id.load(Ordering::Acquire);
        let mut buf = Vec::new();
        let mut chunk: Vec<Item> = Vec::with_capacity(CHUNK);
        for i in 0..total {
            let Some(node) = self.nodes.get(i) else { continue };
            if node.deleted.load(Ordering::Relaxed) {
                continue;
            }
            let Some(v) = self.store.decode(i, &mut buf) else { continue };
            chunk.push(Item { id: node.ext_id, vector: v.to_vec(), payload: node.payload.clone() });
            if chunk.len() == CHUNK {
                out.insert_bulk(&chunk, threads)?;
                chunk.clear();
            }
        }
        out.insert_bulk(&chunk, threads)?;
        Ok(out)
    }
}

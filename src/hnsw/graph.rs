use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;

use parking_lot::RwLock;

use crate::filter::Payload;

const CHUNK: usize = 1024;

/// Write-once slots addressed by index. Slots never move, so a reference
/// handed out by `get` stays valid while other threads keep filling the table.
pub(crate) struct AppendVec<T> {
    chunks: Box<[OnceLock<Chunk<T>>]>,
}

type Chunk<T> = Box<[OnceLock<T>]>;

impl<T> AppendVec<T> {
    pub fn with_capacity(n: usize) -> Self {
        AppendVec { chunks: (0..n.div_ceil(CHUNK)).map(|_| OnceLock::new()).collect() }
    }

    pub fn get(&self, i: usize) -> Option<&T> {
        self.chunks.get(i / CHUNK)?.get()?[i % CHUNK].get()
    }

    /// false if the slot was already taken or `i` is out of range
    pub fn set(&self, i: usize, v: T) -> bool {
        let Some(c) = self.chunks.get(i / CHUNK) else {
            return false;
        };
        let c = c.get_or_init(|| (0..CHUNK).map(|_| OnceLock::new()).collect());
        c[i % CHUNK].set(v).is_ok()
    }
}

pub(crate) struct Node {
    pub ext_id: u64,
    pub deleted: AtomicBool,
    pub payload: Payload,
    /// one adjacency list per layer, 0..=level
    pub links: Box<[RwLock<Vec<u32>>]>,
}

impl Node {
    pub fn new(ext_id: u64, level: u8, payload: Payload, m: usize) -> Self {
        let links = (0..=level as usize)
            .map(|l| RwLock::new(Vec::with_capacity(if l == 0 { m * 2 } else { m })))
            .collect();
        Node { ext_id, deleted: AtomicBool::new(false), payload, links }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_vec_basics() {
        let v = AppendVec::<u32>::with_capacity(3000);
        assert!(v.get(2500).is_none());
        assert!(v.set(2500, 7));
        assert!(!v.set(2500, 8));
        assert_eq!(v.get(2500), Some(&7));
        assert!(!v.set(100_000, 1));
        assert!(v.get(100_000).is_none());
    }
}

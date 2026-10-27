//! Snapshot file layout (little endian):
//!
//! ```text
//! 0      header, 4096 bytes (72 used)
//! 4096   vector records, count * stride_words * 4 bytes, mmap'd on open
//! ...    metadata blob (postcard): config, entry point, adjacency lists, payloads
//! ```

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::Ordering;

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::error::{Error, Result};
use crate::filter::Payload;
use crate::hnsw::{Index, Node};
use crate::store::Segment;

const MAGIC: &[u8; 8] = b"VECDBHNS";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 4096;
const HEADER_USED: usize = 72;

#[derive(Serialize, Deserialize)]
struct NodeMeta {
    ext_id: u64,
    deleted: bool,
    links: Vec<Vec<u32>>,
}

#[derive(Serialize, Deserialize)]
struct Meta {
    config: Config,
    entry: Option<(u32, u8)>,
    nodes: Vec<NodeMeta>,
    payloads: Vec<Payload>,
}

struct Header {
    count: u64,
    stride_words: u64,
    vec_off: u64,
    vec_len: u64,
    meta_off: u64,
    meta_len: u64,
    meta_crc: u32,
}

impl Header {
    fn encode(&self) -> [u8; HEADER_SIZE] {
        let mut b = [0u8; HEADER_SIZE];
        b[0..8].copy_from_slice(MAGIC);
        b[8..12].copy_from_slice(&VERSION.to_le_bytes());
        b[16..24].copy_from_slice(&self.count.to_le_bytes());
        b[24..32].copy_from_slice(&self.stride_words.to_le_bytes());
        b[32..40].copy_from_slice(&self.vec_off.to_le_bytes());
        b[40..48].copy_from_slice(&self.vec_len.to_le_bytes());
        b[48..56].copy_from_slice(&self.meta_off.to_le_bytes());
        b[56..64].copy_from_slice(&self.meta_len.to_le_bytes());
        b[64..68].copy_from_slice(&self.meta_crc.to_le_bytes());
        let crc = crc32fast::hash(&b[16..HEADER_USED]);
        b[12..16].copy_from_slice(&crc.to_le_bytes());
        b
    }

    fn decode(b: &[u8]) -> Result<Header> {
        if b.len() < HEADER_SIZE || &b[0..8] != MAGIC {
            return Err(Error::Corrupt("bad magic".into()));
        }
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        if u32_at(8) != VERSION {
            return Err(Error::Corrupt(format!("unsupported version {}", u32_at(8))));
        }
        if u32_at(12) != crc32fast::hash(&b[16..HEADER_USED]) {
            return Err(Error::Corrupt("header checksum mismatch".into()));
        }
        Ok(Header {
            count: u64_at(16),
            stride_words: u64_at(24),
            vec_off: u64_at(32),
            vec_len: u64_at(40),
            meta_off: u64_at(48),
            meta_len: u64_at(56),
            meta_crc: u32_at(64),
        })
    }
}

impl Index {
    /// Writes a consistent snapshot. Inserts and deletes wait while it runs;
    /// searches don't. The file appears atomically (temp file + rename).
    /// On Windows the target can't be replaced while an index still maps it.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let tmp = path.with_extension("tmp");
        let _w = self.gate.write();

        let count = self.next_id.load(Ordering::Acquire);
        let stride = self.store.stride_words();

        let mut meta = Meta {
            config: self.cfg.clone(),
            entry: *self.entry.read(),
            nodes: Vec::with_capacity(count),
            payloads: Vec::with_capacity(count),
        };
        for i in 0..count {
            let n =
                self.nodes.get(i).ok_or_else(|| Error::Corrupt(format!("node {i} missing during save")))?;
            meta.nodes.push(NodeMeta {
                ext_id: n.ext_id,
                deleted: n.deleted.load(Ordering::Relaxed),
                links: n.links.iter().map(|l| l.read().clone()).collect(),
            });
            meta.payloads.push(n.payload.clone());
        }
        let meta_bytes = postcard::to_stdvec(&meta)?;

        let mut w = BufWriter::new(File::create(&tmp)?);
        w.write_all(&[0u8; HEADER_SIZE])?;
        for i in 0..count {
            let words = self
                .store
                .words(i)
                .ok_or_else(|| Error::Corrupt(format!("vector {i} missing during save")))?;
            w.write_all(bytemuck::cast_slice(words))?;
        }
        let vec_len = (count * stride * 4) as u64;
        w.write_all(&meta_bytes)?;

        let header = Header {
            count: count as u64,
            stride_words: stride as u64,
            vec_off: HEADER_SIZE as u64,
            vec_len,
            meta_off: HEADER_SIZE as u64 + vec_len,
            meta_len: meta_bytes.len() as u64,
            meta_crc: crc32fast::hash(&meta_bytes),
        };
        w.seek(SeekFrom::Start(0))?;
        w.write_all(&header.encode())?;
        let f = w.into_inner().map_err(|e| e.into_error())?;
        f.sync_all()?;
        drop(f);

        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Maps the snapshot and rebuilds the graph. Vector data stays on disk and
    /// is paged in by the OS; new inserts go to memory on top of it.
    pub fn open(path: impl AsRef<Path>) -> Result<Index> {
        Self::open_with(path, None)
    }

    /// Same as `open`, optionally raising `max_elements`.
    pub fn open_with(path: impl AsRef<Path>, max_elements: Option<usize>) -> Result<Index> {
        let file = File::open(path.as_ref())?;
        // SAFETY: the file is treated as immutable while mapped; save() never
        // rewrites in place, it renames a new file over the old path.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        let h = Header::decode(&mmap)?;
        let (vec_off, vec_len) = (h.vec_off as usize, h.vec_len as usize);
        let (meta_off, meta_len) = (h.meta_off as usize, h.meta_len as usize);
        if meta_off.checked_add(meta_len).is_none_or(|e| e > mmap.len()) {
            return Err(Error::Corrupt("metadata runs past end of file".into()));
        }
        let meta_bytes = &mmap[meta_off..meta_off + meta_len];
        if crc32fast::hash(meta_bytes) != h.meta_crc {
            return Err(Error::Corrupt("metadata checksum mismatch".into()));
        }
        let meta: Meta = postcard::from_bytes(meta_bytes)?;

        let mut cfg = meta.config;
        if let Some(cap) = max_elements {
            if cap < h.count as usize {
                return Err(Error::InvalidConfig("max_elements is below the stored count".into()));
            }
            cfg.max_elements = cap;
        }
        cfg.validate()?;

        let count = h.count as usize;
        if meta.nodes.len() != count || meta.payloads.len() != count {
            return Err(Error::Corrupt("node count mismatch".into()));
        }
        if h.stride_words as usize != cfg.quantization.stride_words(cfg.dim)
            || vec_len != count * h.stride_words as usize * 4
        {
            return Err(Error::Corrupt("vector layout does not match config".into()));
        }
        if count > cfg.max_elements {
            return Err(Error::Corrupt("count exceeds max_elements".into()));
        }

        let seg = Segment::new(mmap, vec_off, h.stride_words as usize, count)?;
        let idx = Index::assemble(cfg, Some(seg));

        let mut live = 0;
        {
            let mut ids = idx.ids.write();
            for (i, (nm, payload)) in meta.nodes.into_iter().zip(meta.payloads).enumerate() {
                if nm.links.is_empty() || nm.links.iter().flatten().any(|&x| x as usize >= count) {
                    return Err(Error::Corrupt(format!("node {i} has invalid links")));
                }
                let node = Node::new(nm.ext_id, (nm.links.len() - 1) as u8, payload, idx.cfg.m);
                for (slot, l) in node.links.iter().zip(nm.links) {
                    *slot.write() = l;
                }
                if nm.deleted {
                    node.deleted.store(true, Ordering::Relaxed);
                } else {
                    ids.insert(nm.ext_id, i as u32);
                    idx.pidx.add(i as u32, &node.payload);
                    live += 1;
                }
                idx.nodes.set(i, node);
            }
        }
        if meta.entry.is_some_and(|(e, _)| e as usize >= count) {
            return Err(Error::Corrupt("entry point out of range".into()));
        }
        *idx.entry.write() = meta.entry;
        idx.next_id.store(count, Ordering::Release);
        idx.live.store(live, Ordering::Relaxed);
        Ok(idx)
    }
}

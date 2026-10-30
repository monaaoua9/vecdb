use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::scratch::{Cand, Scratch, scratch};
use super::search::Gate;
use super::{Index, Item, Node};
use crate::error::{Error, Result};
use crate::filter::Payload;

impl Index {
    pub fn insert(&self, ext_id: u64, vector: &[f32], payload: Payload) -> Result<()> {
        let vec = self.space.sanitize(vector)?;
        let _g = self.gate.read();

        let id = self.reserve(ext_id)?;
        self.place(id, ext_id, &vec, payload);
        Ok(())
    }

    /// Insert, or replace the vector and payload when `ext_id` already exists.
    /// The new node is fully linked before the id points at it, so `get` never misses;
    /// the old node is tombstoned (it keeps routing until the next compaction and
    /// uses one capacity slot). A search racing the swap can see both versions.
    pub fn upsert(&self, ext_id: u64, vector: &[f32], payload: Payload) -> Result<()> {
        let vec = self.space.sanitize(vector)?;
        let _g = self.gate.read();

        let id = self.next_slot(&mut self.ids.write())?;
        self.place(id, ext_id, &vec, payload);
        let old = self.ids.write().insert(ext_id, id);
        if let Some(o) = old {
            self.retire(o);
        }
        Ok(())
    }

    fn place(&self, id: u32, ext_id: u64, vec: &[f32], payload: Payload) {
        let level = self.random_level();
        self.store.put(id as usize, vec);
        if let Some(f) = &self.full {
            f.put(id as usize, vec);
        }
        self.nodes.set(id as usize, Node::new(ext_id, level, payload, self.cfg.m));

        self.link(id, level, vec);

        // only now does the filter planner learn about the node
        let node = self.nodes.get(id as usize).unwrap();
        self.pidx.add(id, &node.payload.read());
        self.live.fetch_add(1, Ordering::Relaxed);
    }

    /// Parallel load. Stops at the first error; items inserted before that stay.
    pub fn insert_bulk(&self, items: &[Item], threads: usize) -> Result<()> {
        let next = AtomicUsize::new(0);
        let first_err = parking_lot::Mutex::new(None);
        std::thread::scope(|sc| {
            for _ in 0..threads.max(1) {
                sc.spawn(|| {
                    while first_err.lock().is_none() {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(it) = items.get(i) else { break };
                        if let Err(e) = self.insert(it.id, &it.vector, it.payload.clone()) {
                            first_err.lock().get_or_insert(e);
                        }
                    }
                });
            }
        });
        match first_err.into_inner() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn reserve(&self, ext_id: u64) -> Result<u32> {
        let mut ids = self.ids.write();
        if ids.contains_key(&ext_id) {
            return Err(Error::DuplicateId(ext_id));
        }
        let n = self.next_slot(&mut ids)?;
        ids.insert(ext_id, n);
        Ok(n)
    }

    // the id map lock is what serialises slot allocation
    fn next_slot(&self, _held: &mut HashMap<u64, u32>) -> Result<u32> {
        let n = self.next_id.load(Ordering::Relaxed);
        if n >= self.cfg.max_elements {
            return Err(Error::CapacityExceeded(self.cfg.max_elements));
        }
        self.next_id.store(n + 1, Ordering::Release);
        Ok(n as u32)
    }

    fn link(&self, id: u32, level: u8, vec: &[f32]) {
        // A node taller than the current entry point must hold the entry lock
        // for the whole insert, otherwise two such nodes could race for the top.
        let mut upgrade = None;
        let snap = *self.entry.read();
        if snap.is_none_or(|(_, l)| level > l) {
            let g = self.entry.write();
            if g.is_none_or(|(_, l)| level > l) {
                upgrade = Some(g);
            }
        }
        let cur = match &upgrade {
            Some(g) => **g,
            None => *self.entry.read(),
        };
        let Some((ep_id, top)) = cur else {
            if let Some(mut g) = upgrade {
                *g = Some((id, level));
            }
            return;
        };

        let q = self.space.prepare_indexed(vec);
        let mut s = scratch();

        let ep_dist = self.space.dist(&q, self.store.get(ep_id as usize).unwrap());
        let mut ep = Cand { dist: ep_dist, id: ep_id };
        for l in (level as usize + 1..=top as usize).rev() {
            ep = self.greedy(&q, ep, l, &mut s);
        }

        let node = self.nodes.get(id as usize).unwrap();
        let m = self.cfg.m;
        let mut eps = vec![ep];
        for l in (0..=level.min(top) as usize).rev() {
            let found = self.search_layer(&q, &eps, self.cfg.ef_construction, l, &Gate::All, &mut s);
            let chosen = self.select_neighbors(&found, m, &mut s);

            {
                let mut mine = node.links[l].write();
                for c in &chosen {
                    if !mine.contains(&c.id) {
                        mine.push(c.id);
                    }
                }
                let cap = self.layer_cap(l);
                if mine.len() > cap {
                    self.shrink(id, &mut mine, None, cap, &mut s);
                }
            }
            for c in &chosen {
                self.connect(c.id, id, l, &mut s);
            }
            eps = found;
        }

        if let Some(mut g) = upgrade {
            *g = Some((id, level));
        }
    }

    fn layer_cap(&self, layer: usize) -> usize {
        if layer == 0 { self.cfg.m * 2 } else { self.cfg.m }
    }

    // adds `new` to n's list at `layer`, pruning when the list is full
    fn connect(&self, n: u32, new: u32, layer: usize, s: &mut Scratch) {
        let Some(node) = self.nodes.get(n as usize) else {
            return;
        };
        let cap = self.layer_cap(layer);
        let mut links = node.links[layer].write();
        if links.contains(&new) {
            return;
        }
        if links.len() < cap {
            links.push(new);
            return;
        }
        self.shrink(n, &mut links, Some(new), cap, s);
    }

    // re-selects `owner`'s neighbours from its current list plus `extra`
    fn shrink(&self, owner: u32, links: &mut Vec<u32>, extra: Option<u32>, cap: usize, s: &mut Scratch) {
        let mut buf = std::mem::take(&mut s.buf_b);
        let Some(ov) = self.store.decode(owner as usize, &mut buf) else {
            s.buf_b = buf;
            return;
        };
        let oq = self.space.prepare(ov);

        let mut cands: Vec<Cand> = links
            .iter()
            .copied()
            .chain(extra)
            .filter_map(|x| {
                let v = self.store.get(x as usize)?;
                Some(Cand { dist: self.space.dist(&oq, v), id: x })
            })
            .collect();
        cands.sort_unstable();
        drop(oq);
        s.buf_b = buf;

        let keep = self.select_neighbors(&cands, cap, s);
        links.clear();
        links.extend(keep.iter().map(|c| c.id));
    }

    // the "heuristic" from the paper: take a candidate only if it is closer
    // to the query than to anything already taken
    pub(crate) fn select_neighbors(&self, sorted: &[Cand], m: usize, s: &mut Scratch) -> Vec<Cand> {
        let mut out: Vec<Cand> = Vec::with_capacity(m);
        for c in sorted {
            if out.len() >= m {
                break;
            }
            let Some(cv) = self.store.decode(c.id as usize, &mut s.buf_a) else {
                continue;
            };
            let cq = self.space.prepare(cv);
            let keep = out.iter().all(|sel| {
                self.store.get(sel.id as usize).is_none_or(|sv| self.space.dist(&cq, sv) >= c.dist)
            });
            if keep {
                out.push(*c);
            }
        }
        out
    }
}

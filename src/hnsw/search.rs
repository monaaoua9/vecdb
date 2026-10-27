use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::atomic::Ordering;

use roaring::RoaringBitmap;

use super::scratch::{Cand, Scratch, scratch};
use super::{Hit, Index, SearchParams};
use crate::distance::PreparedQuery;
use crate::error::Result;
use crate::filter::{Filter, FilterStrategy};

/// Decides which visited nodes may enter the result set. Traversal itself is
/// never restricted, so filtered queries can route through non-matching nodes.
pub(crate) enum Gate<'a> {
    /// index construction: tombstones are still valid link targets
    All,
    Live,
    Set {
        set: &'a RoaringBitmap,
        verify: Option<&'a Filter>,
    },
    Pred(&'a Filter),
}

impl Gate<'_> {
    #[inline]
    fn allows(&self, idx: &Index, id: u32) -> bool {
        if matches!(self, Gate::All) {
            return true;
        }
        let Some(n) = idx.nodes.get(id as usize) else {
            return false;
        };
        if n.deleted.load(Ordering::Relaxed) {
            return false;
        }
        match self {
            Gate::All | Gate::Live => true,
            Gate::Set { set, verify } => set.contains(id) && verify.is_none_or(|f| f.matches(&n.payload)),
            Gate::Pred(f) => f.matches(&n.payload),
        }
    }
}

impl Index {
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>> {
        self.search_with(query, &SearchParams::new(k))
    }

    pub fn search_with(&self, query: &[f32], p: &SearchParams) -> Result<Vec<Hit>> {
        let q = self.space.prepare_user(query)?;
        if p.k == 0 {
            return Ok(vec![]);
        }
        let ef = p.ef.unwrap_or(self.cfg.ef_search).max(p.k);
        let found = match &p.filter {
            None => self.ann(&q, p.k, ef, &Gate::Live),
            Some(f) => self.filtered(&q, p.k, ef, f, p.strategy),
        };
        Ok(self.to_hits(found))
    }

    /// Flat scan over every live vector. Ground truth for recall measurements.
    pub fn exact_search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>> {
        let q = self.space.prepare_user(query)?;
        let n = self.next_id.load(Ordering::Acquire);
        Ok(self.to_hits(self.scan(&q, k, 0..n as u32, None)))
    }

    fn to_hits(&self, found: Vec<Cand>) -> Vec<Hit> {
        found
            .into_iter()
            .filter_map(|c| {
                let n = self.nodes.get(c.id as usize)?;
                Some(Hit { id: n.ext_id, distance: c.dist })
            })
            .collect()
    }

    fn ann(&self, q: &PreparedQuery<'_>, k: usize, ef: usize, gate: &Gate<'_>) -> Vec<Cand> {
        let Some((ep_id, top)) = *self.entry.read() else {
            return vec![];
        };
        let mut s = scratch();
        let d = self.space.dist(q, self.store.get(ep_id as usize).unwrap());
        let mut ep = Cand { dist: d, id: ep_id };
        for l in (1..=top as usize).rev() {
            ep = self.greedy(q, ep, l, &mut s);
        }
        let mut out = self.search_layer(q, &[ep], ef.max(k), 0, gate, &mut s);
        out.truncate(k);
        out
    }

    fn filtered(
        &self,
        q: &PreparedQuery<'_>,
        k: usize,
        ef: usize,
        f: &Filter,
        strategy: FilterStrategy,
    ) -> Vec<Cand> {
        let total = self.next_id.load(Ordering::Acquire);
        match strategy {
            FilterStrategy::PostFilter => self.post_filter(q, k, ef, f, total),
            FilterStrategy::PreFilter => match self.pidx.plan(f) {
                Some((set, exact)) => self.scan(q, k, set.iter(), (!exact).then_some(f)),
                None => self.scan(q, k, 0..total as u32, Some(f)),
            },
            FilterStrategy::Auto => match self.pidx.plan(f) {
                Some((set, exact)) if (set.len() as usize) <= self.cfg.brute_force_below => {
                    self.scan(q, k, set.iter(), (!exact).then_some(f))
                }
                Some((set, exact)) => {
                    let gate = Gate::Set { set: &set, verify: (!exact).then_some(f) };
                    self.widen(q, k, ef, total, &gate)
                }
                None => self.widen(q, k, ef, total, &Gate::Pred(f)),
            },
        }
    }

    // A filtered walk can run dry before it finds k matches; retry with a
    // wider beam rather than returning a short list.
    fn widen(&self, q: &PreparedQuery<'_>, k: usize, ef: usize, total: usize, gate: &Gate<'_>) -> Vec<Cand> {
        let mut ef = ef;
        loop {
            let out = self.ann(q, k, ef, gate);
            if out.len() >= k || ef >= total {
                return out;
            }
            ef = (ef * 4).min(total.max(1));
        }
    }

    fn post_filter(&self, q: &PreparedQuery<'_>, k: usize, ef: usize, f: &Filter, total: usize) -> Vec<Cand> {
        let mut want = k * 4;
        loop {
            let raw = self.ann(q, want, ef.max(want), &Gate::Live);
            let exhausted = raw.len() < want || want >= total;
            let mut ok: Vec<Cand> = raw
                .into_iter()
                .filter(|c| self.nodes.get(c.id as usize).is_some_and(|n| f.matches(&n.payload)))
                .collect();
            if ok.len() >= k || exhausted {
                ok.truncate(k);
                return ok;
            }
            want *= 4;
        }
    }

    fn scan(
        &self,
        q: &PreparedQuery<'_>,
        k: usize,
        ids: impl Iterator<Item = u32>,
        pred: Option<&Filter>,
    ) -> Vec<Cand> {
        let mut heap: BinaryHeap<Cand> = BinaryHeap::with_capacity(k + 1);
        for id in ids {
            let Some(n) = self.nodes.get(id as usize) else {
                continue;
            };
            if n.deleted.load(Ordering::Relaxed) || pred.is_some_and(|f| !f.matches(&n.payload)) {
                continue;
            }
            let Some(v) = self.store.get(id as usize) else {
                continue;
            };
            let d = self.space.dist(q, v);
            if heap.len() < k {
                heap.push(Cand { dist: d, id });
            } else if d < heap.peek().unwrap().dist {
                heap.pop();
                heap.push(Cand { dist: d, id });
            }
        }
        heap.into_sorted_vec()
    }

    pub(crate) fn greedy(&self, q: &PreparedQuery<'_>, start: Cand, layer: usize, s: &mut Scratch) -> Cand {
        let mut cur = start;
        loop {
            s.nbrs.clear();
            if let Some(n) = self.nodes.get(cur.id as usize).filter(|n| layer < n.links.len()) {
                s.nbrs.extend_from_slice(&n.links[layer].read());
            }
            let mut moved = false;
            for &nb in &s.nbrs {
                let Some(v) = self.store.get(nb as usize) else {
                    continue;
                };
                let d = self.space.dist(q, v);
                if d < cur.dist {
                    cur = Cand { dist: d, id: nb };
                    moved = true;
                }
            }
            if !moved {
                return cur;
            }
        }
    }

    pub(crate) fn search_layer(
        &self,
        q: &PreparedQuery<'_>,
        eps: &[Cand],
        ef: usize,
        layer: usize,
        gate: &Gate<'_>,
        s: &mut Scratch,
    ) -> Vec<Cand> {
        let Scratch { visited, nbrs, cands, results, .. } = s;
        visited.reset();
        cands.clear();
        results.clear();

        for &e in eps {
            if visited.visit(e.id) {
                cands.push(Reverse(e));
                if gate.allows(self, e.id) {
                    results.push(e);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        while let Some(Reverse(c)) = cands.pop() {
            if results.len() >= ef && c.dist > results.peek().unwrap().dist {
                break;
            }
            nbrs.clear();
            match self.nodes.get(c.id as usize) {
                Some(n) if layer < n.links.len() => nbrs.extend_from_slice(&n.links[layer].read()),
                _ => continue,
            }
            nbrs.retain(|&n| visited.visit(n));
            for &n in nbrs.iter() {
                self.store.prefetch(n as usize);
            }
            for &n in nbrs.iter() {
                let Some(v) = self.store.get(n as usize) else {
                    continue;
                };
                let d = self.space.dist(q, v);
                if results.len() < ef || d < results.peek().unwrap().dist {
                    cands.push(Reverse(Cand { dist: d, id: n }));
                    if gate.allows(self, n) {
                        results.push(Cand { dist: d, id: n });
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut out: Vec<Cand> = results.drain().collect();
        out.sort_unstable();
        out
    }
}

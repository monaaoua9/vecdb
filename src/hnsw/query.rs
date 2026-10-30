use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;

use super::{Hit, Index, SearchParams};
use crate::distance::Space;
use crate::error::{Error, Result};
use crate::quant::Quantization;
use crate::store::VectorRef;

impl Index {
    /// Runs every query with the same params, spread over `threads` workers.
    /// Results come back in query order; the first error wins.
    pub fn search_batch(&self, queries: &[Vec<f32>], p: &SearchParams, threads: usize) -> Result<Vec<Vec<Hit>>> {
        let next = AtomicUsize::new(0);
        let done: Mutex<Vec<(usize, Result<Vec<Hit>>)>> = Mutex::new(Vec::with_capacity(queries.len()));
        std::thread::scope(|sc| {
            for _ in 0..threads.clamp(1, queries.len().max(1)) {
                sc.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(q) = queries.get(i) else { break };
                        let r = self.search_with(q, p);
                        done.lock().push((i, r));
                    }
                });
            }
        });
        let mut done = done.into_inner();
        done.sort_by_key(|(i, _)| *i);
        done.into_iter().map(|(_, r)| r).collect()
    }

    /// Everything within `radius` of the query, nearest first, at most `p.k` hits.
    /// Radius is in the reported distance (squared L2, 1 - cos, negative dot).
    pub fn search_range(&self, query: &[f32], radius: f32, p: &SearchParams) -> Result<Vec<Hit>> {
        let mut k = p.k.min(32);
        loop {
            let mut pp = p.clone();
            pp.k = k;
            let mut hits = self.search_with(query, &pp)?;
            let inside = hits.iter().take_while(|h| h.distance <= radius).count();
            // all of them inside and room to ask for more: there may be further matches
            if inside == hits.len() && hits.len() == k && k < p.k {
                k = (k * 2).min(p.k);
                continue;
            }
            hits.truncate(inside);
            return Ok(hits);
        }
    }

    /// "More like these, less like those": searches around
    /// `2 * mean(positive) - mean(negative)` (just the mean without negatives) and never
    /// returns the ids that were passed in.
    pub fn recommend(&self, positive: &[u64], negative: &[u64], p: &SearchParams) -> Result<Vec<Hit>> {
        if positive.is_empty() {
            return Err(Error::InvalidVector("recommend needs at least one positive id"));
        }
        let mean = |ids: &[u64]| -> Result<Vec<f32>> {
            let mut acc = vec![0.0f32; self.cfg.dim];
            for &id in ids {
                let r = self.get(id).ok_or(Error::NotFound(id))?;
                acc.iter_mut().zip(&r.vector).for_each(|(a, x)| *a += x);
            }
            acc.iter_mut().for_each(|a| *a /= ids.len() as f32);
            Ok(acc)
        };
        let mut q = mean(positive)?;
        if !negative.is_empty() {
            let neg = mean(negative)?;
            q.iter_mut().zip(&neg).for_each(|(a, n)| *a = 2.0 * *a - n);
        }

        let skip: HashSet<u64> = positive.iter().chain(negative).copied().collect();
        let mut pp = p.clone();
        pp.k = p.k + skip.len();
        let mut hits = self.search_with(&q, &pp)?;
        hits.retain(|h| !skip.contains(&h.id));
        hits.truncate(p.k);
        Ok(hits)
    }

    /// Maximal marginal relevance: takes the `fetch` nearest candidates, then picks `p.k` of
    /// them one by one, each time preferring a candidate that is close to the query
    /// (weight `lambda`) and far from what was already picked (weight `1 - lambda`).
    /// `lambda = 1` is plain nearest-neighbour order. Hits keep their distance to the query.
    pub fn search_mmr(&self, query: &[f32], p: &SearchParams, lambda: f32, fetch: usize) -> Result<Vec<Hit>> {
        if !(0.0..=1.0).contains(&lambda) {
            return Err(Error::InvalidVector("mmr lambda must be in 0..=1"));
        }
        let mut pp = p.clone();
        pp.k = fetch.max(p.k);
        let found = self.search_with(query, &pp)?;

        // a candidate deleted since the search just drops out
        let cands: Vec<(Hit, Vec<f32>)> =
            found.into_iter().filter_map(|h| Some((h.clone(), self.get(h.id)?.vector))).collect();
        let plain = Space::new(self.cfg.metric, self.cfg.dim, Quantization::None);
        let between = |a: &[f32], b: &[f32]| plain.dist(&plain.prepare(a), VectorRef::F32(b));

        let mut closest = vec![f32::INFINITY; cands.len()];
        let mut left: Vec<usize> = (0..cands.len()).collect();
        let mut out = Vec::with_capacity(p.k);
        while out.len() < p.k && !left.is_empty() {
            let score = |i: usize| {
                let diversity = if closest[i].is_finite() { (1.0 - lambda) * closest[i] } else { 0.0 };
                -lambda * cands[i].0.distance + diversity
            };
            let (pos, best) =
                left.iter().copied().enumerate().max_by(|x, y| score(x.1).total_cmp(&score(y.1))).unwrap();
            left.swap_remove(pos);
            for &i in &left {
                closest[i] = closest[i].min(between(&cands[i].1, &cands[best].1));
            }
            out.push(cands[best].0.clone());
        }
        Ok(out)
    }
}

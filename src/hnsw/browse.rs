use std::sync::atomic::Ordering;

use roaring::RoaringBitmap;

use super::{Index, Record};
use crate::filter::{Filter, Value};

pub struct Page {
    pub items: Vec<Record>,
    /// pass this back as the cursor to get the following page, `None` at the end
    pub next: Option<u64>,
}

impl Index {
    fn live_matches(&self, id: u32, f: &Filter) -> bool {
        self.nodes
            .get(id as usize)
            .is_some_and(|n| !n.deleted.load(Ordering::Relaxed) && f.matches(&n.payload.read()))
    }

    /// Live vectors matching the filter (all of them without one).
    pub fn count(&self, filter: Option<&Filter>) -> usize {
        let Some(f) = filter else {
            return self.len();
        };
        match self.pidx.plan(f) {
            Some((set, true)) => set.len() as usize,
            Some((set, false)) => set.iter().filter(|&id| self.live_matches(id, f)).count(),
            None => (0..self.next_id.load(Ordering::Acquire) as u32)
                .filter(|&id| self.live_matches(id, f))
                .count(),
        }
    }

    /// Walks live vectors in insertion order, `limit` at a time. Start with cursor 0.
    /// A cursor stays valid across inserts, deletes and payload edits; compaction renumbers
    /// the slots it points into and an upsert moves its id to the end, so a scroll that
    /// spans either should be restarted (or tolerate repeats).
    pub fn scroll(&self, filter: Option<&Filter>, cursor: u64, limit: usize, with_vector: bool) -> Page {
        let total = self.next_id.load(Ordering::Acquire) as u64;
        if limit == 0 || cursor >= total.min(u32::MAX as u64) {
            return Page { items: vec![], next: None };
        }
        let start = cursor as u32;

        let planned = filter.and_then(|f| self.pidx.plan(f));
        let (candidates, verify): (Box<dyn Iterator<Item = u32> + '_>, Option<&Filter>) = match planned {
            Some((set, exact)) => {
                let first = if start == 0 { 0 } else { set.rank(start - 1) as u32 };
                (Box::new((first..).map_while(move |j| set.select(j))), (!exact).then_some(filter).flatten())
            }
            None => (Box::new(start..total as u32), filter),
        };

        let mut items = Vec::with_capacity(limit);
        let mut buf = Vec::new();
        for id in candidates {
            let Some(n) = self.nodes.get(id as usize) else { continue };
            if n.deleted.load(Ordering::Relaxed) || verify.is_some_and(|f| !f.matches(&n.payload.read())) {
                continue;
            }
            if items.len() == limit {
                return Page { items, next: Some(id as u64) };
            }
            let vector = if with_vector {
                let store = self.full.as_ref().unwrap_or(&self.store);
                store.decode(id as usize, &mut buf).map(|v| v.to_vec()).unwrap_or_default()
            } else {
                Vec::new()
            };
            items.push(Record { id: n.ext_id, vector, payload: n.payload.read().clone() });
        }
        Page { items, next: None }
    }

    /// How many live vectors hold each value of `field`, most common first, optionally only
    /// among those matching `filter`. Float values aren't indexed and never show up here.
    pub fn facets(&self, field: &str, filter: Option<&Filter>, limit: usize) -> Vec<(Value, usize)> {
        let allowed: Option<RoaringBitmap> = filter.map(|f| match self.pidx.plan(f) {
            Some((set, true)) => set,
            Some((set, false)) => set.iter().filter(|&id| self.live_matches(id, f)).collect(),
            None => (0..self.next_id.load(Ordering::Acquire) as u32)
                .filter(|&id| self.live_matches(id, f))
                .collect(),
        });
        let mut out: Vec<(Value, usize)> = self
            .pidx
            .values_of(field)
            .into_iter()
            .map(|(v, ids)| (v, allowed.as_ref().map_or(ids.len(), |a| (&ids & a).len()) as usize))
            .filter(|(_, n)| *n > 0)
            .collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| format!("{:?}", a.0).cmp(&format!("{:?}", b.0))));
        out.truncate(limit);
        out
    }
}

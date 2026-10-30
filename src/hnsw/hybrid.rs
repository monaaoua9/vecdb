use std::collections::HashMap;
use std::sync::atomic::Ordering;

use super::{Hit, Index, SearchParams};
use crate::error::Result;
use crate::filter::Value;
use crate::text::TextIndex;

// the usual constant from the RRF paper
const RRF_K: f32 = 60.0;

impl Index {
    /// Turns on BM25 over the string payload field `field` and indexes what is already
    /// stored. Kept up to date by inserts, deletes and payload edits from then on; it lives
    /// in memory only, so call it again after `Index::open`.
    pub fn enable_text_index(&self, field: &str) {
        let mut t = TextIndex::default();
        for id in 0..self.next_id.load(Ordering::Acquire) as u32 {
            let Some(n) = self.nodes.get(id as usize) else { continue };
            if n.deleted.load(Ordering::Relaxed) {
                continue;
            }
            if let Some(Value::Str(s)) = n.payload.read().get(field) {
                t.add(id, s);
            }
        }
        *self.text.write() = Some((field.to_owned(), t));
    }

    pub fn text_field(&self) -> Option<String> {
        self.text.read().as_ref().map(|(f, _)| f.clone())
    }

    /// Keyword search only. `distance` is the negated BM25 score, so lower is still better.
    pub fn search_text(&self, query: &str, p: &SearchParams) -> Result<Vec<Hit>> {
        Ok(self.text_ranked(query, p.k, p)?.into_iter().map(|(id, s)| Hit { id, distance: -s }).collect())
    }

    /// Vector search and BM25 fused with reciprocal rank fusion: a hit's score is the sum of
    /// `1 / (60 + rank)` over the lists it appears in. `distance` is the negated fused score.
    /// Both lists are over-fetched to `4 * k` before fusing.
    pub fn search_hybrid(&self, query: &[f32], text: &str, p: &SearchParams) -> Result<Vec<Hit>> {
        let fetch = (p.k * 4).max(20);
        let mut vp = p.clone();
        vp.k = fetch;
        let by_vec = self.search_with(query, &vp)?;
        let by_text = self.text_ranked(text, fetch, p)?;

        let mut fused: HashMap<u64, f32> = HashMap::new();
        for (rank, h) in by_vec.iter().enumerate() {
            *fused.entry(h.id).or_default() += 1.0 / (RRF_K + rank as f32 + 1.0);
        }
        for (rank, (id, _)) in by_text.iter().enumerate() {
            *fused.entry(*id).or_default() += 1.0 / (RRF_K + rank as f32 + 1.0);
        }
        let mut out: Vec<Hit> = fused.into_iter().map(|(id, s)| Hit { id, distance: -s }).collect();
        out.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
        out.truncate(p.k);
        Ok(out)
    }

    fn text_ranked(&self, query: &str, limit: usize, p: &SearchParams) -> Result<Vec<(u64, f32)>> {
        let guard = self.text.read();
        let Some((_, t)) = guard.as_ref() else {
            return Err(crate::error::Error::InvalidConfig("text index is not enabled".into()));
        };
        let allow = |id: u32| {
            let Some(n) = self.nodes.get(id as usize) else { return false };
            !n.deleted.load(Ordering::Relaxed) && p.filter.as_ref().is_none_or(|f| f.matches(&n.payload.read()))
        };
        Ok(t.search(query, limit, allow)
            .into_iter()
            .filter_map(|(id, s)| Some((self.nodes.get(id as usize)?.ext_id, s)))
            .collect())
    }
}

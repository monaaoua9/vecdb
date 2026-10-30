use std::collections::HashMap;

const K1: f32 = 1.2;
const B: f32 = 0.75;

/// Lower-cases, drops Arabic diacritics and tatweel, folds alef/ya/ta-marbuta variants,
/// then splits on anything that isn't a letter or digit.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        let ch = match ch {
            '\u{064B}'..='\u{065F}' | '\u{0670}' | '\u{0640}' => continue,
            'أ' | 'إ' | 'آ' | 'ٱ' => 'ا',
            'ى' => 'ي',
            'ة' => 'ه',
            c => c,
        };
        if ch.is_alphanumeric() {
            cur.extend(ch.to_lowercase());
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// In-memory BM25 over one string field. Not persisted: it is rebuilt from the
/// payloads when enabled.
#[derive(Default)]
pub(crate) struct TextIndex {
    postings: HashMap<String, HashMap<u32, u32>>,
    // doc -> (length in tokens, distinct terms), needed to unindex a doc
    docs: HashMap<u32, (u32, Vec<String>)>,
    total_len: u64,
}

impl TextIndex {
    pub fn add(&mut self, id: u32, text: &str) {
        let toks = tokenize(text);
        if toks.is_empty() {
            return;
        }
        let mut tf: HashMap<String, u32> = HashMap::new();
        for t in &toks {
            *tf.entry(t.clone()).or_default() += 1;
        }
        let terms: Vec<String> = tf.keys().cloned().collect();
        for (t, n) in tf {
            self.postings.entry(t).or_default().insert(id, n);
        }
        self.total_len += toks.len() as u64;
        self.docs.insert(id, (toks.len() as u32, terms));
    }

    pub fn remove(&mut self, id: u32) {
        let Some((len, terms)) = self.docs.remove(&id) else {
            return;
        };
        self.total_len -= len as u64;
        for t in terms {
            if let Some(p) = self.postings.get_mut(&t) {
                p.remove(&id);
                if p.is_empty() {
                    self.postings.remove(&t);
                }
            }
        }
    }

    /// Best `limit` docs for the query, highest score first. `allow` can veto a doc.
    pub fn search(&self, query: &str, limit: usize, allow: impl Fn(u32) -> bool) -> Vec<(u32, f32)> {
        let n = self.docs.len() as f32;
        if n == 0.0 {
            return vec![];
        }
        let avg = self.total_len as f32 / n;
        let mut scores: HashMap<u32, f32> = HashMap::new();
        let mut seen = std::collections::HashSet::new();
        for term in tokenize(query) {
            if !seen.insert(term.clone()) {
                continue;
            }
            let Some(p) = self.postings.get(&term) else { continue };
            let df = p.len() as f32;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            for (&doc, &tf) in p {
                let len = self.docs[&doc].0 as f32;
                let tf = tf as f32;
                *scores.entry(doc).or_default() += idf * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * len / avg));
            }
        }
        let mut v: Vec<(u32, f32)> = scores.into_iter().filter(|(d, _)| allow(*d)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        v.truncate(limit);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arabic_folding() {
        assert_eq!(tokenize("الْكِتَابُ"), tokenize("الكتاب"));
        assert_eq!(tokenize("أحمد إلى آخر"), tokenize("احمد الي اخر"));
        assert_eq!(tokenize("مدرسة"), tokenize("مدرسه"));
        assert_eq!(tokenize("Hello, World-42!"), ["hello", "world", "42"]);
    }

    #[test]
    fn bm25_prefers_rarer_and_shorter() {
        let mut t = TextIndex::default();
        t.add(1, "the cat sat");
        t.add(2, "the dog sat on the mat with the cat and many other words");
        t.add(3, "the bird");
        let hits = t.search("cat", 10, |_| true);
        assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), [1, 2]);
        t.remove(1);
        assert_eq!(t.search("cat", 10, |_| true).len(), 1);
        assert!(t.search("zzz", 10, |_| true).is_empty());
    }
}

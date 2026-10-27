use std::cell::RefCell;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::ops::{Deref, DerefMut};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Cand {
    pub dist: f32,
    pub id: u32,
}

impl Eq for Cand {}

impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist.total_cmp(&o.dist).then(self.id.cmp(&o.id))
    }
}

impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Bitset plus a list of the words touched, so a reset costs O(nodes visited)
/// instead of O(index size) and the hot part stays small enough for L2
/// (1M nodes = 125 KiB, against 4 MiB for a u32-per-node epoch table).
#[derive(Default)]
pub(crate) struct Visited {
    bits: Vec<u64>,
    touched: Vec<u32>,
}

impl Visited {
    pub fn reset(&mut self) {
        for &w in &self.touched {
            self.bits[w as usize] = 0;
        }
        self.touched.clear();
    }

    /// true the first time an id is seen since the last reset
    #[inline]
    pub fn visit(&mut self, id: u32) -> bool {
        let w = (id >> 6) as usize;
        if w >= self.bits.len() {
            self.bits.resize((w + 1).next_power_of_two().max(256), 0);
        }
        let m = 1u64 << (id & 63);
        let word = self.bits[w];
        if word & m != 0 {
            return false;
        }
        if word == 0 {
            self.touched.push(w as u32);
        }
        self.bits[w] = word | m;
        true
    }
}

#[derive(Default)]
pub(crate) struct Scratch {
    pub visited: Visited,
    pub nbrs: Vec<u32>,
    pub cands: BinaryHeap<Reverse<Cand>>,
    pub results: BinaryHeap<Cand>,
    pub buf_a: Vec<f32>,
    pub buf_b: Vec<f32>,
}

thread_local! {
    static POOL: RefCell<Option<Box<Scratch>>> = const { RefCell::new(None) };
}

/// Per-thread reusable buffers. Taking it out of the thread-local (instead of
/// borrowing) keeps nested use from panicking.
pub(crate) struct ScratchGuard(Option<Box<Scratch>>);

pub(crate) fn scratch() -> ScratchGuard {
    let s = POOL.with(|p| p.borrow_mut().take()).unwrap_or_default();
    ScratchGuard(Some(s))
}

impl Deref for ScratchGuard {
    type Target = Scratch;
    fn deref(&self) -> &Scratch {
        self.0.as_ref().unwrap()
    }
}

impl DerefMut for ScratchGuard {
    fn deref_mut(&mut self) -> &mut Scratch {
        self.0.as_mut().unwrap()
    }
}

impl Drop for ScratchGuard {
    fn drop(&mut self) {
        if let Some(s) = self.0.take() {
            POOL.with(|p| *p.borrow_mut() = Some(s));
        }
    }
}

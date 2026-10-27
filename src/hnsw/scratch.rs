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

#[derive(Default)]
pub(crate) struct Visited {
    marks: Vec<u32>,
    epoch: u32,
}

impl Visited {
    pub fn reset(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.fill(0);
            self.epoch = 1;
        }
    }

    /// true the first time an id is seen since the last reset
    #[inline]
    pub fn visit(&mut self, id: u32) -> bool {
        let i = id as usize;
        if i >= self.marks.len() {
            self.marks.resize((i + 1).next_power_of_two().max(1024), 0);
        }
        if self.marks[i] == self.epoch {
            return false;
        }
        self.marks[i] = self.epoch;
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

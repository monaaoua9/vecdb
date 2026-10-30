use std::ptr::NonNull;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

use std::sync::Arc;

use crate::pq::Codebook;
use crate::quant::{self, Quantization, Sq8Params, code_words};

mod segment;
pub use segment::Segment;

const SLAB_RECORDS: usize = 1024;

const EMPTY: u8 = 0;
const WRITING: u8 = 1;
const READY: u8 = 2;

#[derive(Clone, Copy)]
pub enum VectorRef<'a> {
    F32(&'a [f32]),
    Sq8 { codes: &'a [u8], min: f32, scale: f32, norm_sq: f32 },
    Pq { codes: &'a [u8], book: &'a Codebook },
    Binary { bits: &'a [u32] },
}

struct Slab {
    data: NonNull<u32>,
    words: usize,
    state: Box<[AtomicU8]>,
}

// SAFETY: a record is written by exactly one thread (the CAS winner on `state`)
// and only read after it has published READY with Release ordering.
unsafe impl Send for Slab {}
unsafe impl Sync for Slab {}

impl Slab {
    fn new(stride: usize) -> Self {
        let words = stride * SLAB_RECORDS;
        let b = vec![0u32; words].into_boxed_slice();
        let data = NonNull::new(Box::into_raw(b) as *mut u32).unwrap();
        let state = (0..SLAB_RECORDS).map(|_| AtomicU8::new(EMPTY)).collect();
        Slab { data, words, state }
    }
}

impl Drop for Slab {
    fn drop(&mut self) {
        let raw = std::ptr::slice_from_raw_parts_mut(self.data.as_ptr(), self.words);
        drop(unsafe { Box::from_raw(raw) });
    }
}

/// Append-only record storage. Records live in fixed-size slabs that are
/// allocated on first touch, so growing never moves data a reader may hold.
/// Ids below `base_len` are served straight from the mmap'd segment.
pub struct VectorStore {
    dim: usize,
    quant: Quantization,
    stride: usize,
    capacity: usize,
    slabs: Box<[OnceLock<Slab>]>,
    base: Option<Segment>,
    base_len: usize,
    book: Option<Arc<Codebook>>,
}

impl VectorStore {
    pub fn new(
        dim: usize,
        quant: Quantization,
        capacity: usize,
        base: Option<Segment>,
        book: Option<Arc<Codebook>>,
    ) -> Self {
        let n_slabs = capacity.div_ceil(SLAB_RECORDS);
        let base_len = base.as_ref().map_or(0, |b| b.len());
        VectorStore {
            dim,
            quant,
            stride: quant.stride_words(dim),
            capacity,
            slabs: (0..n_slabs).map(|_| OnceLock::new()).collect(),
            base,
            base_len,
            book,
        }
    }

    pub fn codebook(&self) -> Option<&Arc<Codebook>> {
        self.book.as_ref()
    }

    pub fn stride_words(&self) -> usize {
        self.stride
    }

    pub fn base_len(&self) -> usize {
        self.base_len
    }

    /// `vec` must already be in index space (normalised for cosine).
    /// Returns false if the id is out of range or was already written.
    pub fn put(&self, id: usize, vec: &[f32]) -> bool {
        debug_assert_eq!(vec.len(), self.dim);
        if id < self.base_len || id >= self.capacity {
            return false;
        }
        let slab = self.slabs[id / SLAB_RECORDS].get_or_init(|| Slab::new(self.stride));
        let slot = id % SLAB_RECORDS;
        if slab.state[slot].compare_exchange(EMPTY, WRITING, Ordering::Acquire, Ordering::Relaxed).is_err() {
            return false;
        }

        // SAFETY: we hold the WRITING state for this slot, nobody else touches it
        let rec = unsafe {
            std::slice::from_raw_parts_mut(slab.data.as_ptr().add(slot * self.stride), self.stride)
        };
        match self.quant {
            Quantization::None => bytemuck::cast_slice_mut::<u32, f32>(rec).copy_from_slice(vec),
            Quantization::Sq8 => {
                let cw = code_words(self.dim);
                let (codes, tail) = rec.split_at_mut(cw);
                let p = quant::encode(vec, bytemuck::cast_slice_mut(codes));
                tail[0] = p.min.to_bits();
                tail[1] = p.scale.to_bits();
                tail[2] = p.norm_sq.to_bits();
            }
            Quantization::Pq { m } => {
                let bytes = bytemuck::cast_slice_mut::<u32, u8>(rec);
                self.book.as_ref().unwrap().encode(vec, &mut bytes[..m as usize]);
            }
            Quantization::Binary => quant::binarize(vec, rec),
        }
        slab.state[slot].store(READY, Ordering::Release);
        true
    }

    pub fn words(&self, id: usize) -> Option<&[u32]> {
        if id < self.base_len {
            return self.base.as_ref()?.words(id);
        }
        let slab = self.slabs.get(id / SLAB_RECORDS)?.get()?;
        let slot = id % SLAB_RECORDS;
        if slab.state[slot].load(Ordering::Acquire) != READY {
            return None;
        }
        // SAFETY: READY means the record was fully written and is never mutated again
        Some(unsafe { std::slice::from_raw_parts(slab.data.as_ptr().add(slot * self.stride), self.stride) })
    }

    #[inline]
    pub fn get(&self, id: usize) -> Option<VectorRef<'_>> {
        let w = self.words(id)?;
        Some(match self.quant {
            Quantization::None => VectorRef::F32(bytemuck::cast_slice(w)),
            Quantization::Sq8 => {
                let cw = code_words(self.dim);
                VectorRef::Sq8 {
                    codes: bytemuck::cast_slice(&w[..cw]),
                    min: f32::from_bits(w[cw]),
                    scale: f32::from_bits(w[cw + 1]),
                    norm_sq: f32::from_bits(w[cw + 2]),
                }
            }
            Quantization::Pq { m } => VectorRef::Pq {
                codes: &bytemuck::cast_slice(w)[..m as usize],
                book: self.book.as_deref().unwrap(),
            },
            Quantization::Binary => VectorRef::Binary { bits: w },
        })
    }

    /// Full-precision view of a stored vector. Borrowed for f32 storage,
    /// decoded into `buf` for quantized storage.
    pub fn decode<'a>(&'a self, id: usize, buf: &'a mut Vec<f32>) -> Option<&'a [f32]> {
        match self.get(id)? {
            VectorRef::F32(x) => Some(x),
            VectorRef::Sq8 { codes, min, scale, norm_sq } => {
                buf.resize(self.dim, 0.0);
                quant::decode(&codes[..self.dim], &Sq8Params { min, scale, norm_sq }, buf);
                Some(&buf[..])
            }
            VectorRef::Pq { codes, book } => {
                buf.resize(self.dim, 0.0);
                book.decode(codes, buf);
                Some(&buf[..])
            }
            VectorRef::Binary { bits } => {
                buf.resize(self.dim, 0.0);
                quant::unbinarize(bits, buf);
                Some(&buf[..])
            }
        }
    }

    #[inline]
    pub fn prefetch(&self, id: usize) {
        #[cfg(target_arch = "x86_64")]
        if let Some(w) = self.words(id) {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let p = w.as_ptr() as *const i8;
            let bytes = w.len() * 4;
            for off in (0..bytes).step_by(64).take(16) {
                // SAFETY: prefetch never faults, and `off` is inside the record anyway
                unsafe { _mm_prefetch::<_MM_HINT_T0>(p.add(off)) };
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = id;
    }

    pub fn heap_bytes(&self) -> usize {
        let slabs = self.slabs.iter().filter(|s| s.get().is_some()).count();
        slabs * SLAB_RECORDS * self.stride * 4
    }

    pub fn mapped_bytes(&self) -> usize {
        self.base.as_ref().map_or(0, |b| b.mapped_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_get_f32() {
        let s = VectorStore::new(3, Quantization::None, 5000, None, None);
        assert!(s.put(4000, &[1.0, 2.0, 3.0]));
        assert!(!s.put(4000, &[0.0; 3]));
        match s.get(4000).unwrap() {
            VectorRef::F32(x) => assert_eq!(x, &[1.0, 2.0, 3.0]),
            _ => panic!(),
        }
        assert!(s.get(4001).is_none());
        assert!(s.get(1).is_none());
    }

    #[test]
    fn put_get_sq8() {
        let s = VectorStore::new(5, Quantization::Sq8, 10, None, None);
        let v = [0.1, -0.4, 0.9, 0.0, 0.33];
        assert!(s.put(3, &v));
        let mut buf = Vec::new();
        let back = s.decode(3, &mut buf).unwrap();
        for (a, b) in v.iter().zip(back) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn put_get_binary() {
        let s = VectorStore::new(40, Quantization::Binary, 10, None, None);
        let v: Vec<f32> = (0..40).map(|i| if i % 2 == 0 { 0.3 } else { -0.3 }).collect();
        assert!(s.put(2, &v));
        let mut buf = Vec::new();
        let back = s.decode(2, &mut buf).unwrap();
        assert!(v.iter().zip(back).all(|(a, b)| a.signum() == *b));
    }

    #[test]
    fn out_of_range_put_is_rejected() {
        let s = VectorStore::new(2, Quantization::None, 10, None, None);
        assert!(!s.put(10, &[0.0, 0.0]));
    }
}

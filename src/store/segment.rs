use memmap2::Mmap;

use crate::error::{Error, Result};

/// read-only vector records backed by a mapped snapshot file
pub struct Segment {
    mmap: Mmap,
    offset: usize,
    stride_words: usize,
    count: usize,
}

impl Segment {
    pub fn new(mmap: Mmap, offset: usize, stride_words: usize, count: usize) -> Result<Self> {
        let len = count
            .checked_mul(stride_words * 4)
            .and_then(|n| n.checked_add(offset))
            .ok_or_else(|| Error::Corrupt("vector section overflows".into()))?;
        if len > mmap.len() {
            return Err(Error::Corrupt("vector section runs past end of file".into()));
        }
        bytemuck::try_cast_slice::<u8, u32>(&mmap[offset..len])
            .map_err(|e| Error::Corrupt(format!("vector section misaligned: {e}")))?;

        #[cfg(unix)]
        let _ = mmap.advise(memmap2::Advice::Random);

        Ok(Segment { mmap, offset, stride_words, count })
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn words(&self, id: usize) -> Option<&[u32]> {
        if id >= self.count {
            return None;
        }
        let start = self.offset + id * self.stride_words * 4;
        let bytes = &self.mmap[start..start + self.stride_words * 4];
        Some(bytemuck::cast_slice(bytes))
    }

    pub fn mapped_bytes(&self) -> usize {
        self.mmap.len()
    }
}

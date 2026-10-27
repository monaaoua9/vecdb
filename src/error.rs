use std::io;

use thiserror::Error;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("dimension mismatch: index has {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("invalid vector: {0}")]
    InvalidVector(&'static str),

    #[error("id {0} already exists")]
    DuplicateId(u64),

    #[error("id {0} not found")]
    NotFound(u64),

    #[error("index is full ({0} elements)")]
    CapacityExceeded(usize),

    #[error("invalid config: {0}")]
    InvalidConfig(String),

    #[error("snapshot is corrupt: {0}")]
    Corrupt(String),

    #[error("io: {0}")]
    Io(#[from] io::Error),

    #[error("snapshot encoding: {0}")]
    Codec(#[from] postcard::Error),
}

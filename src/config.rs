use serde::{Deserialize, Serialize};

use crate::distance::Metric;
use crate::error::{Error, Result};
use crate::quant::Quantization;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub dim: usize,
    pub metric: Metric,
    pub quantization: Quantization,
    /// keep full-precision vectors next to the quantized ones and re-score the
    /// final candidates with them (memory-mapped after a reload, so cold pages stay on disk)
    pub rerank: bool,
    /// target out-degree on upper layers; layer 0 keeps up to 2*m
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    /// hard cap, node tables are sized from it up front
    pub max_elements: usize,
    /// seeds the level generator, same seed + same insert order = same graph
    pub seed: u64,
    /// filters matching fewer ids than this are answered by an exact scan
    pub brute_force_below: usize,
}

impl Config {
    pub fn new(dim: usize, metric: Metric) -> Self {
        Config {
            dim,
            metric,
            quantization: Quantization::None,
            rerank: false,
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            max_elements: 1_000_000,
            seed: 0x5eed_cafe,
            brute_force_below: 2048,
        }
    }

    pub fn validate(&self) -> Result<()> {
        let bad = |msg: &str| Err(Error::InvalidConfig(msg.to_owned()));
        if self.dim == 0 || self.dim > 65_536 {
            return bad("dim must be in 1..=65536");
        }
        if self.rerank && self.quantization == Quantization::None {
            return bad("rerank needs a quantized index");
        }
        if let Quantization::Pq { m } = self.quantization
            && (m == 0 || !self.dim.is_multiple_of(m as usize))
        {
            return bad("pq m must divide dim");
        }
        if self.m < 2 || self.m > 256 {
            return bad("m must be in 2..=256");
        }
        if self.ef_construction < self.m {
            return bad("ef_construction must be >= m");
        }
        if self.ef_search == 0 {
            return bad("ef_search must be > 0");
        }
        if self.max_elements == 0 || self.max_elements >= u32::MAX as usize {
            return bad("max_elements must be in 1..u32::MAX");
        }
        Ok(())
    }
}

# Benchmarks

## Read this first

The numbers come from one weak laptop (i5-8365U, 4 cores / 8 threads, 8 GB, Windows 11, AVX2
only). It usually has several terminals open and the C: drive is almost full, so timings move
by 2-5x between runs. Every figure below is a sanity check, not a leaderboard. Rules used:

- compare old vs new interleaved in the same session, never from different days;
- report best-of or medians over several runs, never one run;
- don't believe a gap under about 25% between two builds of vecdb (under about 10% against
  hnswlib on the same run, where both see the same load);
- check free disk on C: before building or benchmarking.

## What there is

| what | command | measures |
|---|---|---|
| criterion micro benchmarks | `cargo bench --bench distance` / `--bench index` | distance kernels, insert, search per quantization (20k x 128) |
| recall / latency sweep | `cargo run --release --example recall -- [n] [dim] [threads] [iid\|latent]` | recall@10 against an exact scan, p50, QPS per `ef` |
| SIFT1M sweep | `cargo run --release --example sift -- <dir> [threads]` | the same on the real SIFT1M data |
| hnswlib reference | `benches/compare/hnswlib_sift.cpp` (build line in its header) | same parameters, same protocol, for a side-by-side table |

SIFT1M protocol: 1M x 128, L2, `M=16`, `ef_construction=200`, k=10, ground truth from the
dataset, one query thread, build on 8 threads. Build, `save`, then run again in a fresh
process that loads the snapshot (`SIFT_SNAPSHOT=<file>`), best of 5 passes over the 10 000
queries. Reloading matters: an index queried right after an in-process parallel build was
measurably slower than the same index after `save` + `open`.

Knobs for `sift`: `SIFT_QUANT=sq8|pq32`, `SIFT_RERANK=1`, `SIFT_PASSES=n`, `SIFT_EF=n`.
Needs about 0.5 GB for the data and 0.6 GB for the snapshot.

## Results (SIFT1M, reloaded, best of 5)

| ef | recall@10 | vecdb p50 | vecdb QPS | hnswlib p50 | hnswlib QPS |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.802 | 69 µs | 14.0 k | 61 µs | 15.3 k |
| 32 | 0.903 | 116 µs | 8.4 k | 102 µs | 9.4 k |
| 64 | 0.963 | 200 µs | 4.9 k | 183 µs | 5.3 k |
| 128 | 0.989 | 367 µs | 2.7 k | 338 µs | 2.8 k |

Recall is identical to hnswlib at every `ef`; hnswlib is roughly 8-12% faster per query on
this machine. `ef` 256 and 512 are left out because the hnswlib passes were visibly disturbed
by background load. Quantized variants (SQ8, PQ32, PQ32 + rerank) are in the README.

## Regression check after the feature work (2026-10-30)

Question: did upserts, payload locks, collections, hybrid search, replication and the rest
slow the search path? Synthetic data, 50 000 x 128 uniform random, L2, `M=16`,
`ef_construction=100`, k=10, `ef=64`, one query thread, 2000 queries, best of 7 passes per
run, old and new alternating. Old is commit `7536f7d` (before the feature work).

| build | runs | median QPS | range |
|---|---:|---:|---|
| old | 6 | 2472 | 2127-3080 |
| new | 8 | 2622 | 2194-4388 |

The ranges overlap and the median is not lower, so there is no measurable slowdown. An
earlier 3-run round showed the new build about 10% slower at `ef=64` and equal at `ef=256`;
the run-to-run spread of the old build alone (about 25%) was bigger than that gap. This is
synthetic data, not comparable to the SIFT1M table.

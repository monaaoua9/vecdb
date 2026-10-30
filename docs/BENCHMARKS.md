# vecdb: benchmarks

Measured 2026-10-29.

| | |
|---|---|
| CPU | Intel i5-8365U, 4 cores / 8 threads, AVX2, no AVX-512 |
| RAM / disk | 8 GB, SSD |
| OS | Windows 11 |
| Toolchain | Rust 1.99.1, `--release` |

## Search quality and latency

```
cargo run --release --example recall -- 50000 128 8 latent
```

50,000 vectors, dim 128, built on 8 threads, data with latent structure. Recall@10 is against exact search. Three back-to-back runs; latency and QPS are the median with min–max in brackets. Recall barely moves, so I give its range.

f32 vectors (24.5 MiB of vectors):

| ef | recall@10 | p50 (µs) | QPS |
|---:|---:|---:|---:|
| 16 | 0.9525 – 0.954 | 64 (63 – 70) | 14,856 (13,144 – 15,115) |
| 32 | 0.9895 – 0.990 | 125 (117 – 130) | 7,525 (7,380 – 8,027) |
| 64 | 0.9995 | 186 (180 – 244) | 4,994 (3,870 – 5,214) |
| 128 | 1.0000 | 375 (321 – 421) | 2,550 (2,298 – 2,986) |
| 256 | 1.0000 | 624 (584 – 736) | 1,518 (1,300 – 1,613) |

SQ8 (6.7 MiB of vectors):

| ef | recall@10 | p50 (µs) | QPS |
|---:|---:|---:|---:|
| 16 | 0.951 – 0.953 | 54 (50 – 62) | 18,142 (15,436 – 19,140) |
| 32 | 0.9915 – 0.992 | 92 (75 – 97) | 9,946 (9,456 – 13,127) |
| 64 | 0.9995 | 165 (135 – 191) | 5,488 (5,248 – 6,678) |
| 128 | 1.0000 | 279 (246 – 337) | 3,272 (2,852 – 3,784) |
| 256 | 1.0000 | 490 (453 – 637) | 1,882 (1,620 – 2,029) |

SQ8 uses 3.7× less memory for the vectors with about the same recall at every ef here, and has the lower median latency at every ef. This data set is small and well-behaved, so don't expect recall to stay this high on a hard one.

## SIMD kernels

```
cargo bench --bench distance -- --warm-up-time 1 --measurement-time 2
```

Criterion, median of the reported interval, ns per call (criterion already reports its own confidence interval, so this is one run):

| dim | op | scalar | AVX2+FMA |
|---:|---|---:|---:|
| 128 | dot | 59.6 | 24.1 |
| 128 | l2_sq | 66.0 | 26.7 |
| 128 | dot_u8 | 53.5 | 27.2 |
| 768 | dot | 209.5 | 71.3 |
| 768 | l2_sq | 253.9 | 83.2 |
| 768 | dot_u8 | 198.8 | 97.1 |
| 1536 | dot | 353.9 | 136.7 |
| 1536 | l2_sq | 396.1 | 142.2 |
| 1536 | dot_u8 | 375.0 | 189.7 |

The scalar baseline is already auto-vectorised to SSE2 by LLVM, so this is AVX2 against a decent baseline, roughly 2–3×.

## SIFT1M

The README has SIFT1M results (recall vs latency for f32, SQ8 and PQ32 + rerank) and a comparison harness against hnswlib (`benches/compare/hnswlib_sift.cpp`). I did not rerun those today: building the index takes minutes and needs the dataset. Treat the README numbers as an earlier run on the same machine.

## Limits of these numbers

- Single machine. Latency moves up to about 35% between runs (ef=64 f32 p50 went from 180 to 244 µs), so don't read differences under that as real.
- Query set drawn from the same distribution as the data.
- No comparison against another engine in this file.
- The AVX-512 and NEON kernels are not exercised here (this CPU has neither).

## Did the feature work slow search down? (2026-10-30)

Synthetic data, 50,000 x 128 uniform random, L2, `M=16`, `ef_construction=100`, k=10, `ef=64`, one query thread, built on one thread, 2000 queries, best of 7 passes per run. Old (commit `7536f7d`, before upserts, payload locks, collections, hybrid search and replication) and new alternated in one session.

| build | runs | median QPS | min – max |
|---|---:|---:|---|
| old | 6 | 2,472 | 2,127 – 3,080 |
| new | 8 | 2,622 | 2,194 – 4,388 |

The ranges overlap and the new median is not lower, so no slowdown shows up. An earlier 3-run round had the new build about 10% slower at `ef=64` and equal at `ef=256`; the old build's own spread (about 25%) was larger than that gap. The laptop is busy (several terminals, disk almost full), so don't read anything into a difference under about 25%. Not comparable to the tables above: different data and size.

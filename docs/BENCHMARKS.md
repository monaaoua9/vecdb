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

50,000 vectors, dim 128, built on 8 threads, data with latent structure. Recall@10 is against exact search. Latency is p50 per query.

f32 vectors (build 5.7 s, 24.5 MiB of vectors):

| ef | recall@10 | p50 | QPS |
|---:|---:|---:|---:|
| 16 | 0.9535 | 55 µs | 17,664 |
| 32 | 0.9890 | 93 µs | 10,254 |
| 64 | 0.9995 | 167 µs | 5,650 |
| 128 | 1.0000 | 272 µs | 3,602 |
| 256 | 1.0000 | 512 µs | 1,925 |

SQ8 (build 4.7 s, 6.7 MiB of vectors):

| ef | recall@10 | p50 | QPS |
|---:|---:|---:|---:|
| 16 | 0.9530 | 39 µs | 24,448 |
| 32 | 0.9920 | 64 µs | 15,401 |
| 64 | 0.9995 | 118 µs | 8,325 |
| 128 | 1.0000 | 219 µs | 4,388 |
| 256 | 1.0000 | 399 µs | 2,480 |

SQ8 uses 3.7× less memory for the vectors with the same recall at every ef here, and is faster. This data set is small and well-behaved, so don't expect the recall to stay this high on a hard one.

## SIMD kernels

```
cargo bench --bench distance -- --warm-up-time 1 --measurement-time 2
```

Criterion, median of the reported interval, ns per call:

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

- Single machine, one run per ef, query set drawn from the same distribution as the data.
- No comparison against another engine in this file.
- The AVX-512 and NEON kernels are not exercised here (this CPU has neither).

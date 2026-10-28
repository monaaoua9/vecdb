# vecdb

An embeddable vector database in Rust: an HNSW graph index, hand-written SIMD
distance kernels (AVX-512, AVX2+FMA, NEON), 8-bit scalar quantization,
memory-mapped snapshots, lock-based concurrent inserts, metadata filtering and
a gRPC front-end. Use it as a library or run it as a server.

```rust
use vecdb::{Config, Filter, Index, Metric, Payload, SearchParams, Value};

let mut cfg = Config::new(768, Metric::Cosine);
cfg.max_elements = 1_000_000;
let index = Index::new(cfg)?;

let mut meta = Payload::new();
meta.insert("lang".into(), Value::Str("ar".into()));
index.insert(42, &embedding, meta)?;

let hits = index.search_with(
    &query,
    &SearchParams::new(10).ef(64).filter(Filter::eq("lang", "ar")),
)?;
index.save("snapshot-000000000001.vdb")?;
let index = Index::open("snapshot-000000000001.vdb")?; // vectors are mmap'd
```

## What is in the box

| | |
|---|---|
| **Index** | HNSW with the paper's diversity heuristic for neighbour selection, seeded level generation, tombstone deletes with start-up compaction |
| **Concurrency** | inserts and searches run in parallel; per-node per-layer `RwLock`s, write-once node/vector slots, no global lock on the query path |
| **SIMD** | `dot`, `l2_sq` and a `u8×f32` dot for AVX-512F, AVX2+FMA, NEON, plus a scalar reference; chosen at run time |
| **Quantization** | SQ8, per-vector min/scale, asymmetric distance (query stays `f32`, vectors are never decoded) — 72.7% smaller at 128 dims, 74.6% at 768 |
| **Storage** | single-file snapshot, vectors served from `mmap`, CRC-checked header and metadata, atomic write-then-rename |
| **Filtering** | `Eq`, `AnyOf`, `Range`, `And`, `Or`, `Not` over typed payloads; roaring-bitmap inverted index; planner picks exact pre-filter scan or filtered graph walk, or force either with `FilterStrategy` |
| **Durability** | `DurableIndex`: snapshot generations + write-ahead log (CRC per record, torn tails truncated, idempotent replay), optional fsync with group commit (concurrent writers share one fsync) |
| **API** | tonic gRPC: insert, streaming bulk insert, search, get, delete, stats, snapshot, health |

Design write-up with diagrams: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Build and run

Needs Rust 1.89+ and `protoc` (only for the server feature).

```sh
cargo build --release                       # library + server
cargo build --release --no-default-features # library only, no tonic/tokio

cargo run --release --bin vecdb-server -- \
    --dim 768 --metric cosine --quantization sq8 \
    --data-dir ./data --snapshot-every-secs 300 --wal-fsync
```

On start-up the server loads the newest `snapshot-*.vdb` in `--data-dir`
(dimensions and metric come from the file) and replays the write-ahead log on
top of it, so acknowledged writes survive a crash. Ctrl-C writes a final
snapshot. Without `--data-dir` it is memory-only. `RUST_LOG=debug` for more output; `VECDB_SIMD=scalar` forces a kernel.

```sh
cargo run --example client     # small gRPC client against a running server
```

## Tests and benchmarks

```sh
cargo test                     # 49 tests: kernels vs scalar, recall, concurrency,
                               # filtering, persistence/corruption, WAL recovery, gRPC end-to-end
cargo bench --bench distance   # criterion, every kernel the CPU supports
cargo bench --bench index      # search latency by ef, insert cost
cargo run --release --example recall -- 50000 128 8 latent
```

Numbers below are from a laptop (i5-8365U, 4 cores / 8 threads, AVX2, no AVX-512),
Windows, `cargo build --release`. Treat them as a sanity check, not a leaderboard.

**Recall and latency** — 50 000 vectors, 128 dims, cosine, k=10, single query thread,
recall measured against an exact scan. Data has latent structure (12 hidden factors),
which is what embedding-model output looks like:

| ef | recall@10 (f32) | p50 | QPS | recall@10 (SQ8) | p50 |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.954 | 80 µs | 13.1 k | 0.951 | 47 µs |
| 32 | 0.990 | 96 µs | 10.1 k | 0.992 | 80 µs |
| 64 | 0.9995 | 142 µs | 6.9 k | 0.9995 | 187 µs |
| 128 | 1.000 | 253 µs | 3.9 k | 1.000 | 232 µs |

SQ8 keeps recall and cuts vector memory from 24.5 MiB to 6.7 MiB. Parallel build:
about 8.7 k inserts/s on 8 threads (`ef_construction = 200`, `M = 16`).

**SIFT1M versus hnswlib** (1 000 000 x 128, L2, `M=16`, `ef_construction=200`, k=10,
ground truth from the dataset, single query thread, build on 8 threads, same laptop,
hnswlib built with `-O3 -march=native`; run one after the other, not concurrently):

| ef | recall@10 vecdb | recall@10 hnswlib | p50 vecdb | p50 hnswlib | QPS vecdb | QPS hnswlib |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.802 | 0.800 | 186 µs | 110 µs | 5.1 k | 8.6 k |
| 32 | 0.903 | 0.903 | 221 µs | 132 µs | 4.2 k | 7.2 k |
| 64 | 0.963 | 0.963 | 344 µs | 180 µs | 2.7 k | 5.3 k |
| 128 | 0.989 | 0.989 | 590 µs | 307 µs | 1.6 k | 3.3 k |
| 256 | 0.997 | 0.997 | 974 µs | 569 µs | 1.0 k | 1.8 k |
| 512 | 0.999 | 0.999 | 1.6 ms | 1.0 ms | 610 | 986 |

Build time: vecdb 239 s (4.2 k inserts/s), hnswlib 219 s (4.6 k inserts/s).

Graph quality is the same: recall matches hnswlib at every `ef`.

How to read the speed columns. In the table above each program built the index and
then queried it in the same process, one pass over the queries. That is a fair
like-for-like setup, but this laptop makes it noisy, and I found a second effect that
matters more: **a vecdb index queried right after an in-process parallel build is
noticeably slower than the same index after `save` + `open`** (ef=64: 250-340 µs
right after the build, 190-220 µs reloaded from the snapshot, several runs each). My
guess is allocation locality (reloading allocates nodes in id order), but I have not
verified that. I could not re-run hnswlib with the best-of-N / reloaded protocol, so I
make no claim about the size of the speed gap beyond "hnswlib was faster in this run".

vecdb reloaded from a snapshot, fastest of 5 passes, same SIFT1M protocol:

| ef | recall@10 (f32) | p50 | QPS | recall@10 (SQ8) | p50 | QPS |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.801 | 73 µs | 12.9 k | 0.797 | 131 µs | 5.5 k |
| 32 | 0.903 | 119 µs | 8.1 k | 0.899 | 210 µs | 4.4 k |
| 64 | 0.963 | 210 µs | 4.7 k | 0.958 | 296 µs | 3.3 k |
| 128 | 0.989 | 389 µs | 2.5 k | 0.982 | 358 µs | 2.5 k |
| 256 | 0.997 | 750 µs | 1.3 k | 0.990 | 766 µs | 1.2 k |
| 512 | 0.999 | 1.4 ms | 685 | 0.992 | 1.4 ms | 685 |

SQ8 shrinks the snapshot from 585 MB to 212 MB (about 2.7x), costs roughly half a point
of recall at ef=64, and on this CPU is *slower* per query at low `ef` (the AVX2
`u8 -> f32` widening plus the extra per-query terms cost more than the saved memory
traffic) and caps out near 0.992 recall. Use it to fit more vectors in RAM, not for speed.

Product quantization (`Quantization::Pq { m }`, `Index::new_pq(cfg, sample, threads)`,
codebook trained on a sample, 256 centroids per subspace, lookup-table distance). SIFT1M with
m=32 (32 bytes per vector, 4x smaller than SQ8's 140): snapshot 102 MB including the graph,
codebook trained in 6 s, recall@10 0.63 / 0.70 / 0.707 at ef 16 / 64 / 512. The ceiling
is the quantization error. `Config::rerank` / `--rerank` keeps full-precision vectors in a
second, memory-mapped section and re-scores the beam with them (numbers below).
`DurableIndex::train_pq` stores the codebook in the data directory; the server takes
`--quantization pq` for a directory that has one.

PQ32 + rerank, SIFT1M, reloaded from the snapshot, fastest of 5 passes:

| ef | recall@10 | p50 | QPS |
|---:|---:|---:|---:|
| 16 | 0.720 | 101 µs | 9.8 k |
| 32 | 0.867 | 142 µs | 7.0 k |
| 64 | 0.949 | 213 µs | 4.7 k |
| 128 | 0.985 | 372 µs | 2.7 k |
| 256 | 0.996 | 642 µs | 1.6 k |
| 512 | 0.999 | 1.2 ms | 847 |

Rerank removes the 0.71 ceiling, and at ef=64 the latency matches plain f32 (213 vs 210 µs)
while the traversal only touches 32-byte codes. Recall is a bit below f32 at the same `ef`
(0.949 vs 0.963) because the graph is built and walked on quantised distances. The snapshot
is 614 MB since it holds both the codes and the f32 section; what shrinks is the memory the
search keeps hot (codes plus graph), not the file. The build is slower (331 s vs 190 s for SQ8).

Where the time goes (f32, ef=64, about 1 100 distance evaluations per query):
distance arithmetic is about 11% of the cycles; the rest is mostly fetching neighbour
lists and vectors. Prefetching vectors matters (310 µs without, about 225 µs with).
Things that made no measurable difference: a smaller visited-set, skipping a
per-candidate node lookup when nothing is deleted, and a flat layer-0 neighbour array
(tried and reverted; the apparent gain in one A/B run came from comparing a
freshly-built index against a reloaded one), and dropping the per-record "written"
check from the search loop (it is one extra load, but in-process p50 at ef=64 only went
from 250-320 to 243 µs, reloaded stayed at 200). The in-process vs reloaded gap is
therefore not that check; what is left is heap layout of the neighbour lists, which
a reload rewrites in id order. Not verified with a hardware-counter profiler, since
none is available on this machine.

Reproduce: `cargo run --release --example sift -- <dir>` and
`benches/compare/hnswlib_sift.cpp` (build line is in its header).

**The hard case.** With i.i.d. Gaussian noise in all 128 coordinates there is no
structure for a graph to exploit and every point is nearly equidistant from every
other. On 20 000 such vectors recall@10 is 0.48 at ef=64, 0.70 at ef=128 and 0.88
at ef=256. Anyone benchmarking an ANN index on random data will see this; it says
more about the data than the index. Run it yourself:
`cargo run --release --example recall -- 20000 128 8`.

**SIMD kernels** (criterion, ns per call; scalar here is already auto-vectorised
to SSE2 by LLVM, so this is AVX2 against a decent baseline):

| dim | `dot` scalar → AVX2 | `l2_sq` scalar → AVX2 | `dot_u8` scalar → AVX2 |
|---:|---:|---:|---:|
| 128 | 50 → 32 | 61 → 28 | 67 → 35 |
| 768 | 216 → 120 | 242 → 126 | 311 → 134 |
| 1536 | 432 → 238 | 458 → 213 | 419 → 299 |

## Layout

```
src/
  distance/   kernels (x86.rs, neon.rs, scalar.rs), metric + SQ8 distance dispatch
  quant.rs    SQ8 encode/decode, record layout
  pq.rs       PQ codebook training, encode/decode, lookup tables
  store/      vector slabs + mmap segment
  hnsw/       graph.rs (node table), insert.rs, search.rs, scratch.rs (per-thread buffers)
  filter/     payload types, filter tree, roaring inverted index
  persist.rs  snapshot save/open
  durable.rs  snapshot + WAL recovery wrapper (DurableIndex)
  server/     tonic service, proto conversion, snapshot rotation
proto/        vectordb.proto
tests/        integration tests (+ common/ data generators)
benches/      criterion benchmarks
examples/     recall sweep, gRPC client
```

## Semantics worth knowing

- Reported `distance`: squared L2, `1 − cos`, or negative inner product. Lower is closer.
- Cosine vectors are normalised on insert; `get` returns the normalised vector.
- With SQ8 or PQ, `get` returns the dequantised vector.
- `Eq` is strict on type (`Int(1)` ≠ `Float(1.0)`); float equality is not indexed.
- Searches never block on inserts; a search racing an insert may or may not see it.
- `save` and `checkpoint` pause inserts and deletes while they run. Searches continue.
- WAL default is flush-to-OS (survives a process crash); `--wal-fsync` also survives power loss.
- Deleted vectors are tombstoned and keep routing queries. `Index::compacted` rebuilds without them; the server does it at start-up when more than `--compact-above` (default 0.3) of nodes are deleted. Payloads can't be updated in place.

## Not done (yet)

No online compaction, single node only. The PQ graph is built on quantised distances (only the final ranking uses full vectors). AVX-512 and NEON kernels are checked by
compilation and review against the AVX2 versions, not by execution on this
machine; the kernel-vs-scalar test runs them wherever the hardware exists (CI
includes an aarch64 job). Details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#8-known-limits-and-what-comes-next).

## License

MIT.

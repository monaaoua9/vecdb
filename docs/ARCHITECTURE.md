# Architecture

This document walks through what happens to a vector from the moment it hits
the gRPC port to the moment it is returned as a neighbour, and explains the
decisions behind each layer. File references are relative to the repo root.

## 1. Big picture

```
                         ┌────────────────────────────┐
   gRPC client ────────▶ │ server/   (tonic, tokio)   │   async, no CPU-heavy work here
                         │  validate · convert · RPC  │
                         └─────────────┬──────────────┘
                                       │ spawn_blocking
                         ┌─────────────▼──────────────┐
                         │ durable::DurableIndex      │   WAL + checkpoints (optional)
                         └─────────────┬──────────────┘
                         ┌─────────────▼──────────────┐
                         │ hnsw::Index                │   thread-safe, &self everywhere
                         │  insert · search · delete  │
                         └───┬──────────┬──────────┬──┘
                             │          │          │
              ┌──────────────▼───┐ ┌────▼─────┐ ┌──▼───────────────┐
              │ graph (nodes)    │ │ filter/  │ │ store/           │
              │ per-node, per-   │ │ payloads │ │ vector records   │
              │ layer RwLock'd   │ │ + roaring│ │ slabs + mmap'd   │
              │ adjacency lists  │ │ inverted │ │ base segment     │
              └──────────────────┘ │ index    │ └──┬───────────────┘
                                   └──────────┘    │ VectorRef
                                             ┌─────▼──────────────┐
                                             │ distance/          │
                                             │ AVX-512 · AVX2 ·   │
                                             │ NEON · scalar      │
                                             └────────────────────┘
        persist.rs: snapshot (header | vectors | graph+payload blob), atomic rename
```

Everything below the server layer is a plain library (`cargo build --no-default-features`
drops tonic/tokio entirely), so the engine can be embedded directly.

## 2. Life of a query

`Index::search_with` (src/hnsw/search.rs)

```
query ──▶ sanitize (dim, NaN/Inf, normalise if cosine) ──▶ PreparedQuery
                                                              │
         entry point (top layer L)                            ▼
              │                                   ┌──────────────────────┐
  layer L     ●──greedy──▶ ●                      │ filter?              │
  layer L-1        ●──greedy──▶ ●                 │  none  → ANN         │
   ...                           ●                │  selective → exact   │
  layer 1                          ●──greedy──▶ ● │            scan over │
              ef = 1 per layer, one hop at a time │            bitmap    │
                                          │       │  broad → ANN with    │
  layer 0     beam search, width ef ◀─────┘       │          gate        │
              ┌───────────────────────────┐       └──────────────────────┘
              │ cands  : min-heap by dist │
              │ results: max-heap, size ef│  visit neighbours of the closest
              │ visited: epoch-stamped    │  candidate, stop when the closest
              └───────────────────────────┘  candidate is worse than the worst result
                        │
                        ▼
              top-k → map internal id → external id → Hit { id, distance }
```

1. **Upper layers (L … 1)** are sparse "express lanes". The search keeps one
   current best node and keeps hopping to a closer neighbour until none is
   closer, then drops a layer. This costs O(log N) hops in expectation, and is
   where the sub-linear behaviour comes from.
2. **Layer 0** holds every vector. A beam search of width `ef` explores the
   neighbourhood: pop the closest unexplored candidate, evaluate its unvisited
   neighbours, keep the best `ef` seen. `ef` is the recall/latency knob.
3. **Distance evaluation** goes through `Space::dist`, which dispatches on how
   the stored vector is encoded (`f32` or SQ8) and on the metric, and ends in
   one of the SIMD kernels. Before evaluating a node's neighbours the search
   issues prefetches for their vector records, so the cache misses overlap
   with the arithmetic.
4. **Visited set** is a per-thread `Vec<u32>` of epoch stamps (src/hnsw/scratch.rs).
   "Clearing" it is an increment, not a memset, so a query allocates nothing
   on the hot path after warm-up.

### Filtered queries

Metadata filters are `Filter` trees (`Eq`, `AnyOf`, `Range`, `And`, `Or`,
`Not`). The planner asks the payload index (roaring bitmaps keyed by
`(field, value)`) what it can say about the filter:

| Planner result | Strategy (`FilterStrategy::Auto`) |
|---|---|
| bitmap with ≤ `brute_force_below` ids | **pre-filter**: exact distance scan over just those ids. 100% recall, cost proportional to selectivity. |
| larger bitmap | **filtered graph walk**: the walk still routes through non-matching nodes, but only matching, live nodes may enter the result set. If it ends with fewer than `k`, `ef` is widened ×4 until it has `k` or has covered the index. |
| not indexable (`Range`, `Not`, float equality) | same walk, with the predicate evaluated on each visited node's payload |

`And` over a mix of indexable and non-indexable children still uses the bitmap
as a candidate bound and re-checks the rest per node. `PreFilter` and
`PostFilter` can be forced from the request for comparison; `PostFilter` runs a
plain ANN search for `4k` and keeps widening until `k` survive.

## 3. Life of an insert

`Index::insert` (src/hnsw/insert.rs)

1. **Validate and normalise** once, at the boundary. Cosine vectors are stored
   unit-length so the hot path is a dot product.
2. **Reserve an internal id** under the id-map lock (duplicate check and
   capacity check are atomic with the reservation).
3. **Draw a level** `⌊-ln(U)·mL⌋` with `mL = 1/ln(M)`, from a seeded
   splitmix64 stream (reproducible builds with a fixed seed and insert order).
4. **Write the vector record**, then publish the node. Slots are write-once, so
   a reader can never see a half-built node.
5. **Link into the graph**, top to bottom:
   - descend greedily from the entry point to the node's level,
   - on each layer ≤ level run a beam search of width `ef_construction`,
   - choose `M` neighbours with the **diversity heuristic** (take a candidate
     only if it is closer to the new node than to every neighbour already
     taken — this is what keeps clusters connected to each other),
   - add reverse edges; if a neighbour's list is full (`M`, or `2M` on layer 0)
     re-run the same heuristic over its old edges plus the new one.
6. If the node is taller than the current entry point, it becomes the new
   entry point.
7. Only then is the node added to the payload index, so filtered queries never
   see a node whose edges are half-written.

## 4. Concurrency model

The goal: inserts and searches run at the same time, searches never block
behind an insert, and nothing needs a global lock on the hot path.

| Structure | Mechanism | Why it is safe |
|---|---|---|
| node table (`AppendVec`) | chunked `OnceLock` slots, never moved | `get` hands out `&T` that stays valid while others append |
| vector records (`VectorStore`) | slabs allocated on first touch; per-record `EMPTY→WRITING→READY` state, `Release` store / `Acquire` load | one writer per record (CAS), readers only see `READY` data |
| adjacency lists | `RwLock<Vec<u32>>` per node *per layer* | readers copy the list out and release the lock before computing distances; writers hold exactly one lock at a time, so there is no lock ordering to get wrong |
| entry point | `RwLock<Option<(id, level)>>` | read briefly per query; a node taller than the current top holds it exclusively for its whole insert, so two "tallest" nodes can't race |
| id map | `RwLock<HashMap>` | held only for the lookup/insert itself |
| snapshots | `RwLock<()>` "gate": insert/delete take it shared, `save` takes it exclusive | `save` sees a state with no half-linked nodes; searches never touch the gate |

Deletes are tombstones: the node stays in the graph (it still routes queries)
but is skipped for results and removed from the payload index. Reusing an
external id after delete creates a fresh node.

Things that can happen under concurrency and are accepted:

- A search racing an insert may or may not see the new node.
- Two inserts that discover each other can end up with a slightly different
  edge set than a serial build. The test suite measures recall after an
  8-thread build against ground truth (tests/concurrency.rs) and the
  `recall` example shows the same recall for 1 thread and 8 threads.

## 5. SIMD distance kernels

`src/distance/` holds three functions per instruction set, picked once at
start-up (`Kernels`, a table of plain `fn` pointers):

| kernel | computes | used for |
|---|---|---|
| `dot(a, b)` | Σ aᵢbᵢ | inner product, cosine (unit vectors) |
| `l2_sq(a, b)` | Σ (aᵢ−bᵢ)² | euclidean |
| `dot_u8(q, codes)` | Σ qᵢ·cᵢ with `u8 → f32` widening | SQ8 vectors |

| ISA | file | notes |
|---|---|---|
| AVX-512F | x86.rs | 2×16 lanes per iteration, masked tail (no scalar remainder loop) |
| AVX2 + FMA | x86.rs | 4×8 lanes per iteration for `dot`/`l2`, `vpmovzxbd` widening for u8 |
| NEON | neon.rs | 4×4 lanes, `vfmaq_f32`, `vaddvq_f32` reduction |
| scalar | scalar.rs | 8 independent accumulators so LLVM can vectorise it; also the reference in tests |

Selection happens with `is_x86_feature_detected!` at run time, so one binary
runs everywhere. `VECDB_SIMD=scalar|avx2|avx512` overrides it for debugging.
The `unsafe` surface is small: each intrinsic function is reachable only
through a safe wrapper that is installed in the table after the CPU check, and
the wrapper asserts equal lengths first. tests/simd.rs compares every kernel
available on the host to the scalar one for dimensions 0…130 plus 255, 256,
257, 768, 1023 and 1536 (every tail length).

### SQ8 and the asymmetric distance trick

Each vector is quantised on its own: `x̂ᵢ = min + scale·cᵢ`, `cᵢ ∈ 0..=255`,
with `(min, scale, ‖x̂‖²)` stored next to the codes (12 bytes). Queries stay
full precision, and the distance is computed without ever decoding the vector:

```
q·x̂   = min·Σqᵢ + scale·(q·c)          Σqᵢ computed once per query
‖q−x̂‖² = ‖q‖² + ‖x̂‖² − 2·q·x̂           ‖x̂‖² read from the record
```

so the inner loop is the single `dot_u8` kernel. Per-vector parameters mean
there is no training pass and streaming inserts work; the cost is a 12-byte
header, so the saving is `1 − (dim + 12)/(4·dim)`: 72.7% at 128 dims, 74.6% at
768, 74.8% at 1536.

## 6. Storage, memory mapping and durability

```
snapshot file
┌──────────────┬──────────────────────────────┬──────────────────────────────┐
│ header 4 KiB │ vector records (fixed stride)│ graph + payloads (postcard)  │
│ magic,version│ mmap'd read-only on open     │ CRC32 in header              │
│ offsets, CRC │ 4096-aligned, zero-copy      │ decoded into memory on open  │
└──────────────┴──────────────────────────────┴──────────────────────────────┘
```

- `Index::open` maps the file, verifies the header and metadata checksums,
  rebuilds the adjacency lists and the payload index, and serves vectors
  straight from the mapping. The OS pages vectors in on demand, so start-up
  cost is proportional to the graph, not to the vector data.
- Records have one layout in memory and on disk, so the mapped base segment
  and the in-memory slabs (for vectors inserted after open) are read through
  the same `VectorStore::words` path.
- `Index::save` writes `*.tmp`, `fsync`s, then renames. A crash leaves either
  the previous snapshot or the new one.
- Snapshots are never overwritten: `snapshot-<generation>.vdb`, highest
  generation wins, older ones are pruned (this also sidesteps Windows refusing
  to replace a file that is still mapped).

### Write-ahead log (`src/durable.rs`)

```
data-dir/  snapshot-00000000000N.vdb   state as of generation N
           wal-00000000000N.log        mutations applied after it
record  =  [len u32][crc32 u32][postcard(Insert{id,vector,payload} | Delete(id))]
```

- **Write path:** apply to the index, then append and flush (or fsync) before
  acknowledging. Failed operations (duplicate id, wrong dimension) are never
  logged. Bulk inserts log first, then apply, in one flush.
- **Recovery:** open the newest snapshot, replay every `wal-<gen>` with
  gen ≥ snapshot gen in order. A record with a bad length or CRC ends the
  replay and the file is truncated there (torn write from a crash). Replay is
  idempotent, so re-running it or replaying records already contained in the
  snapshot is harmless.
- **Checkpoint:** takes mutations exclusive, creates `wal-<N+1>`, writes
  `snapshot-<N+1>` (temp file + rename), switches the log, then deletes
  `wal-<=N` and old snapshots. A crash at any point leaves a state recovery
  can read: either snapshot N plus both logs, or snapshot N+1.
- **Group commit:** writers queue their records under a short lock; one of
  them becomes the leader, writes (and fsyncs) everything queued so far with
  the lock released, then wakes the rest. In the durability test, 8 threads
  doing 480 fsync'd writes produced 120 log batches. A failed write poisons
  the log: every later append returns the error rather than acknowledging
  something that is not on disk.
- **Flush vs fsync:** by default records are handed to the OS before the
  write is acknowledged (survives a process crash). `--wal-fsync` adds an
  fsync per write (survives power loss).

## 7. gRPC surface

`proto/vectordb.proto`, package `vecdb.v1`.

| RPC | Notes |
|---|---|
| `Insert` | one vector + payload |
| `InsertStream` | client-streaming bulk load, applied in batches of 512 with a parallel `insert_bulk` |
| `Search` | `k`, optional `ef`, optional `Filter`, `FilterStrategy`, `with_payload` |
| `Get`, `Delete` | by external id |
| `Stats` | counts, graph shape, active SIMD kernel, heap vs mapped bytes |
| `Snapshot` | cuts a checkpoint (new snapshot generation, retires covered WAL) |

CPU-bound work runs on tokio's blocking pool, so the async workers stay free
for I/O. Errors map onto gRPC codes (`InvalidArgument`, `AlreadyExists`,
`NotFound`, `ResourceExhausted`, …). Filter nesting is capped at 16 levels and
`k` at 10 000. The standard `grpc.health.v1` service is registered.

## 8. Known limits and what comes next

Stated plainly so nobody has to discover them:

- **Compaction is offline.** Deletes leave tombstones until `Index::compacted`
  rebuilds the graph from live vectors (chunked, parallel). `DurableIndex::compact`
  needs `&mut self`, so the server runs it at start-up when the tombstone ratio
  exceeds `--compact-above`, then checkpoints. Compacting a live, shared index
  would need an index swap behind an `ArcSwap`-style handle; not done.
- **Payloads are immutable** once inserted (delete and re-insert to change).
- **`max_elements` is fixed at creation** (it sizes the node table); `open_with`
  can raise it when reloading a snapshot directly.
- **PQ has no rerank stage.** Vectors are only stored as codes, so recall is
  capped by quantisation error (about 0.71 for 32 bytes on SIFT1M). Keeping
  full-precision vectors on disk for a final rerank would lift the cap; that
  is not built. PQ indexes also can't go through `DurableIndex` or the gRPC
  server yet, because the codebook has to exist before the first insert.
- **AVX-512 and NEON kernels are verified by compilation and code review
  against the AVX2 versions, not by execution**, because the development
  machine supports neither. The test that checks every available kernel
  against scalar will exercise them automatically on hardware that has them,
  and CI runs it on `aarch64`.
- Single node. No sharding or replication.

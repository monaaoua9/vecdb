# Design notes

Short version of why vecdb looks the way it does. `ARCHITECTURE.md` has the long walk-through.

## What it is

An embedded vector database: a library first, a gRPC server second. One process owns the
data; the server is a thin layer over the same `DurableIndex` you can use directly
(`--no-default-features` drops tonic and tokio).

## Decisions

**HNSW, one graph, writes under fine-grained locks.** Every node has its own `RwLock` per
layer, so inserts from many threads don't serialise on a global lock and searches never block
on writers for more than one adjacency list. Cost: the graph is not bit-for-bit reproducible
when built with more than one thread.

**Deletes are tombstones.** A deleted or upserted node keeps routing queries (removing it
would cut the graph) and keeps its slot. `compact_online` rebuilds in the background,
replays the writes that came in meanwhile, then swaps behind a gate. An epoch counter makes
a stale rebuild throw its result away if `replace_index` happened in between.

**Snapshots plus a log, not a page store.** State on disk is the newest snapshot generation
and the WAL from that generation on. Snapshots are written to a temp file and renamed, and
the vector section is memory-mapped on open, so reload is fast and cold vectors stay out of
RAM. The WAL has CRC per record and group commit; a torn tail is cut off at recovery.

**SIMD behind one dispatch.** AVX-512, AVX2, NEON and a scalar fallback, chosen once at
start. Every kernel is tested against scalar. AVX-512 and NEON are not run on the dev
machine (see limits in `ARCHITECTURE.md`).

**Quantization is a storage choice, not a different index.** f32, SQ8, PQ and binary all go
through the same graph code via `VectorRef`. SQ8 is for memory (it is slower per query on
AVX2). PQ and binary need `rerank` to be accurate: the beam is re-scored with the f32 copy.

**Filters are planned, not post-applied.** Payload fields get a roaring-bitmap inverted
index; the planner picks between filtering inside the graph walk and a brute-force scan over
the matching set, depending on how selective the filter is.

**Replication is deliberately small.** One leader streams its latest snapshot and then tails
its WAL; replicas are in-memory, read-only, asynchronous. The leader keeps the previous log
through a checkpoint so a caught-up replica doesn't re-download. No failover, no sharding.

## Things I would not change

- Errors bubble up as one `Error` enum; the server maps them to gRPC codes at the edge.
- `Index` is `&self` everywhere and `Send + Sync`, so callers don't need their own lock.
- The server never does CPU-heavy work on the async runtime (`spawn_blocking`).

## Things I'd look at next

- Persist the BM25 index (it is rebuilt from payloads at start-up).
- Per-key permissions for the API keys (today every key can do everything).
- Chained replicas and replica persistence.

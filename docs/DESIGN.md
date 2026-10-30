# vecdb: design decisions

Why the engine is built the way it is. Query/insert walkthroughs, SIMD details and file formats: [ARCHITECTURE.md](ARCHITECTURE.md).

## Scope

An embedded vector database: HNSW approximate nearest-neighbour search with filtering, SQ8 / PQ / binary quantization, upserts and in-place payload edits, BM25 text search fused with vector search, named collections, a WAL plus snapshots with online compaction, and an optional gRPC server that can also run as a read replica. The engine is a plain library; `--no-default-features` drops tonic and tokio. Needs Rust 1.95+ (what `parking_lot` and `lock_api` ask for).

## Decisions

**Searches never wait for inserts.** Each node's adjacency list has its own `RwLock` per layer. A reader copies the list out and releases the lock before computing distances. A writer holds exactly one lock at a time, so there is no lock ordering to get wrong. The accepted cost: a search racing an insert may or may not see the new node, and two concurrent inserts can end up with slightly different edges than a serial build. Tests measure recall after an 8-thread build against ground truth.

**Fixed-stride vector records, one layout in memory and on disk.** A snapshot's vector section is mmap'd read-only, so start-up cost scales with the graph, not the vectors, and the OS pages vectors in on demand. Vectors inserted after open live in slabs with the same layout and go through the same read path.

**Deletes and upserts leave tombstones; compaction runs online.** A deleted node keeps routing queries but is skipped for results and removed from the payload index. An upsert allocates a new node and tombstones the old one, because the vector can't change under a node other threads may be reading. `DurableIndex::compact_online` rebuilds the graph in the background while recording the writes that arrive, replays them onto the new index and swaps it in behind a gate; the server checks every `--compact-check-secs`. An epoch counter makes a stale rebuild throw its result away if the index was replaced meanwhile. Until it runs, tombstones still cost a capacity slot.

**Runtime-dispatched SIMD kernels.** AVX-512, AVX2+FMA and NEON implementations plus a scalar fallback, chosen at startup. A test checks every available kernel against scalar.

**SQ8 with asymmetric distance.** Each vector is quantized on its own (min, scale and norm stored in a 12-byte header), so there's no training pass and streaming inserts work. The query stays f32 and the distance is computed without decoding the vector, using the `dot_u8` kernel. Saves about 73% of vector memory at 128 dims, ~75% at 768. PQ is available too, with optional f32 rerank because PQ alone caps recall (~0.71 on SIFT1M with 32 bytes).

**WAL + generational snapshots with group commit.** Snapshots are never overwritten (`snapshot-<gen>.vdb`, highest generation wins), which also avoids Windows refusing to replace a mapped file. Recovery replays WAL files at or above the snapshot generation; replay is idempotent. Writers queue under a short lock and one becomes the leader that writes and fsyncs everything queued. A failed write poisons the log instead of acknowledging unwritten data.

**Flush by default, fsync on request.** Records go to the OS before the write is acknowledged (survives a process crash). `--wal-fsync` adds an fsync per write (survives power loss).

**Payloads are editable, vectors are not.** `set_payload` and `merge_payload` change a node's payload in place (it sits behind its own `RwLock`) and the filter index follows. Both are WAL records like inserts, so replay stays idempotent.

**Binary quantization is a coarse filter.** One sign bit per component, Hamming distance by popcount. Alone it gets about 0.40 recall@10 on the test data, with `rerank` about 0.99, so treat it as a pre-ranking step, not an index you query raw. It assumes components are roughly centred on zero.

**Hybrid search is rank fusion, not score mixing.** The BM25 index (in memory, rebuilt from payloads at start-up, with Arabic folding of diacritics, tatweel, alef, ya and ta-marbuta) and the vector search each give a ranking, merged by reciprocal rank fusion with k=60. BM25 scores and distances live on different scales, so mixing the scores directly would need tuning per data set; ranks don't.

**Collections share the process, not the index.** The default collection is the index the server started with; others live under `<data-dir>/collections/<name>/` with their own config, WAL and snapshots, picked by the `x-collection` gRPC header. Dropping one writes a marker first, so a crash half-way doesn't resurrect it.

**Replication is small on purpose.** The leader streams its newest snapshot, then tails the WAL files; a replica keeps everything in memory, applies the records and refuses writes. The leader keeps the previous log through a checkpoint so a caught-up replica crosses into the next one without re-downloading (found by a regression test: deleting it at once made every replica resync at every checkpoint). A replica that fell behind a deleted log is told to resync from a fresh snapshot.

**Offline tools share the engine's code.** `vecdb-cli` (inspect, verify, backup, restore, JSONL export/import) opens the data directory with the same loaders the server uses, so `verify` checks what the server would actually read.

## What the benchmarks do and don't say

On random Gaussian vectors recall is poor (0.48 at ef=64 on 20k vectors) because there's no structure for a graph to exploit. That is a property of the data, not a bug; the benchmarks use SIFT-style and latent-structure data.

## Known limits

- AVX-512 and NEON kernels are verified by compilation and review against AVX2, not by execution, since the development machine has neither. CI runs the kernel test on aarch64 (NEON); whether a GitHub runner has AVX-512 is up to the machine it lands on.
- `max_elements` is fixed at creation. Replication is one leader, in-memory asynchronous replicas, no failover, no chaining, and it was only tested on one machine (no network cut mid-transfer). Collections dropped on the leader stay on replicas. No sharding.
- The BM25 index is not persisted. API keys all have full access. The metrics port is plain HTTP.
- PQ needs a trained codebook before the first insert, and a PQ replica needs the leader to have a snapshot (the codebook only travels in one).
- Property and fuzz tests are small because of the development machine; a bigger run belongs on CI or a server.

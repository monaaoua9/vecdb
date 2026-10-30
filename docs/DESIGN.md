# vecdb: design decisions

Why the engine is built the way it is. Query/insert walkthroughs, SIMD details and file formats: [ARCHITECTURE.md](ARCHITECTURE.md).

## Scope

An embedded vector database: HNSW approximate nearest-neighbour search with filtering, SQ8 / PQ quantization, a WAL plus snapshots, and an optional gRPC server. The engine is a plain library; `--no-default-features` drops tonic and tokio.

## Decisions

**Searches never wait for inserts.** Each node's adjacency list has its own `RwLock` per layer. A reader copies the list out and releases the lock before computing distances. A writer holds exactly one lock at a time, so there is no lock ordering to get wrong. The accepted cost: a search racing an insert may or may not see the new node, and two concurrent inserts can end up with slightly different edges than a serial build. Tests measure recall after an 8-thread build against ground truth.

**Fixed-stride vector records, one layout in memory and on disk.** A snapshot's vector section is mmap'd read-only, so start-up cost scales with the graph, not the vectors, and the OS pages vectors in on demand. Vectors inserted after open live in slabs with the same layout and go through the same read path.

**Deletes are tombstones.** A deleted node keeps routing queries but is skipped for results and removed from the payload index. Simple and safe under concurrency; the cost is that compaction is offline (`Index::compacted` rebuilds the graph), run at start-up when the tombstone ratio is high.

**Runtime-dispatched SIMD kernels.** AVX-512, AVX2+FMA and NEON implementations plus a scalar fallback, chosen at startup. A test checks every available kernel against scalar.

**SQ8 with asymmetric distance.** Each vector is quantized on its own (min, scale and norm stored in a 12-byte header), so there's no training pass and streaming inserts work. The query stays f32 and the distance is computed without decoding the vector, using the `dot_u8` kernel. Saves about 73% of vector memory at 128 dims, ~75% at 768. PQ is available too, with optional f32 rerank because PQ alone caps recall (~0.71 on SIFT1M with 32 bytes).

**WAL + generational snapshots with group commit.** Snapshots are never overwritten (`snapshot-<gen>.vdb`, highest generation wins), which also avoids Windows refusing to replace a mapped file. Recovery replays WAL files at or above the snapshot generation; replay is idempotent. Writers queue under a short lock and one becomes the leader that writes and fsyncs everything queued. A failed write poisons the log instead of acknowledging unwritten data.

**Flush by default, fsync on request.** Records go to the OS before the write is acknowledged (survives a process crash). `--wal-fsync` adds an fsync per write (survives power loss).

## What the benchmarks do and don't say

On random Gaussian vectors recall is poor (0.48 at ef=64 on 20k vectors) because there's no structure for a graph to exploit. That is a property of the data, not a bug; the benchmarks use SIFT-style and latent-structure data.

## Known limits

- AVX-512 and NEON kernels are verified by compilation and review against AVX2, not by execution, since the development machine has neither. CI runs the kernel test on aarch64.
- Compaction is offline. `max_elements` is fixed at creation. Payloads are immutable. Single node, no sharding or replication.
- PQ needs a trained codebook before the first insert.

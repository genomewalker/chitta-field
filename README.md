# chitta-field

Organic associative memory substrate for cognitive AI companions.

chitta-field is the backing store for the [cc-soul](https://github.com/genomewalker/cc-soul) daemon. It is designed around the constraints of shared NFS storage, multiple concurrent writers, and sub-millisecond in-process recall.

## What it is

A persistent cognitive memory system built on three layers:

1. **Append-only op log** — durable write path, one segment file per process
2. **In-RAM indexes** — semantic index, BM25 keyword index, cortical sparse index, triplet store, symbol/call graph, temporal index
3. **Snapshots** — periodic binary dumps that accelerate startup by avoiding full log replay

Memories decay over time. High-value memories can be pinned (zero decay). The cortical index encodes memories as Sparse Distributed Representations (SDR, 64 active bits out of 16,384), enabling sub-millisecond associative recall without a learned ANN index for the hot path.

## Architecture

```
chitta-field/
├── Op Log (append-only)
│   └── {instance_id}_{first_seqno}.seg    (one file per writer process)
│
├── Snapshots (binary, periodic)
│   ├── chitta.snapshot                    (full state: payloads + states + indexes)
│   └── cortex.snapshot                    (cortical sparse codes only)
│
└── In-RAM (rebuilt from log on open)
    ├── SemanticIndex  (ANN: IVF coarse + LSH probing, 768-dim BGE embeddings)
    ├── KeywordIndex   (BM25)
    ├── CorticalIndex  (SDR, 64-of-16384 active bits)
    ├── TemporalIndex  (time-ordered, range queries)
    ├── TripletStore   (subject/predicate/object graph)
    ├── SymbolIndex + CallGraph (code intelligence)
    ├── ArtifactIndex  (file path to memory mapping)
    ├── ThemeOrgan     (cluster centroids)
    ├── SessionRegistry / TranscriptRegistry / TaskRegistry
    └── LearnerSet     (resonance / route learners)
```

### Multi-writer design (Upanishads model)

Each process that opens a `ChittaField` is assigned a unique `InstanceId` at startup. All writes go to `{instance_id}_{first_seqno}.seg`, a segment file owned exclusively by that process. No cross-process locking is needed or used.

On `ChittaField::open`, the library scans `data_dir` for all `*.seg` files and replays them in sequence-number order to reconstruct the full in-RAM state. New writers append to their own segment; readers pick up changes by periodically re-opening or through the snapshot path.

This makes chitta-field safe on NFS without lock managers or fencing beyond the fencing token embedded in task/event ops.

### Cortical index (SPAF)

The Sparse Predictive Associative Field encodes each memory's 768-dim embedding into a 64-of-16384 sparse code via a `SparseEncoder`. Recall against the cortical index is a bitset intersection operation, O(active_bits) per candidate, giving sub-millisecond latency even at tens of thousands of memories.

A `ProductQuantizer` compresses residual embeddings for approximate recall at scale. A `LiteEncoder` (bag-of-words to sparse code, no ONNX dependency) provides a fallback encoding path that works without a runtime embedder.

### Iterative recall (CTM-inspired)

Recall in chitta-field is not a single forward pass. Inspired by the Continuous Thought Machine (Sakana AI, 2025), the resonance engine runs up to **3 iterative passes** per query:

1. **Pass 1** — standard semantic + BM25 + spreading activation with original query `q₀`
2. **Refinement** — refined query `q₁ = normalize(0.7·q₀ + 0.3·mean(top-k embeddings))`
3. **Pass 2+** — repeat with refined query; stop when entropy delta < 0.01 or top-k set is stable

After the final pass, `cf_record_recall_batch` atomically commits all learning:
- **Per-memory retrieval context** — each retrieved memory logs a 32-dim quantized sketch of the query context (up to 8 entries, FIFO). The cached mean signature is used to boost future recall of memories whose context matches the current query.
- **Sync-weighted Hebbian learning** — co-retrieved pairs update `CoActivationStats` (sim_count × diversity_count). Edge strengthening scales with this multiplier, so memories that reliably fire together across *diverse* query contexts earn much stronger assoc edges than accidental co-retrievals.

## Key features

- **Multi-writer NFS** — no cross-process locks; per-instance segment files
- **Sub-millisecond cortical recall** — SDR bitset intersection
- **Iterative resonance** — up to 3 passes with query refinement and entropy early-stop
- **Retrieval context history** — per-memory 32-dim query sketch log for context-aware reranking
- **Sync-weighted Hebbian edges** — co-activation diversity scales edge reinforcement
- **Semantic recall** — ANN (IVF + LSH) over BGE-base-en-v1.5 768-dim embeddings
- **Keyword recall** — BM25 over memory content
- **Temporal recall** — time-range queries with kind/realm filters
- **Association graph** — directed weighted edges between memories (DerivedFrom, SameSession, SameArtifact, CoRetrieved, Supports, Contradicts)
- **Triplet store** — subject/predicate/object knowledge graph
- **Code intelligence** — symbol index + call graph with semantic search
- **Theme clustering** — centroid-based theme assignment and recall
- **Decay and demotion** — configurable per-memory decay rate; `cf_run_demotion` demotes weak memories to cold tier
- **Resonance learning** — feedback signal updates retrieval weights via `LearnerSet`
- **Session / transcript / task registries** — first-class domain objects for AI session management
- **Snapshot acceleration** — `cf_save_full_snapshot` / `cf_save_snapshot` skip log replay on next open

## Memory model

Each memory carries:

| Field | Type | Description |
|-------|------|-------------|
| `memory_id` | `u64` | Monotonically allocated, per-instance |
| `chunk_hash` | `[u8; 32]` | SHA-256 of (kind, realm, content, embedding) |
| `kind` | `String` | Semantic category (e.g. `"ssl"`, `"episode"`, `"fact"`) |
| `realm` | `String` | Namespace / project scope |
| `content` | `Vec<u8>` | Raw bytes (typically UTF-8 text) |
| `embedding` | `Vec<f32>` | 768-dim BGE embedding |
| `confidence` | `f32` | 0.0-1.0, updated by feedback |
| `decay_rate` | `f32` | Strength loss per time unit; 0.0 = pinned |
| `strength` | `f32` | Current salience (decays over time) |
| `sparse_code` | `Option<SparseCode>` | 64 active bit indices into 16,384-dim cortex |
| `authored_at_ms` | `i64` | Original authorship timestamp |
| `created_at_ms` | `i64` | Ingestion timestamp |

## C FFI

chitta-field compiles to a static library (`libchitta_field.a`) and exposes a C API via `include/chitta_field.h`. All functions return `0` on success, negative on error. Errors are readable via `cf_last_error`.

### Lifecycle

```c
CfHandle* h = cf_open("/path/to/field-dir", NULL);
// ... use h ...
cf_close(h);
```

### Write operations

```c
uint64_t memory_id;
cf_put_memory(h, "fact", "project/foo",
    content, content_len,
    embedding, 768,
    0.8f, 0.01f, authored_at_ms,
    &memory_id);

cf_update_state(h, memory_id, +0.1f, 0.0f, -1.0f, /*touch=*/1, /*pin=*/-1);
cf_forget(h, memory_id);
cf_add_assoc_edge(h, src_id, dst_id, /*edge_type=*/0, 1.0f);
```

### Read operations

```c
CfRecallHit hits[32];
size_t written;
cf_recall_semantic(h, query_embedding, 768, "project/foo", 10,
    hits, 32, &written);

cf_recall_temporal(h, start_ms, end_ms, "project/foo", 20,
    hits, 32, &written);

cf_recall_keyword(h, "search terms", 10, hits, 32, &written);

cf_expand_associations(h, seed_ids, seed_count, /*max_hops=*/2, 20,
    hits, 32, &written);
```

### Triplets

```c
uint64_t triplet_id;
cf_add_triplet(h, "chitta-field", "is_part_of", "cc-soul", 1.0f, memory_id, &triplet_id);
cf_invalidate_triplet(h, triplet_id);

char buf[65536]; size_t written;
cf_query_subject(h, "chitta-field", buf, sizeof(buf), &written);
```

### Code intelligence

```c
uint64_t sym_id;
cf_upsert_symbol(h, "function", "put_memory", "(kind, realm, ...) -> Result<...>",
    "src/store.rs", 30, 120, repo_id, embedding, 768, "Store a new memory", 0, &sym_id);

cf_add_sym_call_edge(h, caller_id, callee_id);

CfSymbolHit syms[16]; size_t written;
cf_search_symbols_by_name(h, "recall", 10, syms, 16, &written);
cf_search_symbols_semantic(h, query_embedding, 768, 10, syms, 16, &written);
```

### Maintenance

```c
cf_flush(h);
cf_save_full_snapshot(h);
cf_save_snapshot(h);              // cortical index only
uint64_t demoted = cf_run_demotion(h, now_ms);
```

## Building

Requires Rust 1.92.0 (pinned via `rust-toolchain.toml`). If using conda, unset any conda-injected linker overrides before building; `build.sh` handles this automatically.

```bash
# Build static lib + binaries
./build.sh build --release

# Run tests
./build.sh test

# Build with optimized profile (LTO, single codegen unit)
./build.sh build --release --lib
```

The static library is written to `target/release/libchitta_field.a`. The C header is at `include/chitta_field.h`.

## CLI tools

### encode

Index all unencoded memories into the cortical sparse index and optionally save snapshots.

```bash
./build.sh run --bin encode --release -- \
    --field-dir ~/.claude/mind/chitta-field \
    [--encode-pq] \
    [--save-snapshot] \
    [--save-full-snapshot]
```

### migrate

Import memories and triplets from a JSONL export.

```bash
./build.sh run --bin migrate --release -- \
    --memories /tmp/chitta_migration/memories.jsonl \
    --triplets /tmp/chitta_migration/triplets.jsonl \
    --field-dir ~/.claude/mind/chitta-field
```

Memories without embeddings (`"embedding": null`) receive a zero-vector placeholder. Re-encode them afterwards with `encode`. Pinned memories are imported with `confidence=1.0` and `decay_rate=0.0`.

### train_lite

Train the bag-of-words lite encoder from existing memories and save it to disk. This enables `cf_encode_lite` without an ONNX runtime.

```bash
./build.sh run --bin train_lite --release -- \
    --field-dir ~/.claude/mind/chitta-field
```

## Data directory layout

```
~/.claude/mind/chitta-field/
├── {instance_id}_{seqno}.seg    (op log segment, one per writer process)
├── chitta.snapshot              (full state snapshot, optional, speeds startup)
└── cortex.snapshot              (cortical index snapshot, optional)
```

## Theoretical foundations

chitta-field's design is grounded in computational neuroscience and information retrieval research. Each component maps to a specific body of literature.

### Sparse Distributed Representations (SDR)

The cortical index encodes each memory as a sparse distributed representation: exactly K=64 active features out of N=16,384 (`K_ACTIVE=64`, `N_ATOMS=16384` in `cortex.rs`). SDRs were developed at Numenta as a model of neocortical computation. Their key properties (high capacity, fault tolerance, and fast overlap-based matching) are the reason cortical recall is sub-millisecond without a learned index structure.

> Ahmad, S., & Hawkins, J. (2016). How do neurons operate on sparse distributed representations? A mathematical theory of sparsity, neurons and active dendrites. *arXiv:1601.00720*.
>
> Hawkins, J., & Ahmad, S. (2016). Why neurons have thousands of synapses, a theory of sequence memory in neocortex. *Frontiers in Neural Circuits*, 10, 23. https://doi.org/10.3389/fncir.2016.00023

### Product-key sparse encoding

Computing the top-K atoms over N=16,384 full-dimensional atoms would be O(N·d). `SparseEncoder` in `cortex.rs` instead uses the product-key decomposition: the 768-dim embedding is split into two 384-dim halves, each scored against 128-centroid sub-dictionaries, and the top-K atoms are selected from the 256-candidate shortlist. This reduces the cost to O(sqrt(N) · d), the same approach introduced for large memory layers.

> Lample, G., Sablayrolles, A., Ranzato, M. A., Denoyer, L., & Jégou, H. (2019). Large memory layers with product keys. *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*. https://arxiv.org/abs/1907.05242

### Product quantization for residual compression

`ProductQuantizer` in `pq.rs` compresses residual embeddings (768-dim, 32 subvectors of 24 dims each, 256 centroids per subspace) to 32 bytes using standard product quantization.

> Jégou, H., Douze, M., & Schmid, C. (2011). Product quantization for nearest neighbor search. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 33(1), 117-128. https://doi.org/10.1109/TPAMI.2010.57

### ART-inspired prototype learning

`PrototypeIndex` in `prototype.rs` assigns memories to prototype clusters using a vigilance threshold (`VIGILANCE=0.003`). New memories are committed to a new prototype if no existing prototype exceeds the vigilance criterion, a direct implementation of the Adaptive Resonance Theory commitment rule.

> Carpenter, G. A., & Grossberg, S. (1987). A massively parallel architecture for a self-organizing neural pattern recognition machine. *Computer Vision, Graphics, and Image Processing*, 37(1), 54-115. https://doi.org/10.1016/S0734-189X(87)80014-2
>
> Grossberg, S. (1987). Competitive learning: From interactive activation to adaptive resonance. *Cognitive Science*, 11(1), 23-63. https://doi.org/10.1111/j.1551-6708.1987.tb00862.x

### Complementary learning systems

chitta-field separates fast episodic storage (op log, segment files, analogous to hippocampus) from slow cortical consolidation (snapshot, prototype index, cortical SDR codes, analogous to neocortex). This two-speed architecture reflects the CLS theory of memory consolidation.

> McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. *Psychological Review*, 102(3), 419-457. https://doi.org/10.1037/0033-295X.102.3.419

### Decay and the spacing effect

Each memory has a per-instance `decay_rate` (strength loss per unit time). `PlasticityLearner` in `plasticity.rs` adjusts this rate using an exponentially weighted moving average of inter-access intervals: frequently accessed memories earn lower decay rates. This models the spacing effect, where repeated retrieval at spaced intervals strengthens retention.

> Ebbinghaus, H. (1885). *Über das Gedächtnis: Untersuchungen zur experimentellen Psychologie*. Duncker & Humblot. (English translation: Memory: A Contribution to Experimental Psychology, 1913.)

### Thompson sampling for retrieval route selection

`RouteLearner` in `route.rs` uses `BetaPrior` (Thompson sampling) to select among retrieval routes (Semantic, Keyword, Temporal, Hybrid, Full) based on observed recall quality. Beta-distributed priors are updated per route per query intent; sampling provides exploration-exploitation balance.

> Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. *Biometrika*, 25(3-4), 285-294. https://doi.org/10.1093/biomet/25.3-4.285
>
> Chapelle, O., & Li, L. (2011). An empirical evaluation of Thompson sampling. *Advances in Neural Information Processing Systems 24 (NIPS 2011)*. https://papers.nips.cc/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html

### BM25 keyword index

The keyword index in `keyword.rs` uses BM25 with standard parameters (k1=1.2, b=0.75), a probabilistic term-weighting model that remains state-of-the-art for sparse keyword retrieval.

> Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389. https://doi.org/10.1561/1500000019

### Embeddings

Semantic recall uses BGE-base-en-v1.5 embeddings (768-dim, from BAAI), provided via the VakYantra ONNX embedder in the cc-soul daemon.

> Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). C-Pack: Packaged resources to advance general Chinese embedding. *arXiv:2309.07597*. https://arxiv.org/abs/2309.07597

The semantic index uses a two-tier ANN strategy: LSH probing (4 tables, 12 bits) for primary candidate generation, falling back to IVF coarse quantizer (256 random-projection centroids) if LSH yields no candidates. Exact cosine reranking runs over the bounded candidate set (1024–16384). This bounds query cost well below brute-force even at 50K+ memories.

## Dependencies

| Crate | Purpose |
|-------|---------|
| `serde` + `serde_json` + `rmp-serde` | Op serialization (MessagePack for log, JSON for FFI returns) |
| `bincode` | Snapshot serialization |
| `memmap2` | Memory-mapped segment file reads |
| `parking_lot` | `RwLock` for all in-RAM indexes |
| `crc32fast` | Frame checksums in op log |
| `sha2` | Chunk hash (content dedup) |
| `byteorder` | Frame header encoding |
| `thiserror` | Error types |

No async runtime. No database engine. No network. All state lives in RAM, written through to the op log.

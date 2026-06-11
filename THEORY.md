# THEORY — chitta-field as a coordination-free replicated memory

This document states what chitta-field *is* as a distributed system, which
correctness properties it has, which it lacks, and the theory that should
guide the next phase. Everything here is grounded in code as of v2.1.0;
file:line references are to this repo unless noted.

## 1. The discovered topology

chitta-field was designed as an embedded store for one daemon. In production
it is something else: **a leaderless multi-master replicated store over a
shared NFS substrate.** Every cluster node where a session starts runs its own
`chittad` (node-local socket, `XDG_RUNTIME_DIR`), and all of them write one
store directory concurrently. Coordination is limited to:

- **instance-partitioned WALs** — each daemon writes only its own segments
  (`segments/{instance_id:08x}_{first_seqno}.seg`, `log.rs`);
- **instance-partitioned id allocators** (`ids.rs`) — no key collisions on
  create;
- **snapshot families per instance** + the V23 manifest commit record
  (`manifest.rs`, `snapshot.rs`).

This topology cannot be replaced by single-writer election: the substrate
offers no reliable locks (NFS), and deleted files resurrect on this volume.
The store is therefore *forced* toward coordination-free correctness — the
CRDT regime — and the design should embrace that rather than fight it.

## 2. Convergence: the property we want

**Strong eventual consistency (SEC):** any two daemons that have replayed the
same *set* of WAL segments hold identical state, regardless of the order in
which the ops were originally interleaved.

Replay applies all instances' segments in **directory-sort order**
(`log.rs::replay`), which is instance-id order — not causal order, not
timestamp order. SEC therefore holds iff `apply_op` (`field.rs:1340`) is
insensitive to cross-instance interleaving. Today it is not, for three
specific mechanisms:

### 2.1 The op taxonomy

| Class | Examples | Replay semantics today | Verdict |
|---|---|---|---|
| **(a) Commutative by construction** | `PutPayload` (partitioned ids), `AddTriplet` (replay_add by id), `UpsertArtifact` (`entry().or_insert`), `AddAssocEdge` (multiset append) | order-insensitive | safe |
| **(b) Timestamp-guarded state deltas** | `UpdateState` (strength/confidence deltas, touch, pin, status) | see 2.2 — out-of-order ops are **wholly discarded** | unsafe across instances |
| **(c) Absolute writes without timestamps** | content updates, kind updates, some organ events | last-in-replay-order wins (= instance-id order, arbitrary) | unsafe across instances |

### 2.2 The full-drop guard (the sharpest finding)

`MemoryState::apply_delta` (`state.rs:221`) begins:

```rust
if delta.op_ts_ms > 0 && delta.op_ts_ms <= self.last_state_op_ts_ms { return; }
```

This guard was written for idempotent re-replay within one instance. Across
instances it has a harsher consequence: if segment order applies a *newer* op
before an *older* one, the older op is rejected **in its entirety — additive
fields included**. A `strength_delta` is not an LWW register; dropping it is
data loss, not conflict resolution. So even the "commutative-looking" parts
of class (b) are only confluent when segment sort order happens to agree with
per-memory timestamp order.

### 2.3 The causality gap

`Op::PutPayload` creates state via `or_insert_with` (`field.rs:1399`), and
`Op::UpdateState` silently no-ops when the state is absent (`field.rs:1435`).
In real time, an update always follows the create it depends on (a daemon can
only update an id it has seen). In *replay* order, an update from a
lexically-earlier instance is applied before the create from a
lexically-later one — and is silently dropped.

### 2.4 The actual safe envelope (and why it mostly works today)

The system currently converges in practice because of **session affinity**:
a memory is usually touched only by the node that created it within one WAL
epoch, and snapshots collapse history before cross-instance writes to the
same memory accumulate. Since v2.2.0 the envelope is the full op set: the
`replay_confluent_*` tests in `store.rs` assert convergence under random
instance permutations, and `merge_replay_applies_*` / `orphan_delta_*`
assert that both pre-v2.2 loss modes stay fixed.

## 3. The fix: deterministic merge replay — **implemented in v2.2.0**

Make replay a **k-way merge of segments ordered by `(op_ts_ms, instance_id,
seqno)`** instead of segment concatenation. Consequences:

1. The `apply_delta` guard becomes *sound*: apply order equals timestamp
   order, so the guard only ever rejects true duplicates (idempotent
   re-replay), never reorders.
2. Class (c) ops become honest LWW registers: give every absolute write an
   `op_ts_ms` and the tiebreak `(ts, instance_id)` is a total order.
3. The causality gap closes for free in practice (creates carry the earliest
   timestamp for their id), with a belt-and-braces option: buffer orphan
   deltas keyed by memory_id and apply them when the create arrives.

After this change SEC is provable by the standard CRDT argument: every op
class is either (a) commutative, (b) guarded by a total order, or (c) an LWW
register in that same total order; state is then a deterministic function of
the op *set*.

**Property test, not proof assistant:** the invariant is enforced by
permutation tests — apply a random op set under random instance assignments
(which permutes replay order) and assert state equality. See
`replay_confluent_for_disjoint_memories` for the template.

## 4. The manifest should be a version vector — **implemented in v2.2.0**

> Status: `Manifest.families` is the version vector (joined per-entry across
> both slots); `CheckpointSet.covered` is the per-writer coverage vector used
> by both the replay skip and `prune_covered_segments`. The scalar-seqno skip
> survives only as the fallback for pre-v2.2 snapshots without a family.
> Known residual: an orphaned delta older than an already-applied delta on
> the same memory is still rejected by the `apply_delta` guard (idempotency
> for legacy full-replay); becomes irrelevant once all snapshots carry
> coverage vectors and replay never re-walks covered ops.

The V23 manifest (`manifest.rs`) is a scalar `generation` counter. With N
concurrent writers, two daemons can both compute `generation = g+1` and the
slot write becomes last-rename-wins: atomic and validated, but one writer's
commit record is shadowed (observed in production, 2026-06-10).

The classic fix: the commit record becomes a **map
`instance_id → CheckpointRef{family, max_seqno}`** — a version vector. Joins
are per-entry max, so concurrent commits merge instead of shadowing. This
also yields a *formal pruning rule*:

> WAL segment S of instance i with range `[a, b]` may be pruned iff the
> merged manifest's entry for i has `max_seqno ≥ b`.

That rule is **NFS-resurrection-proof**: a resurrected segment is dominated
by the manifest and replay can skip it outright, which the current
filename-seqno heuristic cannot guarantee.

## 5. Why not single-writer leases (considered, rejected)

Fencing primitives exist in embryo (`lineage_epoch`, `writer_uuid`,
`vector_space_id` in the `.shdr` header). A lease/epoch scheme could force a
single writer per store. Rejected because: (i) lease files need the very
storage semantics NFS denies us (atomic visibility, reliable deletion);
(ii) it adds an availability failure mode (lease-holder node dies → store
read-only until TTL); (iii) the workload — append-mostly memory accumulation
with per-instance keys — is precisely the workload CRDTs are good at. Keep
fencing for what it already does well: excluding *foreign vector spaces*,
not serializing writers.

## 6. The cognitive reading: consolidation *is* the merge function

The distributed-systems theory and the memory theory coincide, and that is a
design resource, not a metaphor:

- **Multiple hippocampi, one cortex.** Per-node daemons are episodic buffers;
  the shared store is consolidated cortex. Biology integrates parallel
  episodic traces without a global lock; the CRDT merge function is the
  systems-consolidation operator. Each semilattice chosen for SEC has an
  independent psychological motivation: bounded-sum strength (supra-additive
  trace merging), merged access-timestamp sets (spacing effect, ACT-R
  power-law activation), max-merge for monotone scores.
- **Forgetting is the tombstone problem, solved.** Deletes are the hard part
  of CRDTs; here `forget` is a flag plus monotone decay plus thresholded GC —
  a naturally convergent delete. The interference/decay machinery is not just
  scoring; it is what makes coordination-free deletion sound.
- **Eventual consistency is cognitively justified.** `competitive_weight` is
  a local approximation of a global graph property; biological memory has no
  synchronous global state either. Staleness tolerance is a feature with a
  principled bound (the refresh interval), not a bug.
- **The falsifiable claim:** recall quality must be insensitive to which node
  wrote a memory and to replay interleaving. Run LOCOMO with single-writer
  vs. shuffled multi-writer replay; SEC predicts identical scores. Divergence
  localizes exactly the ops that violate §2.

## 7. Invariants to enforce going forward

1. **Confluence** — state is a function of the op set (permutation property
   test; §3).
2. **Prefix recoverability** — after any crash, `open()` yields the state of
   some prefix of committed history (manifest commit point + torn-tail
   truncation preserve this; tested by `torn_tail_*` and
   `test_manifest_commits_snapshot_family`).
3. **Monotone lattice merges** — every per-memory scalar that multiple
   writers may touch must declare its merge (max / bounded-sum / LWW-by-ts).
   New `Op` variants must state their class from §2.1 in a doc comment.
4. **No cross-language lock interleaving** — FFI is one-directional; C++
   locks strictly outermost (documented at both boundaries in cc-soul).

## 8. Roadmap (theory → practice)

Phase 1 — *correctness* (small, high payoff):
merge replay by `(op_ts, instance_id, seqno)`; orphan-delta buffer;
`op_ts_ms` on every absolute write; manifest v2 as version vector with the
§4 pruning rule; extend the permutation test to the full op set.

Phase 2 — *architecture*: organ trait (behavior moves out of the
`ChittaField` god object; locks become organ-local); per-organ sections of
the V23 snapshot get dirty-tracking so unchanged organs are not re-cloned or
re-written (kills the 2× RSS spike and most of the save cost); single FFI
error envelope.

> Status (v2.5.5): the OrganApply trait landed — 44 op variants across 20
> organs apply themselves in their own modules; apply_op dispatches organs
> first and its central match keeps only multi-structure ops, with the
> consumed variants listed explicitly so a new Op variant still breaks the
> build until it gets a handler or an organ. Earlier (v2.3.0): the two
> heaviest save costs are gone — payload embeddings
> (632MB of the body, duplicating the .emb sidecar) are stripped at save and
> rehydrated at open (missing ones self-heal via embed_pending), and the six
> index sidecars (~800MB) are dirty-skipped via a SemanticIndex mutation
> counter when the index hasn't changed since the last save. Class-(c) ops
> (`UpdateMemoryContent`/`UpdateMemoryKind`/`UpdateSymbolDescription`) now
> carry `op_ts_ms` — honest LWW registers under merge replay. Remaining:
> organ trait, per-organ dirty-tracking beyond SemanticIndex, FFI envelope.

Phase 3 — *cognition*: explicit consolidation operator (the merge function,
run as sleep-phase compaction rather than implicit replay); cross-node
episodic→semantic promotion; LOCOMO-driven evaluation loop as CI for memory
quality; online learned-scorer updates with the SEC-safe merge.

> Status (v2.5.0): cross-context provenance landed — RecordRecallBatch ops
> accrue the set of distinct recalling instances per memory (live, replay,
> and sync_foreign paths; capped at 8), persisted as the V23
> "recall_provenance" section (the first field added with zero migration
> code, as designed), and feeding a config-gated generality boost
> (`cross_context_weight`, default 0.05). The recall-level SEC experiment is
> now a unit test: `recall_ranking_invariant_under_writer_permutation`.
> Remaining: explicit consolidation pass, LOCOMO-in-CI, online scorer merge.

Phase 4 — *scale*: realm-sharded stores with per-shard manifests; L2/L3
tiers on object storage; replication beyond a single NFS volume using the
same version-vector machinery (it is already a replication protocol — it
just doesn't know it yet).

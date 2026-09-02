# LOCKING — chitta-field lock hierarchy and the rules the post-mortems imply

Status as of 2026-09-02.

This is the doc-level companion to the in-code lock notes. The comments at the
sites cited below are authoritative and stay where they are; this file states
the hierarchy once, and turns each post-mortem into a rule you can check a diff
against.

Every lock here is a `parking_lot::RwLock`. Two properties drive everything
else:

- **Writer-preferring.** A queued writer blocks readers that arrive after it.
  One slow writer therefore stalls the whole read fleet, not just itself.
- **Not reentrant.** Re-acquiring a lock you already hold on the same thread
  deadlocks. There is no debug assertion for this in release builds.

The combination is why a long computation under a *read* guard is just as
dangerous as one under a write guard: the read guard blocks the queued writer,
and the queued writer blocks every later reader. One such convoy already
required SIGKILL to clear (commit a20070a).

## 1. Tier order

Canonical definition: `src/field.rs:129-151`. When holding more than one lock,
acquire in ascending tier order and release before acquiring anything from a
lower tier.

| Tier | Locks | Role |
|---|---|---|
| 0 | `log`, `id_alloc` | append / allocate primitives |
| 1 | `payloads`, `states`, `assoc_edges` | core memory data |
| 2 | `semantic_idx`, `time_idx`, `keyword_idx`, `artifact_idx`, `hdc_idx`, cortical maps, `realm_members` | derived indexes |
| 3 | `triplet_store`, `symbol_idx`, `call_graph`, `code_files` | knowledge graph |
| 4 | `scoring_pipeline`, `learners`, `ack_scores`, `coactivation_stats`, `cw_refresh_inflight` | scoring / stats |
| 5 | organs: `event_tape`, `decision_tape`, `turiya_monitor`, `observer_state`, `interaction_ledger`, `predicate_store`, ... | append-mostly organs |

Within tier 1 the order is **`payloads` before `states`**. This is not
arbitrary and is asserted at two audited sites: `src/store.rs:1871` and
`src/store.rs:2354`.

`session_recent` is ordered *after* `payloads`. Holding `session_recent` while
reaching for `payloads` is an inversion; `src/store.rs:1630-1637` collects ids
under `session_recent` alone and drops it before touching `payloads`/`states`.

There is **no static ordering check**. The only enforcement is a runtime cycle
detector: build with `--features deadlock-detection` (CI does) to get a checker
thread that dumps lock cycles. New multi-lock sites must be tier-ordered by
review.

## 2. Rules

### R1 — Never compute under a guard. Snapshot, drop, compute, re-acquire.

The two-phase pattern (collect under reads → drop → apply under one brief
write) is the default for anything that is not O(1). Cloning is cheaper than
the convoy: `TurnEvent` is 32 bytes, and the tape clone was chosen precisely
because it beats holding the lock.

Post-mortems that produced this rule:

- `src/store.rs:4469-4474` — `run_sequitur` takes ~30s on a large tape. Held
  under `event_tape.read()` it blocked a queued tape writer (`put_memory` /
  `log_event`), and writer-fairness then blocked every subsequent reader. Fix:
  clone the tape under a brief read, run off-lock.
- `src/store.rs:4505-4512` — the O(n) `cdawg.surprisal` sweep stalled recall
  for 13-35s by the same mechanism. Fix: snapshot tape+cdawg under one brief
  read in tape→cdawg order, compute the removal mask off-lock, apply under a
  short write. Events appended during the sweep (indices past the mask) are
  kept, which is what makes the off-lock computation safe.
- `src/store.rs:4532-4533` — the FEP rebuild is O(events); same treatment,
  explicitly cross-referenced to `run_sequitur`.
- `src/store.rs:8516-8519` — `predicate_run` spawns subprocesses. Commands are
  collected under a read guard in phase 1; the subprocesses run in phase 2
  outside every lock. A subprocess under a guard is an unbounded hold.

### R2 — The analogy lane runs on copies, never under the triplet guard.

Hypervector scoring is a long CPU pass over the whole fact table. `Fact`
(`src/analogy.rs:42-44`) exists so that scoring never touches
`triplet_store`. `analogy_snapshot` (`src/store.rs:2209-2212`) copies the lane
out and drops both guards before the caller does any hypervector work.

Corollary (`src/store.rs:2271-2277`): the snapshot carries **realm only**, not
payload text. Text for the ranked subset is fetched afterwards by
`analogy_texts`, which takes `payloads` alone — a different lock from the
triplet guard, and never held together with it. Fetching text inside the
snapshot both widened the copy to the size of the store and coupled two tiers
for no benefit, since the realm filter discards most of the over-fetch.

### R3 — Batch a burst into one acquisition.

Re-acquiring a write lock per item lets writer-preference starve readers across
the whole burst. `backfill_stage` (`src/store.rs:6155-6158`) takes **one**
`semantic_idx.write()` for an entire batch, so a waiting recall reader stalls
for one brief metadata write rather than for the full chunk.

### R4 — Split a durable write into stage / plan / apply by lock requirement.

`src/store.rs:6091-6108` is the worked example, and the reasoning generalises to
any long index mutation:

| Phase | Lock requirement | Why |
|---|---|---|
| `stage` | under the C++ `rpc_mutex` | WAL + payload meta + metadata upsert; brief write |
| `plan` | **without** the `rpc_mutex` | HNSW plans computed under `semantic_idx.read()`, concurrent with recall; holding `rpc_mutex` here would block recall for the whole search |
| `apply` | under the `rpc_mutex` | preserves the `rpc → semantic_idx` order, so no new ABBA against `sync_foreign` |

Crash safety is what lets the phases be separated: durable state lands in
`stage`, so a crash between `stage` and `apply` leaves the memory
`embed_pending` and it is re-picked on the next drain. No torn graph, no lost
write.

### R5 — Do not hold a store lock across a slow syscall.

`src/store.rs:549-552`: on a replicating NFS mount (Isilon) a multi-GB
`ftruncate` blocks for tens of seconds. Run under the snapshot-save lock it
starved recall and every other op (observed: a 72-deep pool stall). Snapshot
pruning therefore uses a plain `unlink`, which is metadata-only, and explicitly
does **not** truncate before unlinking.

### R6 — A lock taken alone creates no ordering obligation; say so.

`src/store.rs:5292-5293` builds the window-membership set from `time_idx` and
drops it before any lane runs, with the note that it is taken alone and so
forms no ordering pair. Recording this is what keeps a later reader from
"fixing" it into a longer hold.

### R7 — Check existence under a short read before the WAL append.

`record_outcome` (`src/store.rs:2181-2185`) validates the id under a brief
`states.read()`, appends, then accrues under one `states` write. The same note
carries the operational rule: **never call it while iterating recall results** —
a queued writer stalls every later reader.

## 3. Reviewing a diff

Ask, in order:

1. Does this hold two locks at once? If so, are they in ascending tier order,
   and is `payloads` before `states`?
2. Is there anything between acquire and release that is not O(1) — a sort, a
   scan, a rebuild, a syscall, a subprocess, an allocation proportional to the
   store? If yes, apply R1.
3. Does a loop re-acquire the same write lock per iteration? If yes, apply R3.
4. Does a read guard span a call whose cost depends on data size rather than on
   the item being handled? That is the convoy shape, and it is a bug on the
   recall path.

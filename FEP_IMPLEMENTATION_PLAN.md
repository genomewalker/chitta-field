# FEP / Hopfield Implementation Plan (chitta-field)

**Scope**: Implement the v5.3.0 FEP features retracted in `cc-soul/CHANGELOG.md:7`.

---

## 1. Reality Check (per-feature, with file:line evidence)

### Feature A — FEP attractor network (umbrella item)
Umbrella claim; see B–H below. No `hopfield.rs` file exists: `Grep "hopfield|Hopfield"` across `/maps/projects/fernandezguerra/apps/repos/chitta-field` returned **no matches**.

### Feature B — Asymmetric prototype transitions (`strengthen_transition` forward/reverse 1.0 / 0.3)
**Partially exists, but currently SYMMETRIC.** `PrototypeIndex::strengthen_transition` at `chitta-field/src/organ/prototype.rs:169-178` writes the same value to both `(a,b)` and `(b,a)` keys — see lines 176-177. The API exists; the asymmetry is not implemented. `transitions: HashMap<(ProtoId, ProtoId), f32>` at `prototype.rs:21` supports direction already.

### Feature C — Asymmetric triplet weights (`reverse_weight` field on `TripletEntry`)
**Does not exist.** `Grep "reverse_weight"` returned **no matches** anywhere in chitta-field. `TripletEntry` at `organ/triplet.rs:6-16` has fields: `id, subject, predicate, object, weight, valid_from_ms, valid_to_ms, source_memory_id, source_file`. No reverse weight. Also no `#[serde(default)]` markers on any field yet — backward-compat rewrite required if added.

### Feature D — Surprise-modulated plasticity (`MemoryState.surprise`)
**Does not exist.** `Grep "surprise"` returned **no matches** in chitta-field. `MemoryState` at `state.rs:79-107` has `strength, decay_rate, confidence, access_count, …, epistemic_status` but no `surprise` field. `PlasticityLearner` at `learner/plasticity.rs:12-68` is access-interval EWMA only — it does not consume any surprise/prediction-error signal. The claim "`PlasticityLearner` uses it" is aspirational.

### Feature E — Attractor-based pattern completion (`CorticalIndex::attractor_settle`, `Route::Attractor`)
**Does not exist.** `attractor_settle` has no definition; `Route::Attractor` is not present. The required substrate exists: `PrototypeIndex::transition(a,b)` (`prototype.rs:188-190`), centroid storage (`prototype.rs:11-15`), and `CorticalIndex::strengthen_proto_transitions` (`organ/cortex.rs:454-466`).

### Feature F — Hopfield network module (`hopfield.rs`)
**Does not exist.** No `hopfield.rs` in the crate.

### Feature G — Self-orthogonalizing sparse encoder (FEP rule + Gram-Schmidt decorrelation)
**Current encoder is classic Hebbian/Oja.** `SparseEncoder::update` at `organ/cortex.rs:158-182` implements Oja's rule (`*w += ENCODER_LR * act * (x - act * *w)`) plus per-atom L2 renormalization. No prediction-error term, no complexity penalty, no inter-atom decorrelation step. Rewrite, not extension.

### Feature H — Adaptive vigilance (`CorticalIndex::adapt_vigilance`)
**Does not exist.** `PrototypeIndex` has a constant `VIGILANCE: f32 = 0.003` (`prototype.rs:7`) and a field `vigilance: f32` (`prototype.rs:22`), but it is never mutated after construction (`prototype.rs:38`). No adapt method is exposed on `CorticalIndex`.

### Feature I — Free-energy merge in `find_dup_pairs`
`find_dup_pairs` uses cosine-only today. The reconstruction-error branch does not exist because `cf_reconstruction_error` does not exist.

### FFI symbols
All six missing from `ffi.rs`. Existing FFI scaffolding is mature (5266-line file) and adding six C functions is mechanical.

### Supporting substrate that DOES exist and we can reuse
- `SparseEncoder::encode` / `decode` (`cortex.rs:72-129`, `135-154`) — reconstruction error is trivially `‖embedding − decode(encode(embedding))‖`.
- `PrototypeIndex.transitions` directed map (`prototype.rs:21`).
- `CorticalIndex.mem_codes` (`cortex.rs:205`) — per-memory sparse codes for Hopfield co-activation.
- `strengthen_proto_transitions` (`cortex.rs:454-466`) — hook for co-retrieval updates.
- `RwLock<SparseEncoder>` and `RwLock<CorticalIndex>` on the field (`field.rs:111-112`) — lock discipline already in place.

---

## 2. Dependency Graph

```
                 ┌── SparseEncoder (exists, needs rewrite in Phase 3)
                 │
reconstruction_error ──► memory_surprise (state field)
         │                       │
         │                       └──► surprise-modulated plasticity (Phase 2)
         │                       └──► free-energy merge criterion (Phase 4)
         │
         └──► adapt_vigilance (Phase 2)
         └──► free-energy merge (Phase 4)

strengthen_transition (asymmetric) ──► attractor_settle (Phase 3)
                                  ──► Hopfield settle (Phase 3)

TripletEntry.reverse_weight (Phase 2, independent)

Hopfield co-activation ◄── mem_codes index (exists)
                      ──► cf_hopfield_co_retrieval, cf_hopfield_stats
```

Phase-1 primitives (reconstruction_error, memory_surprise read/write) unblock every downstream feature including MCP wrappers on the daemon side.

---

## 3. Phasing (one PR each)

### Phase 1 — FFI stubs + reconstruction_error (unblocks MCP)
Minimal scope so `cc-soul` MCP wrappers compile and return real values.

**Create/modify**
- `src/ffi.rs` (~+200 LOC): add `cf_reconstruction_error(field_handle, embedding_ptr, len, out_f32)`, `cf_memory_surprise(field_handle, mem_id, out_f32)` (reads `MemoryState.surprise` default 0.0), and stub `cf_search_attractor`/`cf_hopfield_co_retrieval`/`cf_hopfield_stats`/`cf_adapt_vigilance` returning `Err::NotImplemented` with stable signatures.
- `src/state.rs` (~+4 LOC): add `#[serde(default)] pub surprise: f32` to `MemoryState` and to `new()`. Backward-compatible due to serde default.
- `src/field.rs` (~+30 LOC): `pub fn reconstruction_error(&self, embedding: &[f32]) -> f32` wrapping `sparse_encoder.read().encode` + `decode` + L2 norm.

**Test/bench**: unit test on `reconstruction_error` — encoding a known embedding gives error < 1.0; encoding a zero vector gives well-defined output. FFI roundtrip test under `tests/ffi_smoke.rs`.

**Est LOC**: ~300.

### Phase 2 — Asymmetric transitions, surprise plasticity, triplet reverse_weight
**Modify**
- `src/organ/prototype.rs`: split `strengthen_transition` into forward `delta` and `reverse_delta = delta * REVERSE_RATIO` (0.3), remove the symmetric write. Add `adapt_vigilance(&mut self, pred_error: f32)` method; move `vigilance` from constant to tunable.
- `src/organ/triplet.rs`: add `#[serde(default)] pub reverse_weight: f32` to `TripletEntry` (line 6), update `add`/`replay_add`/`insert_with_id` (lines 42-134) — 9 call sites. `serde(default)` loads old snapshots with `reverse_weight = 0.0`; a migration helper lazily back-fills `weight * 0.3` on access.
- `src/learner/plasticity.rs`: extend `record_access` to take optional `surprise: f32`; high surprise multiplies returned `decay_rate` by a damping factor (slower decay). Keep API back-compat via overload `record_access_with_surprise`.
- `src/ffi.rs`: promote `cf_memory_surprise` write path (`cf_set_memory_surprise`) and `cf_adapt_vigilance` from stub to real.
- `src/store.rs`: in `put_memory` (`store.rs:106`) compute surprise = `field.reconstruction_error(embedding)` and store into `MemoryState.surprise`; pass to plasticity learner.

**Est LOC**: ~450. **Test**: prototype transition `(A→B).strengthen(0.1)` yields `transition(A,B)=0.1`, `transition(B,A)=0.03`. Snapshot-load test for old triplets (no `reverse_weight` key) — must deserialize.

### Phase 3 — Attractor settle + Hopfield module + encoder rewrite
**Create**
- `src/organ/hopfield.rs` (~250 LOC): `HopfieldIndex` holding `HashMap<(MemoryId, MemoryId), f32>` asymmetric couplings, plus `co_activation_count`. API: `record_co_retrieval(&[MemoryId], weight)`, `settle(seed: &[MemoryId], steps: usize) -> Vec<(MemoryId, f32)>`, `stats() -> HopfieldStats`. Couplings are discovered lazily from `CorticalIndex.mem_codes` overlap (sparse dot) at settle time — avoids O(N²) storage.

**Modify**
- `src/organ/cortex.rs`: `pub fn attractor_settle(&self, query: &SparseCode, steps: usize) -> SparseCode` — iteratively blends query with nearest prototype centroid, then follows top asymmetric transitions; 3-5 steps hard-capped.
- `src/organ/cortex.rs::SparseEncoder::update`: replace Oja with FEP rule: `Δw = LR * (x - decode(code)_slice) * act − λ * w`, then Gram-Schmidt partial decorrelation across co-active atom pairs (1% projection removal, `K_ACTIVE² = 4096` pairs per step — manageable).
- `src/learner/route.rs`: add `Route::Attractor` variant.
- `src/ffi.rs`: lift `cf_search_attractor`, `cf_hopfield_co_retrieval`, `cf_hopfield_stats` to real impls. Hopfield must live behind `RwLock` on `ChittaField` next to `cortical_idx` (`field.rs:112`).

**Est LOC**: ~800. **Bench**: `settle()` on 10k memories with 5-hop traversal under 50 ms; encoder update throughput ≥ current Oja impl within 2×.

### Phase 4 — Free-energy merge + integration polish
**Modify**
- Dedup path in `store.rs` `find_dup_pairs`: for candidate pair, compute `merged_code = centroid blend`, then `ΔF = reconstruction_error(A)+reconstruction_error(B) − reconstruction_error(merged) − λ * complexity`. Merge iff `ΔF > 0`; cosine threshold remains fallback gate.
- `src/learner/plasticity.rs`: wire Hopfield co-retrieval into recall-effects drain path (`store.rs:497 drain_pending_recall_effects`).
- Docs + restore CHANGELOG section in `cc-soul`.

**Est LOC**: ~250. **Test**: deterministic dup-pair fixture where cosine ≥ threshold but free-energy delta is negative → pair rejected.

---

## 4. Risk List

1. **Triplet backward compatibility.** `TripletEntry` (`organ/triplet.rs:6-16`) has **no `#[serde(default)]` on any existing field**. The log replay (`replay_add`) deserializes old entries. Adding `reverse_weight` with `#[serde(default)]` is safe, but we cannot retroactively decorate old fields, so a snapshot schema bump + migration marker in `snapshot.rs` may be needed.
2. **Encoder rewrite regression.** Replacing Oja with FEP+Gram-Schmidt (`cortex.rs:158-182`) changes the atom basis; **every indexed memory's sparse code becomes stale**. We need either (a) a re-encode sweep on first boot after upgrade (expensive) or (b) a versioned encoder with both running in parallel during migration.
3. **Hopfield memory blow-up.** Naive pairwise couplings on N=50k memories = 2.5B entries. The lazy sparse-dot-on-demand design mitigates this but makes `settle()` latency sensitive to `mem_codes` lookups. Needs bench early.
4. **Adapt_vigilance feedback instability.** Lowering vigilance creates more prototypes; more prototypes changes the prediction-error distribution; feedback loop can oscillate. Needs EMA smoothing and min/max clamps.
5. **Asymmetric transition semantics.** Current code writes symmetric to both keys (`prototype.rs:176-177`). Call sites that read `transition(a,b)` (e.g., `cortex.rs:403`) implicitly assume symmetry. Breaking this may alter recall ordering in existing snapshots → needs a snapshot-version check.
6. **Lock discipline.** Hopfield co-retrieval mutates during recall, but recall holds `cortical_idx.read()`. Moving Hopfield behind a third RwLock adds a fourth lock to the field. Deadlock ordering must be documented.

---

## 5. Recommendation

**Build Phase 1 only. Defer Phases 2-4; reassess after real usage signal.**

Rationale:
- Phase 1 alone (~300 LOC) unblocks the MCP daemon wrappers that cc-soul already references and removes a public retraction from the changelog. It delivers *real* FEP-flavored information (reconstruction error is a genuine, cheap free-energy proxy given the existing encoder).
- Phases 2-4 total ~1500 LOC of new code on top of a snapshot-schema migration, an encoder rewrite that invalidates every existing sparse code, and a new async-correct organ (Hopfield). For a single maintainer, the payoff is unclear.
- The Spisak & Friston Neurocomputing 2026 citation should be verified before committing to the full stack.
- Honest posture: ship Phase 1, then **move the rest of the retracted items to a GitHub issue labeled "research exploration"** with this plan as the body. Revisit only if Phase-1 usage data justifies it. Otherwise abandon.

---

## Top 3 Risks (summary)

1. **Snapshot backward compatibility** on `TripletEntry.reverse_weight` and `MemoryState.surprise` — Phase-2 migration risk.
2. **Encoder rewrite invalidates all existing sparse codes**, forcing a full re-encode sweep on upgrade — Phase-3 silent cost.
3. **Adapt-vigilance oscillation + asymmetric-transition semantics change** breaking recall ordering on existing deployments — Phase-2/3 correctness risk.

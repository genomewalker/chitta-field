# LEARNER_AUDIT

## Caveat — do not believe without verification

This audit is derived from primary-source grep/read of `chitta-field/src/` on 2026-04-16. A prior review contained fabrications; every claim below cites `file:line`. Re-run `/usr/bin/grep -rn <symbol> src/` to independently confirm before acting.

## Status table

| Learner | Type(s) | Status | Evidence (file:line) |
|---|---|---|---|
| `bandit.rs` | `BetaPrior`, `GaussianPrior` | WIRED (transitively) | Used as primitives inside `route.rs`, `context.rs`, `domain_reliability.rs`. Not referenced directly outside `src/learner/`. |
| `context.rs` | `ContextLearner` | REGISTERED-BUT-UNUSED on the live path | Instantiated in `LearnerSet::new` (`learner/mod.rs:26`), held in `field.rs:110,420`. Called only from advisory wrappers `Field::recommended_window` (`store.rs:1256-1260`) and `record_context_outcome`-ish path at `store.rs:1265-1268`, reached only from FFI `cf_recommended_window` (`ffi.rs:868-879`). No recall or write path in `store.rs` calls it — window sizing inside `recall_*` is hard-coded. |
| `domain_reliability.rs` | `DomainReliability` | WIRED (live write + recall) | Write: correction penalty at `store.rs:224-228` (inside `put_memory` when `kind == "correction"`). Recall: `poe_mul` multiplier applied per-hit at `store.rs:584`. Success reinforcement (`record_success`) is **never called** anywhere (grep: zero non-test hits). |
| `plasticity.rs` | `PlasticityLearner` | WIRED (live read path only) | `record_access` called inside `Field::get_memory` at `store.rs:249-253`; returned rate applied as `decay_rate` delta at `store.rs:256-266`. Not called from recall ranking (`recall_*`) — only from single-memory `get_memory`. |
| `route.rs` | `RouteLearner`, `Route`, `QueryIntent` | REGISTERED-BUT-ADVISORY (not wired to dispatch) | `select_route` (`store.rs:1272-1275`) and `feedback` (`store.rs:1250-1251`) are exposed via FFI (`cf_select_route` `ffi.rs:650`, `cf_route_feedback` `ffi.rs:671`, `cf_feedback` `ffi.rs:858`). **The returned `Route` is not consumed inside Rust recall** — internal recall paths do not branch on it; the C caller is expected to pick a channel. `best_route` has no non-test caller. |

## Orphaned / partially-wired wiring sketches

### 1. `ContextLearner` — wire to recall window sizing

Currently `recommended_window` is FFI-only. Live recall functions (`Field::recall_*` near `store.rs:570+`) use fixed `k`. Proposed wiring inside the recall entry point:

```rust
pub fn recall(&self, query: &str, session_type: Option<&str>, k: Option<usize>) -> Result<Vec<RecallHit>> {
    let effective_k = k.unwrap_or_else(|| {
        self.learners.read().context
            .recommended_window(session_type.unwrap_or("general"))
    });
    let hits = self.recall_semantic(query, effective_k)?;
    // After user feedback (or implicit: hits.len() > 0 * mean_score):
    // self.learners.write().context.record_outcome(session_type, effective_k, quality);
    Ok(hits)
}
```

State needed: a `session_type` argument threaded through `recall` (currently absent). Outcome feedback requires either piggy-backing on `cf_feedback` or a new `cf_recall_outcome`.

### 2. `RouteLearner` — wire to in-Rust dispatch

Today the caller picks a channel based on the returned `Route` integer. To close the loop inside the store:

```rust
pub fn recall(&self, query: &str, k: usize) -> Result<Vec<RecallHit>> {
    let (episode_id, route) = self.select_route(query);
    let hits = match route {
        Route::Semantic => self.recall_semantic(query, k)?,
        Route::Keyword  => self.recall_keyword(query, k)?,
        Route::Temporal => self.recall_temporal(query, k)?,
        Route::Artifact => self.recall_artifact(query, k)?,
        Route::Hybrid   => self.recall_hybrid(query, k)?,
        Route::Full     => self.recall_full(query, k)?,
    };
    // Implicit reward = mean top-k score, or defer to cf_feedback(episode_id, ...).
    Ok(hits)
}
```

State needed: confirm the `recall_{keyword,temporal,artifact,hybrid,full}` helpers exist in `store.rs`. Episode id should be returned alongside hits so external feedback can still update the arm.

### 3. `DomainReliability::record_success` — close the reinforcement loop

Penalty side is wired at `store.rs:224`. Success side is dead (zero callers). Minimal wiring inside `Field::feedback` (`store.rs:1250`):

```rust
pub fn feedback(&self, episode_id: u64, reward: f32) -> Result<()> {
    self.learners.write().route.feedback(episode_id, reward);
    if reward >= 0.5 {
        if let Some(realms) = self.last_episode_realms.read().get(&episode_id) {
            let mut dr = self.learners.write();
            for realm in realms { dr.domain_reliability.record_success(realm); }
        }
    }
    Ok(())
}
```

State needed: a new field `last_episode_realms: RwLock<HashMap<u64, Vec<String>>>` populated at recall time.

### 4. `PlasticityLearner` — extend to recall scoring

Currently only single-memory `get_memory` touches it. To reflect recall-time usage, call `record_access` for each returned hit inside the recall tail (near `store.rs:611-614`, where `hit_ids` is already collected). A one-liner inside `enqueue_recall_effects` would suffice — no new state required.

## Summary

- **`ContextLearner`** is instantiated and reachable only through FFI-exposed advisory getters; the recall path never consults it.
- **`RouteLearner`** is half-wired: `select_route`/`feedback` exist but the `Route` it returns is never used to dispatch recall inside Rust; dispatch is delegated to the C caller.
- **`DomainReliability::record_success`** has zero callers anywhere in the tree — only the penalty half of the PoE feedback loop is live; reliability can only decay.

Fully wired: `DomainReliability::record_correction`+`reliability`, `PlasticityLearner::record_access` on `get_memory`, and the `bandit.rs` primitives via composition.

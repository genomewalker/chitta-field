//! Analogical retrieval over the triplet lane (Vector Symbolic Architecture).
//!
//! Built on hdc.rs's 8192-bit binary vectors: XOR binding (self-inverse, so
//! unbind == bind) and majority bundling. Two modes embedding similarity cannot
//! express, because both are about *structure*, not surface content:
//!
//! - `proportional`  a:b :: c:?  — Kanerva record mapping. The mapping vector is
//!   `record(a) ⊛ record(c)`; applying it to `atom(b)` estimates the filler that
//!   plays b's role for c. Raw atoms alone cannot do this: `word_hv` is
//!   hash-random, so `atom(a) ⊛ atom(b) ⊛ atom(c)` is white noise — the relation
//!   has to come from the fillers each entity is *bound to* in the store.
//!
//! - `structural`  — per-memory signatures that are filler-INVARIANT: an entity
//!   contributes its Weisfeiler-Lehman role code (the multiset of predicate
//!   slots it fills), never its name. Two memories describing the same shape
//!   with different entities in different realms therefore match, which is the
//!   whole cross-realm point.
//!
//! Nothing here touches a lock: every entry point takes an owned `Fact` slice
//! that the caller copied out from under the triplet-store read guard.
//!
//! Cost discipline (learned the hard way — the first version wedged the live
//! daemon for six minutes while holding the RPC mutex):
//!   * every string is normalised and interned ONCE, into `FactTable`;
//!   * no scan is nested inside another scan — proportional is O(facts + vocab);
//!   * bundling picks its terms before it materialises any hypervector, so a
//!     hub entity cannot balloon into hundreds of megabytes of throwaway 1 KiB
//!     vectors.

use crate::hdc::{bind, bundle, similarity, word_hv, HdcVec};
use crate::ids::MemoryId;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

/// Terms past this cap are dropped from a bundle.
/// ceiling: a member's similarity to a majority bundle decays as 1/sqrt(n), so
/// past ~64 terms the recovered signal sinks toward the 0.5 noise floor and the
/// ranking becomes guesswork; upgrade: select terms by predicate overlap with
/// the probe instead of by sort order.
const MAX_BUNDLE_TERMS: usize = 64;

/// Lock-free copy of one live triplet. Analogy scoring runs on these and never
/// under the triplet-store read guard — parking_lot is writer-preferring and a
/// scoring pass held there starves every queued writer.
#[derive(Clone, Debug)]
pub struct Fact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub memory_id: Option<MemoryId>,
}

fn norm(token: &str) -> String {
    token.trim().to_lowercase()
}

/// Atomic symbol for a token. Entity and predicate strings are treated as whole
/// symbols (not bag-of-words) so distinct names stay quasi-orthogonal — a
/// prerequisite for the mapping algebra below.
pub fn atom(token: &str) -> HdcVec {
    word_hv(&norm(token))
}

#[inline] fn role_fwd()  -> HdcVec { word_hv("VSA:ROLE:FWD") }
#[inline] fn role_rev()  -> HdcVec { word_hv("VSA:ROLE:REV") }
#[inline] fn role_subj() -> HdcVec { word_hv("VSA:ROLE:SUBJ") }
#[inline] fn role_pred() -> HdcVec { word_hv("VSA:ROLE:PRED") }
#[inline] fn role_obj()  -> HdcVec { word_hv("VSA:ROLE:OBJ") }

fn xor3(a: &HdcVec, b: &HdcVec, c: &HdcVec) -> HdcVec {
    std::array::from_fn(|i| a[i] ^ b[i] ^ c[i])
}

/// Bundle at most `MAX_BUNDLE_TERMS` terms in a caller-independent order: sort
/// by the string key, cap, and only THEN build the hypervectors. Sorting is what
/// makes two rebuilds bit-identical; deferring `make` is what bounds memory —
/// an entity with 100k facts would otherwise allocate 100k × 1 KiB of vectors
/// just to discard all but 64 of them.
fn bundle_top<F>(mut keyed: Vec<(String, usize)>, make: F) -> Option<HdcVec>
where
    F: Fn(usize) -> HdcVec,
{
    if keyed.is_empty() {
        return None;
    }
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    keyed.truncate(MAX_BUNDLE_TERMS);
    let hvs: Vec<HdcVec> = keyed.iter().map(|(_, i)| make(*i)).collect();
    Some(bundle(&hvs))
}

// ── Interned fact table ───────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct TFact {
    s: u32,
    p: u32,
    o: u32,
    mem: Option<MemoryId>,
}

/// Normalised, interned view of a fact slice.
///
/// `norm` allocates a `String` on every call. The first version of
/// `proportional` called it inside a scan nested per candidate — O(candidates ×
/// facts) allocations — which on the live store ran for over six minutes while
/// holding the daemon's RPC mutex, blocking every queued write. Interning once
/// turns every subsequent comparison into a `u32` equality.
#[derive(Default)]
pub struct FactTable {
    vocab: Vec<String>,
    ids: HashMap<String, u32>,
    facts: Vec<TFact>,
}

impl FactTable {
    pub fn build(facts: &[Fact]) -> FactTable {
        let mut t = FactTable {
            vocab: Vec::new(),
            ids: HashMap::new(),
            facts: Vec::with_capacity(facts.len()),
        };
        for f in facts {
            let s = t.intern(&f.subject);
            let p = t.intern(&f.predicate);
            let o = t.intern(&f.object);
            t.facts.push(TFact { s, p, o, mem: f.memory_id });
        }
        t
    }

    fn intern(&mut self, token: &str) -> u32 {
        let key = norm(token);
        if let Some(&id) = self.ids.get(&key) {
            return id;
        }
        let id = self.vocab.len() as u32;
        self.vocab.push(key.clone());
        self.ids.insert(key, id);
        id
    }

    fn entity(&self, name: &str) -> Option<u32> {
        self.ids.get(&norm(name)).copied()
    }

    fn text(&self, id: u32) -> &str {
        &self.vocab[id as usize]
    }

    pub fn len(&self) -> usize {
        self.facts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    /// Distinct interned symbols (entities and predicates share one space, as
    /// `atom` does not distinguish them).
    pub fn vocab_len(&self) -> usize {
        self.vocab.len()
    }
}

// ── Proportional analogy ──────────────────────────────────────────────────────

/// Kanerva record for one entity: a bundle of `(relation-slot ⊛ partner)` over
/// every fact the entity participates in. Direction is part of the slot, so
/// `x uses y` and `y uses x` bind into different roles.
fn entity_record(t: &FactTable, e: u32) -> Option<HdcVec> {
    let mut keyed: Vec<(String, usize)> = Vec::new();
    for (i, f) in t.facts.iter().enumerate() {
        if f.s == e {
            keyed.push((format!("f\u{1}{}\u{1}{}", t.text(f.p), t.text(f.o)), i));
        } else if f.o == e {
            keyed.push((format!("r\u{1}{}\u{1}{}", t.text(f.p), t.text(f.s)), i));
        }
    }
    bundle_top(keyed, |i| {
        let f = t.facts[i];
        if f.s == e {
            bind(&bind(&role_fwd(), &word_hv(t.text(f.p))), &word_hv(t.text(f.o)))
        } else {
            bind(&bind(&role_rev(), &word_hv(t.text(f.p))), &word_hv(t.text(f.s)))
        }
    })
}

/// One ranked answer to `a:b :: c:?`.
#[derive(Clone, Debug)]
pub struct ProportionalHit {
    pub answer: String,
    pub score: f32,
    /// Fact in the store that links `c` to `answer`, when one exists — the
    /// memory that can be credited for the answer.
    pub predicate: String,
    pub memory_id: Option<MemoryId>,
}

/// Solve `a:b :: c:?` over a prepared table. Returns candidate fillers ranked by
/// similarity to the estimate `record(a) ⊛ record(c) ⊛ atom(b)`.
///
/// Empty when either `a` or `c` has no facts: with no bound fillers there is no
/// relation to transfer, and a hash-random guess would be worse than silence.
///
/// Cost is O(facts + candidates), with two passes over the table and no nested
/// scan. Candidates are restricted to entities that fill at least one of `c`'s
/// relations — an entity sharing no predicate with `c` is hash-orthogonal to the
/// estimate and can only score at the 0.5 noise floor, so scoring it is wasted
/// work as well as noise in the ranking.
pub fn proportional_on(
    t: &FactTable,
    a: &str,
    b: &str,
    c: &str,
    limit: usize,
) -> Vec<ProportionalHit> {
    let (ea, ec) = match (t.entity(a), t.entity(c)) {
        (Some(x), Some(y)) => (x, y),
        _ => return Vec::new(),
    };
    let (rec_a, rec_c) = match (entity_record(t, ea), entity_record(t, ec)) {
        (Some(ra), Some(rc)) => (ra, rc),
        _ => return Vec::new(),
    };
    let estimate = xor3(&rec_a, &rec_c, &atom(b));

    // Pass 1: c's predicate set, and the fact linking each entity to c.
    // The live fact order is HashMap order, so "first match wins" would have let
    // the credited memory change between restarts; pick the smallest
    // (predicate, memory) instead so the citation is stable.
    let mut c_preds: HashSet<u32> = HashSet::new();
    let mut linked: HashMap<u32, (u32, Option<MemoryId>)> = HashMap::new();
    for f in &t.facts {
        let partner = if f.s == ec {
            Some(f.o)
        } else if f.o == ec {
            Some(f.s)
        } else {
            None
        };
        if let Some(other) = partner {
            c_preds.insert(f.p);
            let cand = (f.p, f.mem);
            linked
                .entry(other)
                .and_modify(|cur| {
                    if (t.text(cand.0), cand.1) < (t.text(cur.0), cur.1) {
                        *cur = cand;
                    }
                })
                .or_insert(cand);
        }
    }

    // Pass 2: candidate fillers.
    let eb = t.entity(b);
    let mut candidates: HashSet<u32> = HashSet::new();
    for f in &t.facts {
        if !c_preds.contains(&f.p) {
            continue;
        }
        for side in [f.s, f.o] {
            if side != ea && side != ec && Some(side) != eb {
                candidates.insert(side);
            }
        }
    }

    let mut hits: Vec<ProportionalHit> = candidates
        .into_iter()
        .map(|id| {
            let name = t.text(id);
            let link = linked.get(&id);
            ProportionalHit {
                answer: name.to_string(),
                score: similarity(&estimate, &word_hv(name)),
                predicate: link.map(|(p, _)| t.text(*p).to_string()).unwrap_or_default(),
                memory_id: link.and_then(|(_, m)| *m),
            }
        })
        .collect();

    hits.sort_by(|x, y| {
        y.score
            .partial_cmp(&x.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| x.answer.cmp(&y.answer))
    });
    hits.truncate(limit);
    hits
}

/// Convenience wrapper that builds a throwaway table. Callers that already hold
/// an `AnalogyIndex` should use `AnalogyIndex::proportional`, which reuses the
/// cached table instead of re-interning the whole store per call.
pub fn proportional(facts: &[Fact], a: &str, b: &str, c: &str, limit: usize) -> Vec<ProportionalHit> {
    proportional_on(&FactTable::build(facts), a, b, c, limit)
}

// ── Structural signatures ─────────────────────────────────────────────────────

/// One round of Weisfeiler-Lehman colour refinement over a fact set: an entity's
/// code is the sorted multiset of the `(predicate, direction)` slots it fills,
/// never its name. Isomorphic graphs get identical codes, which is what lets a
/// pattern in one realm match the same pattern in another.
///
/// ceiling: a single WL round — two graphs that differ only past the 1-hop role
/// structure collide; upgrade: iterate the labels to a fixpoint.
fn role_codes(facts: &[Fact]) -> HashMap<String, HdcVec> {
    let mut slots: HashMap<String, Vec<String>> = HashMap::new();
    for f in facts {
        slots.entry(norm(&f.subject)).or_default().push(format!("{}>", norm(&f.predicate)));
        slots.entry(norm(&f.object)).or_default().push(format!(">{}", norm(&f.predicate)));
    }
    slots
        .into_iter()
        .map(|(entity, mut labels)| {
            labels.sort();
            (entity, word_hv(&format!("VSA:VAR:{}", labels.join("|"))))
        })
        .collect()
}

/// Filler-invariant signature of a fact set: bundle over facts of
/// `(R_subj ⊛ var(s)) ⊛ (R_pred ⊛ atom(p)) ⊛ (R_obj ⊛ var(o))`.
pub fn signature(facts: &[Fact]) -> Option<HdcVec> {
    if facts.is_empty() {
        return None;
    }
    let codes = role_codes(facts);
    let var = |name: &str| codes.get(&norm(name)).copied().unwrap_or_else(|| atom(name));
    let keyed: Vec<(String, usize)> = facts
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let key = format!("{}\u{1}{}\u{1}{}", norm(&f.subject), norm(&f.predicate), norm(&f.object));
            (key, i)
        })
        .collect();
    bundle_top(keyed, |i| {
        let f = &facts[i];
        xor3(
            &bind(&role_subj(), &var(&f.subject)),
            &bind(&role_pred(), &atom(&f.predicate)),
            &bind(&role_obj(), &var(&f.object)),
        )
    })
}

/// Facts anchored on the entities a free-text probe mentions, so `structural`
/// works without a memory id. Token-level overlap, since entity names are often
/// multi-word.
pub fn facts_for_text(facts: &[Fact], text: &str) -> Vec<Fact> {
    let probe: HashSet<String> = text
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|t| t.len() >= 3)
        .map(|t| t.to_lowercase())
        .collect();
    if probe.is_empty() {
        return Vec::new();
    }
    let mentions = |name: &str| {
        name.split(|ch: char| !ch.is_alphanumeric())
            .any(|t| t.len() >= 3 && probe.contains(&t.to_lowercase()))
    };
    facts
        .iter()
        .filter(|f| mentions(&f.subject) || mentions(&f.object))
        .cloned()
        .collect()
}

/// memory_id → structural signature, for every memory that sourced ≥1 triplet,
/// plus the interned table both modes scan.
///
/// Not persisted: rebuilt from the triplet lane on first use and whenever the
/// live triplet count moves.
///
/// ceiling: staleness is keyed on triplet count alone, so an in-place
/// invalidate-and-replace that leaves the count unchanged is missed until the
/// next count move; upgrade: carry a mutation counter like HdcStore's.
/// The two halves are built independently and on demand. Interning the table is
/// linear and cheap; the signature map costs one 8192-bit majority bundle per
/// source memory, seconds and ~1 KiB per memory on a six-figure store. Making
/// `proportional` pay for signatures it never reads would just move the stall
/// the caller already hit.
#[derive(Default)]
pub struct AnalogyIndex {
    sigs: HashMap<MemoryId, HdcVec>,
    table: FactTable,
    table_from: usize,
    table_built: bool,
    sigs_from: usize,
    sigs_built: bool,
}

impl AnalogyIndex {
    /// Staleness of the signature map (structural mode).
    pub fn is_stale(&self, triplet_count: usize) -> bool {
        !self.sigs_built || self.sigs_from != triplet_count
    }

    /// Staleness of the interned table (both modes).
    pub fn is_table_stale(&self, triplet_count: usize) -> bool {
        !self.table_built || self.table_from != triplet_count
    }

    /// Intern the facts. Linear, no hypervector work.
    pub fn rebuild_table(&mut self, facts: &[Fact], triplet_count: usize) {
        self.table = FactTable::build(facts);
        self.table_from = triplet_count;
        self.table_built = true;
    }

    /// Build one structural signature per source memory. The expensive half.
    pub fn rebuild_signatures(&mut self, facts: &[Fact], triplet_count: usize) {
        let mut by_memory: HashMap<MemoryId, Vec<Fact>> = HashMap::new();
        for f in facts {
            if let Some(mid) = f.memory_id {
                by_memory.entry(mid).or_default().push(f.clone());
            }
        }
        self.sigs = by_memory
            .into_iter()
            .filter_map(|(mid, group)| signature(&group).map(|hv| (mid, hv)))
            .collect();
        self.sigs_from = triplet_count;
        self.sigs_built = true;
    }

    pub fn rebuild(&mut self, facts: &[Fact], triplet_count: usize) {
        self.rebuild_table(facts, triplet_count);
        self.rebuild_signatures(facts, triplet_count);
    }

    pub fn len(&self) -> usize {
        self.sigs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sigs.is_empty()
    }

    pub fn table(&self) -> &FactTable {
        &self.table
    }

    /// `a:b :: c:?` against the cached table — no re-interning per call.
    pub fn proportional(&self, a: &str, b: &str, c: &str, limit: usize) -> Vec<ProportionalHit> {
        proportional_on(&self.table, a, b, c, limit)
    }

    pub fn signature_of(&self, id: MemoryId) -> Option<HdcVec> {
        self.sigs.get(&id).copied()
    }

    /// Rank indexed memories by signature similarity to `probe`.
    pub fn rank(&self, probe: &HdcVec, exclude: Option<MemoryId>, limit: usize) -> Vec<(MemoryId, f32)> {
        let mut scored: Vec<(MemoryId, f32)> = self
            .sigs
            .iter()
            .filter(|(id, _)| Some(**id) != exclude)
            .map(|(&id, hv)| (id, similarity(probe, hv)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(limit);
        scored
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fact(s: &str, p: &str, o: &str, mem: u64) -> Fact {
        Fact { subject: s.into(), predicate: p.into(), object: o.into(), memory_id: Some(mem) }
    }

    /// A fact with no source memory — the shape production actually writes, via
    /// add_triplet's 3-argument form.
    fn orphan(s: &str, p: &str, o: &str) -> Fact {
        Fact { subject: s.into(), predicate: p.into(), object: o.into(), memory_id: None }
    }

    /// paris:france :: tokyo:? over a 20-triplet store with distractor relations.
    fn geo_store() -> Vec<Fact> {
        vec![
            fact("paris", "capital_of", "france", 1),
            fact("tokyo", "capital_of", "japan", 2),
            fact("rome", "capital_of", "italy", 3),
            fact("madrid", "capital_of", "spain", 4),
            fact("berlin", "capital_of", "germany", 5),
            fact("paris", "located_in", "europe", 6),
            fact("tokyo", "located_in", "asia", 7),
            fact("rome", "located_in", "europe", 8),
            fact("madrid", "located_in", "europe", 9),
            fact("berlin", "located_in", "europe", 10),
            fact("france", "currency", "euro", 11),
            fact("japan", "currency", "yen", 12),
            fact("italy", "currency", "euro", 13),
            fact("spain", "currency", "euro", 14),
            fact("germany", "currency", "euro", 15),
            fact("paris", "population", "2m", 16),
            fact("tokyo", "population", "14m", 17),
            fact("rome", "population", "3m", 18),
            fact("madrid", "population", "3m", 19),
            fact("berlin", "population", "4m", 20),
        ]
    }

    #[test]
    fn binding_roundtrip_recovers_filler_above_noise() {
        // A record bundling four bound (role, filler) pairs; unbinding the role
        // must bring the right filler back well clear of the 0.5 noise floor.
        let facts = vec![
            fact("chitta", "uses", "duckdb", 1),
            fact("chitta", "has", "memory", 1),
            fact("chitta", "written_in", "rust", 1),
            fact("chitta", "runs_on", "linux", 1),
        ];
        let table = FactTable::build(&facts);
        let rec = entity_record(&table, table.entity("chitta").unwrap()).expect("record");
        let probe = bind(&rec, &bind(&role_fwd(), &atom("uses")));
        let hit = similarity(&probe, &atom("duckdb"));
        for distractor in ["memory", "rust", "linux", "postgres"] {
            let miss = similarity(&probe, &atom(distractor));
            assert!(hit > miss + 0.05, "unbind lost the filler: duckdb={hit} {distractor}={miss}");
        }
        assert!(hit > 0.55, "filler similarity {hit} not above noise floor");
    }

    #[test]
    fn proportional_analogy_capital_country() {
        let facts = geo_store();
        let hits = proportional(&facts, "paris", "france", "tokyo", 3);
        assert!(!hits.is_empty(), "no candidates ranked");
        assert_eq!(hits[0].answer, "japan", "ranked: {:?}", hits.iter().map(|h| (&h.answer, h.score)).collect::<Vec<_>>());
        assert_eq!(hits[0].predicate, "capital_of");
        assert_eq!(hits[0].memory_id, Some(2));
    }

    #[test]
    fn proportional_analogy_follows_the_probed_role() {
        // Same pair of entities, different role probed: paris:europe :: tokyo:asia.
        let facts = geo_store();
        let hits = proportional(&facts, "paris", "europe", "tokyo", 3);
        assert_eq!(hits[0].answer, "asia", "ranked: {:?}", hits.iter().map(|h| (&h.answer, h.score)).collect::<Vec<_>>());
    }

    #[test]
    fn proportional_is_empty_without_grounding() {
        let facts = geo_store();
        assert!(proportional(&facts, "nowhere", "france", "tokyo", 3).is_empty());
    }

    /// Regression: the live store's triplets carry no source_memory_id, so the
    /// signature index is empty while the fact table is NOT. The old code took
    /// that path into an O(candidates × facts) scan that ran for minutes under
    /// the RPC mutex. Proportional must still answer, and must answer fast.
    #[test]
    fn proportional_works_when_no_fact_has_a_source_memory() {
        let facts: Vec<Fact> = geo_store()
            .into_iter()
            .map(|f| orphan(&f.subject, &f.predicate, &f.object))
            .collect();

        let mut idx = AnalogyIndex::default();
        idx.rebuild(&facts, facts.len());
        assert_eq!(idx.len(), 0, "no fact has a memory, so no signature can be built");
        assert_eq!(idx.table().len(), facts.len(), "the fact table must still be populated");

        let hits = idx.proportional("paris", "france", "tokyo", 3);
        assert_eq!(hits[0].answer, "japan", "ranked: {:?}", hits.iter().map(|h| &h.answer).collect::<Vec<_>>());
        assert_eq!(hits[0].memory_id, None, "an orphan fact credits no memory");
        assert_eq!(hits[0].predicate, "capital_of", "the linking predicate survives without a memory id");
    }

    /// The work bound. A store this size takes the old nested scan into the
    /// billions of string allocations; linear passes finish in well under a
    /// second. The assertion is on the wall clock deliberately — the bug this
    /// guards was a timeout, not a wrong answer.
    #[test]
    fn proportional_is_linear_on_a_large_store() {
        let mut facts: Vec<Fact> = Vec::new();
        for i in 0..20_000u64 {
            facts.push(orphan(&format!("city{i}"), "capital_of", &format!("country{i}")));
            facts.push(orphan(&format!("city{i}"), "located_in", &format!("region{}", i % 50)));
        }
        facts.push(orphan("paris", "capital_of", "france"));
        facts.push(orphan("tokyo", "capital_of", "japan"));
        facts.push(orphan("paris", "located_in", "region1"));
        facts.push(orphan("tokyo", "located_in", "region1"));

        let mut idx = AnalogyIndex::default();
        idx.rebuild(&facts, facts.len());
        assert!(idx.table().vocab_len() > 40_000, "expected a large vocabulary");

        let started = std::time::Instant::now();
        let hits = idx.proportional("paris", "france", "tokyo", 5);
        let elapsed = started.elapsed();

        assert_eq!(hits[0].answer, "japan", "ranked: {:?}", hits.iter().map(|h| &h.answer).collect::<Vec<_>>());
        assert!(
            elapsed < std::time::Duration::from_secs(10),
            "proportional took {elapsed:?} on 40k facts — the nested scan is back"
        );
    }

    /// Candidates are confined to entities filling one of c's relations, so an
    /// unrelated lane of the store cannot enlarge the scoring set.
    #[test]
    fn candidates_are_confined_to_the_probed_relations() {
        let mut facts = geo_store();
        for i in 0..50u64 {
            facts.push(orphan(&format!("pkg{i}"), "depends_on", &format!("lib{i}")));
        }
        // tokyo has no `depends_on` fact, so no pkg/lib may be ranked.
        let hits = proportional(&facts, "paris", "france", "tokyo", 100);
        assert!(
            hits.iter().all(|h| !h.answer.starts_with("pkg") && !h.answer.starts_with("lib")),
            "unrelated lane leaked into candidates: {:?}",
            hits.iter().map(|h| &h.answer).collect::<Vec<_>>()
        );
        assert_eq!(hits[0].answer, "japan");
    }

    #[test]
    fn structural_matches_isomorphic_pattern_across_realms() {
        // Memory 1 (realm alpha) and memory 2 (realm beta) share a shape with no
        // shared entity names; ten distractors have different shapes.
        let mut facts = vec![
            fact("chittad", "depends_on", "blas", 1),
            fact("blas", "provided_by", "openblas", 1),
            fact("chittad", "built_by", "cmake", 1),
            fact("snakemake", "depends_on", "conda", 2),
            fact("conda", "provided_by", "miniforge", 2),
            fact("snakemake", "built_by", "pip", 2),
        ];
        let shapes: [(&str, &str, &str); 10] = [
            ("a1", "wrote", "a2"),
            ("b1", "reviewed", "b2"),
            ("c1", "deleted", "c2"),
            ("d1", "owns", "d2"),
            ("e1", "reads", "e2"),
            ("f1", "blocks", "f2"),
            ("g1", "extends", "g2"),
            ("h1", "mirrors", "h2"),
            ("i1", "signs", "i2"),
            ("j1", "hosts", "j2"),
        ];
        for (i, (s, p, o)) in shapes.iter().enumerate() {
            let mid = 100 + i as u64;
            facts.push(fact(s, p, o, mid));
            facts.push(fact(o, "tagged", "misc", mid));
        }

        let mut idx = AnalogyIndex::default();
        idx.rebuild(&facts, facts.len());
        assert_eq!(idx.len(), 12);

        let probe = idx.signature_of(1).expect("probe signature");
        let ranked = idx.rank(&probe, Some(1), 3);
        assert_eq!(ranked[0].0, 2, "isomorphic memory not top-1: {ranked:?}");
        assert!(ranked[0].1 > 0.99, "isomorphic signatures should be near-identical: {}", ranked[0].1);
        assert!(ranked[0].1 > ranked[1].1 + 0.3, "structural gap too small: {ranked:?}");
    }

    #[test]
    fn structural_signature_is_filler_invariant_but_shape_sensitive() {
        let a = vec![fact("x", "uses", "y", 1), fact("y", "needs", "z", 1)];
        let renamed = vec![fact("p", "uses", "q", 2), fact("q", "needs", "r", 2)];
        let reshaped = vec![fact("x", "uses", "y", 3), fact("z", "needs", "y", 3)];
        let sa = signature(&a).unwrap();
        assert_eq!(sa, signature(&renamed).unwrap(), "renaming entities changed the signature");
        assert!(
            similarity(&sa, &signature(&reshaped).unwrap()) < 0.9,
            "a different join direction must change the signature"
        );
    }

    #[test]
    fn rebuild_is_deterministic() {
        let mut facts = geo_store();
        let mut a = AnalogyIndex::default();
        a.rebuild(&facts, facts.len());
        // Same store, shuffled input order: signatures must be bit-identical.
        facts.reverse();
        let mut b = AnalogyIndex::default();
        b.rebuild(&facts, facts.len());
        assert_eq!(a.len(), b.len());
        for (id, hv) in &a.sigs {
            assert_eq!(Some(*hv), b.signature_of(*id), "signature drift for memory {id}");
        }
    }

    /// Ranking must not depend on the order the triplet store handed facts over —
    /// that order is HashMap order and changes between restarts.
    #[test]
    fn proportional_is_order_independent() {
        let mut facts = geo_store();
        let forward = proportional(&facts, "paris", "france", "tokyo", 5);
        facts.reverse();
        let reversed = proportional(&facts, "paris", "france", "tokyo", 5);
        let names = |v: &[ProportionalHit]| v.iter().map(|h| h.answer.clone()).collect::<Vec<_>>();
        assert_eq!(names(&forward), names(&reversed), "ranking depends on fact order");
        assert_eq!(forward[0].memory_id, reversed[0].memory_id, "credited memory depends on fact order");
    }

    /// proportional must not pay for the signature map. The table alone has to
    /// be enough to answer, or every proportional call on a live store would
    /// bundle one 8192-bit vector per memory for nothing.
    #[test]
    fn proportional_needs_only_the_table() {
        let facts = geo_store();
        let mut idx = AnalogyIndex::default();
        idx.rebuild_table(&facts, facts.len());
        assert!(!idx.is_table_stale(facts.len()), "table should be fresh");
        assert!(idx.is_stale(facts.len()), "signatures must still be unbuilt");
        assert_eq!(idx.proportional("paris", "france", "tokyo", 3)[0].answer, "japan");
        assert_eq!(idx.len(), 0, "proportional built signatures it does not read");
    }

    /// The structural half is the expensive one: one 8192-bit majority bundle and
    /// ~1 KiB of index per source memory. The live store has six figures of
    /// memories, so anything superlinear here becomes a multi-minute stall with
    /// the RPC mutex held — the exact failure this lane already caused once.
    #[test]
    fn signature_rebuild_cost_is_linear_in_memories() {
        let build = |memories: u64| -> std::time::Duration {
            // The shape production actually writes: subject is the memory id.
            let mut facts = Vec::new();
            for m in 0..memories {
                facts.push(fact(&m.to_string(), "has_flag", "verified", m));
                facts.push(fact(&m.to_string(), "references", &format!("doc{m}"), m));
                facts.push(fact(&m.to_string(), "ingested_from", "transcript", m));
            }
            let mut idx = AnalogyIndex::default();
            let started = std::time::Instant::now();
            idx.rebuild_signatures(&facts, facts.len());
            let elapsed = started.elapsed();
            assert_eq!(idx.len() as u64, memories, "one signature per source memory");
            elapsed
        };
        let small = build(500);
        let large = build(2_000);
        assert!(
            large < small * 12 + std::time::Duration::from_secs(1),
            "signature rebuild is superlinear: 500 memories {small:?}, 2000 memories {large:?}"
        );
    }

    #[test]
    fn index_staleness_tracks_triplet_count() {
        let facts = geo_store();
        let mut idx = AnalogyIndex::default();
        assert!(idx.is_stale(facts.len()));
        idx.rebuild(&facts, facts.len());
        assert!(!idx.is_stale(facts.len()));
        assert!(idx.is_stale(facts.len() + 1));
    }

    #[test]
    fn facts_for_text_anchors_on_mentioned_entities() {
        let facts = geo_store();
        let picked = facts_for_text(&facts, "what is the capital of japan?");
        assert!(!picked.is_empty());
        assert!(picked.iter().all(|f| f.subject == "japan" || f.object == "japan"));
    }

    #[test]
    fn signature_of_empty_fact_set_is_none() {
        assert!(signature(&[]).is_none());
    }

    /// A hub entity must not materialise one hypervector per fact just to keep
    /// 64 of them. This would allocate ~50 MB of throwaway vectors before the fix.
    #[test]
    fn hub_entity_bundling_stays_bounded() {
        let mut facts: Vec<Fact> = Vec::new();
        for i in 0..50_000u64 {
            facts.push(orphan("hub", "links", &format!("leaf{i}")));
        }
        facts.push(orphan("hub", "capital_of", "france"));
        facts.push(orphan("other", "capital_of", "japan"));
        let table = FactTable::build(&facts);
        let started = std::time::Instant::now();
        let rec = entity_record(&table, table.entity("hub").unwrap()).expect("record");
        assert!(started.elapsed() < std::time::Duration::from_secs(10), "hub bundling is unbounded");
        assert_ne!(rec, [0u64; 128], "hub record collapsed to the zero vector");
    }
}

//! Executable Constraint Store — Datalog-like facts with provenance, scope, and branches.
//!
//! References:
//!   - Ramp Labs (2025). Latent Briefing — task-guided memory selection
//!   - Packer et al. (2023). MemGPT — context repository pattern
//!   - GPT-5.4 + Opus brainstorm (2026-04-11) — contradiction lattice + unification recall

use crate::ids::MemoryId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// How a fact was established.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    /// Origin: "user", "tool", "distillation", "inference", "correction"
    pub source: String,
    /// Session where fact was established
    pub session_id: Option<String>,
    /// Basis: "stated", "observed", "derived", "corrected"
    pub confidence_basis: String,
}

impl Default for Provenance {
    fn default() -> Self {
        Self {
            source: "tool".into(),
            session_id: None,
            confidence_basis: "observed".into(),
        }
    }
}

/// A single executable constraint (subject-predicate-object fact with metadata).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    pub id: u64,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
    pub scope: String,
    pub branch_id: u64,
    pub provenance: Provenance,
    pub valid_from_ms: i64,
    pub valid_to_ms: i64,
    pub source_memory_id: Option<MemoryId>,
}

impl Constraint {
    pub fn is_active(&self) -> bool {
        self.valid_to_ms == 0
    }
}

/// Branch status for rival interpretation management.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BranchStatus {
    Active,
    Merged,
    Abandoned,
}

/// A branch represents a rival interpretation of conflicting facts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    pub id: u64,
    pub parent_id: u64,
    pub scope: String,
    pub created_ms: i64,
    pub status: BranchStatus,
}

/// Result of asserting a fact — may include conflict info.
#[derive(Debug, Clone)]
pub struct AssertResult {
    pub fact_id: u64,
    pub conflict: Option<ConflictInfo>,
}

/// Info about a detected conflict during assertion.
#[derive(Debug, Clone)]
pub struct ConflictInfo {
    pub rival_fact_id: u64,
    pub rival_object: String,
    pub new_branch_id: u64,
}

/// Result of explaining a fact — provenance chain + conflicts.
#[derive(Debug, Clone)]
pub struct Explanation {
    pub fact: Constraint,
    pub supporting: Vec<u64>,
    pub conflicting: Vec<u64>,
    pub branch: Option<Branch>,
}

/// The constraint store — indexed fact base with branch management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintStore {
    next_id: u64,
    next_branch_id: u64,
    constraints: Vec<Constraint>,
    id_to_index: HashMap<u64, usize>,
    branches: Vec<Branch>,
    branch_id_to_index: HashMap<u64, usize>,

    // Indexes for fast lookup
    by_subject: HashMap<String, Vec<u64>>,
    by_predicate: HashMap<String, Vec<u64>>,
    by_object: HashMap<String, Vec<u64>>,
    by_scope: HashMap<String, Vec<u64>>,
    by_branch: HashMap<u64, Vec<u64>>,
}

impl ConstraintStore {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            next_branch_id: 1,
            constraints: Vec::new(),
            id_to_index: HashMap::new(),
            branches: Vec::new(),
            branch_id_to_index: HashMap::new(),
            by_subject: HashMap::new(),
            by_predicate: HashMap::new(),
            by_object: HashMap::new(),
            by_scope: HashMap::new(),
            by_branch: HashMap::new(),
        }
    }

    pub fn count(&self) -> usize {
        self.constraints.iter().filter(|c| c.is_active()).count()
    }

    pub fn branch_count(&self) -> usize {
        self.branches.iter().filter(|b| b.status == BranchStatus::Active).count()
    }

    /// Assert a new fact. Auto-detects conflicts (same subject+predicate, different object
    /// in same scope+branch) and creates a rival branch if needed.
    pub fn assert_fact(
        &mut self,
        subject: String,
        predicate: String,
        object: String,
        confidence: f32,
        scope: String,
        branch_id: u64,
        provenance: Provenance,
        valid_from_ms: i64,
        source_memory_id: Option<MemoryId>,
    ) -> AssertResult {
        let id = self.next_id;
        self.next_id += 1;

        // Check for conflicts: same subject+predicate, different object, same scope+branch, still active
        let conflict = self.detect_conflict(&subject, &predicate, &object, &scope, branch_id);

        let (actual_branch_id, conflict_info) = if let Some(rival_id) = conflict {
            // Create a new branch for the new fact
            let new_branch = self.create_branch_internal(branch_id, scope.clone(), valid_from_ms);
            let rival = &self.constraints[self.id_to_index[&rival_id]];
            let info = ConflictInfo {
                rival_fact_id: rival_id,
                rival_object: rival.object.clone(),
                new_branch_id: new_branch,
            };
            (new_branch, Some(info))
        } else {
            (branch_id, None)
        };

        self.insert_constraint(Constraint {
            id,
            subject,
            predicate,
            object,
            confidence: confidence.clamp(0.0, 1.0),
            scope,
            branch_id: actual_branch_id,
            provenance,
            valid_from_ms,
            valid_to_ms: 0,
            source_memory_id,
        });

        AssertResult { fact_id: id, conflict: conflict_info }
    }

    /// Assert with an explicit ID (for WAL replay).
    pub fn replay_assert(
        &mut self,
        id: u64,
        subject: String,
        predicate: String,
        object: String,
        confidence: f32,
        scope: String,
        branch_id: u64,
        provenance: Provenance,
        valid_from_ms: i64,
        source_memory_id: Option<MemoryId>,
    ) {
        if id >= self.next_id {
            self.next_id = id + 1;
        }
        self.insert_constraint(Constraint {
            id,
            subject,
            predicate,
            object,
            confidence: confidence.clamp(0.0, 1.0),
            scope,
            branch_id,
            provenance,
            valid_from_ms,
            valid_to_ms: 0,
            source_memory_id,
        });
    }

    /// Soft-retract a fact by setting valid_to_ms.
    pub fn retract(&mut self, id: u64, now_ms: i64) -> bool {
        if let Some(&idx) = self.id_to_index.get(&id) {
            self.constraints[idx].valid_to_ms = now_ms;
            true
        } else {
            false
        }
    }

    /// Replay a retraction (WAL replay).
    pub fn replay_retract(&mut self, id: u64, valid_to_ms: i64) {
        if let Some(&idx) = self.id_to_index.get(&id) {
            self.constraints[idx].valid_to_ms = valid_to_ms;
        }
    }

    /// Pattern-match query with wildcards. "_" or empty string matches anything.
    pub fn query_unify(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
        scope: Option<&str>,
    ) -> Vec<&Constraint> {
        // Start with the most selective index
        let candidates: Vec<u64> = if let Some(s) = subject.filter(|s| !s.is_empty() && *s != "_") {
            self.by_subject.get(s).cloned().unwrap_or_default()
        } else if let Some(p) = predicate.filter(|p| !p.is_empty() && *p != "_") {
            self.by_predicate.get(p).cloned().unwrap_or_default()
        } else if let Some(o) = object.filter(|o| !o.is_empty() && *o != "_") {
            self.by_object.get(o).cloned().unwrap_or_default()
        } else {
            // No filter — scan all
            self.constraints.iter().filter(|c| c.is_active()).map(|c| c.id).collect()
        };

        candidates.iter()
            .filter_map(|&id| {
                let idx = *self.id_to_index.get(&id)?;
                let c = &self.constraints[idx];
                if !c.is_active() { return None; }
                if let Some(s) = subject.filter(|s| !s.is_empty() && *s != "_") {
                    if c.subject != s { return None; }
                }
                if let Some(p) = predicate.filter(|p| !p.is_empty() && *p != "_") {
                    if c.predicate != p { return None; }
                }
                if let Some(o) = object.filter(|o| !o.is_empty() && *o != "_") {
                    if c.object != o { return None; }
                }
                if let Some(sc) = scope.filter(|s| !s.is_empty() && *s != "_") {
                    if c.scope != sc { return None; }
                }
                Some(c)
            })
            .collect()
    }

    /// Follow a predicate chain: given subject S and predicates [P1, P2, ...],
    /// find S -P1-> X -P2-> Y -...-> result.
    pub fn query_chain(
        &self,
        start_subject: &str,
        predicates: &[&str],
        max_depth: usize,
    ) -> Vec<Vec<&Constraint>> {
        if predicates.is_empty() || max_depth == 0 {
            return vec![];
        }

        let depth = predicates.len().min(max_depth);
        let mut chains: Vec<Vec<&Constraint>> = vec![];

        // First hop
        let first_matches = self.query_unify(Some(start_subject), Some(predicates[0]), None, None);
        if depth == 1 {
            return first_matches.into_iter().map(|c| vec![c]).collect();
        }

        // Recursive hops
        for first in &first_matches {
            let sub_chains = self.query_chain(&first.object, &predicates[1..], depth - 1);
            for mut chain in sub_chains {
                chain.insert(0, first);
                chains.push(chain);
            }
        }

        chains
    }

    /// Get a fact by ID.
    pub fn get(&self, id: u64) -> Option<&Constraint> {
        self.id_to_index.get(&id).map(|&idx| &self.constraints[idx])
    }

    /// Explain a fact: return it with supporting and conflicting fact IDs + branch info.
    pub fn explain(&self, id: u64) -> Option<Explanation> {
        let fact = self.get(id)?.clone();
        let branch = if fact.branch_id > 0 {
            self.branch_id_to_index.get(&fact.branch_id)
                .map(|&idx| self.branches[idx].clone())
        } else {
            None
        };

        // Find conflicting facts: same subject+predicate, different object
        let conflicting: Vec<u64> = self.query_unify(Some(&fact.subject), Some(&fact.predicate), None, Some(&fact.scope))
            .iter()
            .filter(|c| c.id != id && c.object != fact.object)
            .map(|c| c.id)
            .collect();

        // Find supporting facts: same subject+predicate+object (confirmations)
        let supporting: Vec<u64> = self.query_unify(Some(&fact.subject), Some(&fact.predicate), Some(&fact.object), None)
            .iter()
            .filter(|c| c.id != id)
            .map(|c| c.id)
            .collect();

        Some(Explanation { fact, supporting, conflicting, branch })
    }

    /// Create a branch for rival interpretations.
    pub fn create_branch(&mut self, parent_id: u64, scope: String, created_ms: i64) -> u64 {
        self.create_branch_internal(parent_id, scope, created_ms)
    }

    /// Replay a branch creation (WAL replay).
    pub fn replay_create_branch(&mut self, id: u64, parent_id: u64, scope: String, created_ms: i64) {
        if id >= self.next_branch_id {
            self.next_branch_id = id + 1;
        }
        let idx = self.branches.len();
        self.branches.push(Branch {
            id,
            parent_id,
            scope,
            created_ms,
            status: BranchStatus::Active,
        });
        self.branch_id_to_index.insert(id, idx);
    }

    /// Resolve a conflict: winner branch stays Active, loser branch → Abandoned,
    /// and all loser's facts get retracted.
    pub fn resolve_branch(&mut self, winner_id: u64, loser_id: u64, now_ms: i64) -> bool {
        // Abandon loser branch
        if let Some(&idx) = self.branch_id_to_index.get(&loser_id) {
            self.branches[idx].status = BranchStatus::Abandoned;
        } else {
            return false;
        }

        // Retract all facts on the loser branch
        if let Some(fact_ids) = self.by_branch.get(&loser_id).cloned() {
            for fid in fact_ids {
                self.retract(fid, now_ms);
            }
        }

        // Mark winner as merged (it absorbed the loser's scope)
        if let Some(&idx) = self.branch_id_to_index.get(&winner_id) {
            self.branches[idx].status = BranchStatus::Merged;
        }

        true
    }

    /// Replay a branch resolution (WAL replay).
    pub fn replay_resolve_branch(&mut self, winner_id: u64, loser_id: u64, now_ms: i64) {
        self.resolve_branch(winner_id, loser_id, now_ms);
    }

    /// List all active branches.
    pub fn list_branches(&self) -> Vec<&Branch> {
        self.branches.iter().filter(|b| b.status == BranchStatus::Active).collect()
    }

    // ── Internal helpers ─────────────────────────────────────────────────

    fn insert_constraint(&mut self, c: Constraint) {
        let idx = self.constraints.len();
        self.id_to_index.insert(c.id, idx);
        self.by_subject.entry(c.subject.clone()).or_default().push(c.id);
        self.by_predicate.entry(c.predicate.clone()).or_default().push(c.id);
        self.by_object.entry(c.object.clone()).or_default().push(c.id);
        self.by_scope.entry(c.scope.clone()).or_default().push(c.id);
        self.by_branch.entry(c.branch_id).or_default().push(c.id);
        self.constraints.push(c);
    }

    fn detect_conflict(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        scope: &str,
        branch_id: u64,
    ) -> Option<u64> {
        let matches = self.query_unify(Some(subject), Some(predicate), None, Some(scope));
        matches.iter()
            .find(|c| c.object != object && c.branch_id == branch_id)
            .map(|c| c.id)
    }

    fn create_branch_internal(&mut self, parent_id: u64, scope: String, created_ms: i64) -> u64 {
        let id = self.next_branch_id;
        self.next_branch_id += 1;
        let idx = self.branches.len();
        self.branches.push(Branch {
            id,
            parent_id,
            scope,
            created_ms,
            status: BranchStatus::Active,
        });
        self.branch_id_to_index.insert(id, idx);
        id
    }
}

impl Default for ConstraintStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> i64 {
        1712851200000 // 2024-04-11 fixed timestamp
    }

    #[test]
    fn test_assert_and_query() {
        let mut store = ConstraintStore::new();
        let result = store.assert_fact(
            "user".into(), "prefers".into(), "Rust".into(),
            0.9, "global".into(), 0, Provenance::default(), now(), None,
        );
        assert_eq!(result.fact_id, 1);
        assert!(result.conflict.is_none());

        let matches = store.query_unify(Some("user"), Some("prefers"), None, None);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].object, "Rust");
    }

    #[test]
    fn test_conflict_detection() {
        let mut store = ConstraintStore::new();
        store.assert_fact(
            "user".into(), "prefers".into(), "Rust".into(),
            0.9, "global".into(), 0, Provenance::default(), now(), None,
        );
        let result = store.assert_fact(
            "user".into(), "prefers".into(), "Python".into(),
            0.7, "global".into(), 0, Provenance::default(), now(), None,
        );
        assert!(result.conflict.is_some());
        let conflict = result.conflict.unwrap();
        assert_eq!(conflict.rival_object, "Rust");
        assert!(conflict.new_branch_id > 0);

        // Both facts visible
        let matches = store.query_unify(Some("user"), Some("prefers"), None, None);
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_retract() {
        let mut store = ConstraintStore::new();
        let r = store.assert_fact(
            "file".into(), "modified_at".into(), "1234".into(),
            1.0, "global".into(), 0, Provenance::default(), now(), None,
        );
        assert!(store.retract(r.fact_id, now() + 1000));
        let matches = store.query_unify(Some("file"), None, None, None);
        assert!(matches.is_empty()); // retracted facts are not active
    }

    #[test]
    fn test_chain_query() {
        let mut store = ConstraintStore::new();
        store.assert_fact("A".into(), "causes".into(), "B".into(), 1.0, "global".into(), 0, Provenance::default(), now(), None);
        store.assert_fact("B".into(), "causes".into(), "C".into(), 1.0, "global".into(), 0, Provenance::default(), now(), None);
        store.assert_fact("C".into(), "causes".into(), "D".into(), 1.0, "global".into(), 0, Provenance::default(), now(), None);

        let chains = store.query_chain("A", &["causes", "causes"], 3);
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].len(), 2);
        assert_eq!(chains[0][1].object, "C");
    }

    #[test]
    fn test_explain() {
        let mut store = ConstraintStore::new();
        let r1 = store.assert_fact("X".into(), "is".into(), "red".into(), 0.9, "global".into(), 0, Provenance::default(), now(), None);
        let _r2 = store.assert_fact("X".into(), "is".into(), "blue".into(), 0.7, "global".into(), 0, Provenance::default(), now(), None);

        let explanation = store.explain(r1.fact_id).unwrap();
        assert_eq!(explanation.conflicting.len(), 1);
    }

    #[test]
    fn test_branch_resolve() {
        let mut store = ConstraintStore::new();
        let _r1 = store.assert_fact("X".into(), "color".into(), "red".into(), 0.9, "global".into(), 0, Provenance::default(), now(), None);
        let r2 = store.assert_fact("X".into(), "color".into(), "blue".into(), 0.7, "global".into(), 0, Provenance::default(), now(), None);
        let conflict = r2.conflict.unwrap();

        // Resolve: trunk (0) wins, new branch loses
        store.resolve_branch(0, conflict.new_branch_id, now() + 1000);

        let matches = store.query_unify(Some("X"), Some("color"), None, None);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].object, "red");
    }
}

impl crate::organ::OrganApply for ConstraintStore {
    /// Organ-owned WAL replay (THEORY.md §8 Phase 2).
    fn apply(&mut self, op: crate::ops::Op) -> Option<crate::ops::Op> {
        use crate::ops::Op;
        match op {
            Op::AssertConstraint(c) => {
                use crate::organ::constraint::Provenance;
                self.replay_assert(
                    c.fact_id, c.subject, c.predicate, c.object, c.confidence,
                    c.scope, c.branch_id,
                    Provenance {
                        source: c.provenance_source,
                        session_id: c.provenance_session,
                        confidence_basis: c.provenance_basis,
                    },
                    c.valid_from_ms, c.source_memory_id,
                );
                    None
                }
            Op::RetractConstraint(r) => {
                self.replay_retract(r.fact_id, r.retracted_at_ms);
                    None
                }
            Op::CreateBranch(b) => {
                self.replay_create_branch(b.branch_id, b.parent_id, b.scope, b.created_ms);
                    None
                }
            Op::ResolveBranch(r) => {
                self.replay_resolve_branch(r.winner_id, r.loser_id, r.resolved_at_ms);
                    None
                }
            other => Some(other),
        }
    }
}

//! Epistemic Debt Memory — uncertainty boundaries and competing hypotheses.
//!
//! Tracks things the system doesn't know but should: unresolved questions,
//! competing interpretations, and decisions made under uncertainty. The
//! fragility score captures how much current behavior depends on unverified
//! assumptions. Reference: epistemic vigilance (Sperber et al. 2010).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DebtStatus {
    Open,
    Resolved,
    Deferred,
}

impl DebtStatus {
    pub fn to_u8(self) -> u8 {
        match self {
            DebtStatus::Open => 0,
            DebtStatus::Resolved => 1,
            DebtStatus::Deferred => 2,
        }
    }

    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => DebtStatus::Open,
            1 => DebtStatus::Resolved,
            _ => DebtStatus::Deferred,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebtEvidence {
    pub memory_ids: Vec<u64>,
    pub confidence: f32,
    pub note: Option<String>,
    pub attached_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicDebt {
    pub id: u64,
    pub pattern: String,
    pub competing_hypotheses: Vec<String>,
    pub discriminating_test: Option<String>,
    pub fragility_score: f32,
    pub domain: String,
    pub status: DebtStatus,
    pub created_ms: i64,
    pub resolved_ms: i64,
    pub resolution: Option<String>,
    pub realm: String,
    pub source_session: Option<String>,
    #[serde(default)]
    pub evidence: Vec<DebtEvidence>,
    #[serde(default)]
    pub auto_resolved: bool,
}

pub struct DebtStats {
    pub total: usize,
    pub open: usize,
    pub resolved: usize,
    pub deferred: usize,
    pub avg_fragility_open: f32,
}

pub struct EpistemicDebtStore {
    next_id: u64,
    debts: Vec<EpistemicDebt>,
    id_to_index: HashMap<u64, usize>,
    by_domain: HashMap<String, Vec<u64>>,
}

impl EpistemicDebtStore {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            debts: Vec::new(),
            id_to_index: HashMap::new(),
            by_domain: HashMap::new(),
        }
    }

    pub fn register(
        &mut self,
        pattern: String,
        competing_hypotheses: Vec<String>,
        discriminating_test: Option<String>,
        fragility_score: f32,
        domain: String,
        realm: String,
        source_session: Option<String>,
        now_ms: i64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let debt = EpistemicDebt {
            id,
            pattern,
            competing_hypotheses,
            discriminating_test,
            fragility_score: fragility_score.clamp(0.0, 1.0),
            domain,
            status: DebtStatus::Open,
            created_ms: now_ms,
            resolved_ms: 0,
            resolution: None,
            realm,
            source_session,
            evidence: Vec::new(),
            auto_resolved: false,
        };
        self.insert(debt);
        id
    }

    pub fn replay_register(&mut self, debt: EpistemicDebt) {
        if debt.id >= self.next_id {
            self.next_id = debt.id + 1;
        }
        self.insert(debt);
    }

    fn insert(&mut self, debt: EpistemicDebt) {
        let id = debt.id;
        let idx = self.debts.len();
        self.by_domain
            .entry(debt.domain.clone())
            .or_default()
            .push(id);
        self.debts.push(debt);
        self.id_to_index.insert(id, idx);
    }

    pub fn get(&self, id: u64) -> Option<&EpistemicDebt> {
        self.id_to_index.get(&id).map(|&idx| &self.debts[idx])
    }

    fn get_mut(&mut self, id: u64) -> Option<&mut EpistemicDebt> {
        self.id_to_index
            .get(&id)
            .copied()
            .map(|idx| &mut self.debts[idx])
    }

    pub fn resolve(&mut self, id: u64, resolution: String, now_ms: i64) -> bool {
        if let Some(debt) = self.get_mut(id) {
            debt.status = DebtStatus::Resolved;
            debt.resolved_ms = now_ms;
            debt.resolution = Some(resolution);
            true
        } else {
            false
        }
    }

    pub fn defer(&mut self, id: u64) -> bool {
        if let Some(debt) = self.get_mut(id) {
            debt.status = DebtStatus::Deferred;
            true
        } else {
            false
        }
    }

    pub fn replay_update(
        &mut self,
        id: u64,
        status: DebtStatus,
        resolved_ms: i64,
        resolution: Option<String>,
    ) {
        if let Some(debt) = self.get_mut(id) {
            debt.status = status;
            debt.resolved_ms = resolved_ms;
            if resolution.is_some() {
                debt.resolution = resolution;
            }
        }
    }

    pub fn query(
        &self,
        status: Option<DebtStatus>,
        domain: Option<&str>,
        realm: Option<&str>,
        min_fragility: Option<f32>,
        limit: usize,
    ) -> Vec<&EpistemicDebt> {
        let candidates: Box<dyn Iterator<Item = &EpistemicDebt>> = if let Some(d) = domain {
            if let Some(ids) = self.by_domain.get(d) {
                Box::new(ids.iter().rev().filter_map(|id| self.get(*id)))
            } else {
                return Vec::new();
            }
        } else {
            Box::new(self.debts.iter().rev())
        };

        candidates
            .filter(|d| status.map_or(true, |s| d.status == s))
            .filter(|d| realm.map_or(true, |r| d.realm == r))
            .filter(|d| min_fragility.map_or(true, |f| d.fragility_score >= f))
            .take(limit)
            .collect()
    }

    pub fn get_fragile_decisions(&self, threshold: f32, limit: usize) -> Vec<&EpistemicDebt> {
        let mut open: Vec<&EpistemicDebt> = self
            .debts
            .iter()
            .filter(|d| d.status == DebtStatus::Open && d.fragility_score >= threshold)
            .collect();
        open.sort_by(|a, b| {
            b.fragility_score
                .partial_cmp(&a.fragility_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        open.truncate(limit);
        open
    }

    pub fn has_open_debt_for_domain(&self, domain: &str) -> bool {
        self.by_domain
            .get(domain)
            .map(|ids| {
                ids.iter()
                    .any(|id| self.get(*id).map_or(false, |d| d.status == DebtStatus::Open))
            })
            .unwrap_or(false)
    }

    pub fn stats(&self) -> DebtStats {
        let mut open = 0usize;
        let mut resolved = 0usize;
        let mut deferred = 0usize;
        let mut fragility_sum = 0.0f32;

        for debt in &self.debts {
            match debt.status {
                DebtStatus::Open => {
                    open += 1;
                    fragility_sum += debt.fragility_score;
                }
                DebtStatus::Resolved => resolved += 1,
                DebtStatus::Deferred => deferred += 1,
            }
        }

        DebtStats {
            total: self.debts.len(),
            open,
            resolved,
            deferred,
            avg_fragility_open: if open > 0 {
                fragility_sum / open as f32
            } else {
                0.0
            },
        }
    }

    pub fn count(&self) -> usize {
        self.debts.len()
    }

    /// Attach evidence to an open debt. Returns true if debt found.
    pub fn attach_evidence(
        &mut self,
        id: u64,
        memory_ids: Vec<u64>,
        confidence: f32,
        note: Option<String>,
        now_ms: i64,
    ) -> bool {
        if let Some(debt) = self.get_mut(id) {
            debt.evidence.push(DebtEvidence {
                memory_ids,
                confidence: confidence.clamp(0.0, 1.0),
                note,
                attached_ms: now_ms,
            });
            true
        } else {
            false
        }
    }

    /// WAL replay for attach evidence.
    pub fn replay_attach_evidence(
        &mut self,
        id: u64,
        memory_ids: Vec<u64>,
        confidence: f32,
        note: Option<String>,
        attached_ms: i64,
    ) {
        if let Some(debt) = self.get_mut(id) {
            debt.evidence.push(DebtEvidence {
                memory_ids,
                confidence: confidence.clamp(0.0, 1.0),
                note,
                attached_ms,
            });
        }
    }

    /// Auto-resolve a debt if cumulative evidence confidence exceeds threshold.
    /// Returns true if the debt was resolved.
    pub fn auto_resolve_if_ready(&mut self, id: u64, threshold: f32, now_ms: i64) -> bool {
        if let Some(debt) = self.get_mut(id) {
            if debt.status != DebtStatus::Open {
                return false;
            }
            let total_confidence: f32 = debt.evidence.iter().map(|e| e.confidence).sum();
            if total_confidence >= threshold {
                debt.status = DebtStatus::Resolved;
                debt.resolved_ms = now_ms;
                debt.auto_resolved = true;
                debt.resolution = Some(format!(
                    "auto-resolved: cumulative evidence confidence {:.2} >= {:.2}",
                    total_confidence, threshold
                ));
                return true;
            }
        }
        false
    }

    /// Get all open debts with their evidence for the learning cycle.
    pub fn open_debts_with_evidence(&self) -> Vec<&EpistemicDebt> {
        self.debts
            .iter()
            .filter(|d| d.status == DebtStatus::Open)
            .collect()
    }
}

impl crate::organ::OrganApply for EpistemicDebtStore {
    /// Organ-owned WAL replay (THEORY.md §8 Phase 2).
    fn apply(&mut self, op: crate::ops::Op) -> Option<crate::ops::Op> {
        use crate::ops::Op;
        match op {
            Op::RegisterDebt(d) => {
                self.replay_register(crate::organ::epistemic_debt::EpistemicDebt {
                    id: d.debt_id,
                    pattern: d.pattern,
                    competing_hypotheses: d.competing_hypotheses,
                    discriminating_test: d.discriminating_test,
                    fragility_score: d.fragility_score,
                    domain: d.domain,
                    status: crate::organ::epistemic_debt::DebtStatus::Open,
                    created_ms: d.created_ms,
                    resolved_ms: 0,
                    resolution: None,
                    realm: d.realm,
                    source_session: d.source_session,
                    evidence: Vec::new(),
                    auto_resolved: false,
                });
                    None
                }
            Op::UpdateDebt(u) => {
                let status = crate::organ::epistemic_debt::DebtStatus::from_u8(u.status);
                self.replay_update(u.debt_id, status, u.resolved_ms, u.resolution);
                    None
                }
            Op::AttachDebtEvidence(e) => {
                self.replay_attach_evidence(
                    e.debt_id,
                    e.evidence_memory_ids,
                    e.confidence,
                    e.note,
                    e.attached_ms,
                );
                    None
                }
            other => Some(other),
        }
    }
}

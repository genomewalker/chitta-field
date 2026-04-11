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
}

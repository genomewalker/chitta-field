use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Lifecycle FSM ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WisdomLifecycle {
    Candidate = 0,
    Provisional = 1,
    Trusted = 2,
    Demoted = 3,
}

impl WisdomLifecycle {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Candidate,
            1 => Self::Provisional,
            2 => Self::Trusted,
            3 => Self::Demoted,
            _ => Self::Candidate,
        }
    }

    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::Provisional => "provisional",
            Self::Trusted => "trusted",
            Self::Demoted => "demoted",
        }
    }
}

// ── Data Model ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WisdomCandidate {
    pub id: u64,
    pub cluster_key: String,
    pub domain: String,
    pub action: String,
    pub summary: String,
    pub episode_ids: Vec<u64>,
    pub debt_ids: Vec<u64>,
    pub support_count: u32,
    pub cross_session_count: u32,
    pub mean_surprise: f32,
    pub promotion_score: f32,
    pub contradiction_count: u32,
    pub lifecycle: WisdomLifecycle,
    pub memory_id: Option<u64>,
    pub created_ms: i64,
    pub updated_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WisdomPromotionStats {
    pub total_candidates: usize,
    pub by_lifecycle: HashMap<String, usize>,
    pub promoted_count: usize,
    pub demoted_count: usize,
}

// ── Promotion thresholds ──────────────────────────────────────────────────

pub const MIN_SUPPORT_COUNT: u32 = 4;
pub const MIN_CROSS_SESSIONS: u32 = 2;
pub const PROMOTION_SCORE_THRESHOLD: f32 = 0.72;

// ── Store ─────────────────────────────────────────────────────────────────

pub struct WisdomPromotionStore {
    next_id: u64,
    candidates: Vec<WisdomCandidate>,
    id_to_index: HashMap<u64, usize>,
    by_cluster_key: HashMap<String, u64>,
    by_domain: HashMap<String, Vec<u64>>,
}

impl WisdomPromotionStore {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            candidates: Vec::new(),
            id_to_index: HashMap::new(),
            by_cluster_key: HashMap::new(),
            by_domain: HashMap::new(),
        }
    }

    /// Upsert a candidate by cluster_key. Returns the candidate id (existing or new).
    pub fn upsert_candidate(
        &mut self,
        cluster_key: String,
        domain: String,
        action: String,
        summary: String,
        episode_ids: Vec<u64>,
        debt_ids: Vec<u64>,
        support_count: u32,
        cross_session_count: u32,
        mean_surprise: f32,
        promotion_score: f32,
        now_ms: i64,
    ) -> u64 {
        if let Some(&existing_id) = self.by_cluster_key.get(&cluster_key) {
            if let Some(&idx) = self.id_to_index.get(&existing_id) {
                let c = &mut self.candidates[idx];
                c.summary = summary;
                c.episode_ids = episode_ids;
                c.debt_ids = debt_ids;
                c.support_count = support_count;
                c.cross_session_count = cross_session_count;
                c.mean_surprise = mean_surprise;
                c.promotion_score = promotion_score;
                c.updated_ms = now_ms;
            }
            return existing_id;
        }

        let id = self.next_id;
        self.next_id += 1;

        let candidate = WisdomCandidate {
            id,
            cluster_key: cluster_key.clone(),
            domain: domain.clone(),
            action,
            summary,
            episode_ids,
            debt_ids,
            support_count,
            cross_session_count,
            mean_surprise,
            promotion_score,
            contradiction_count: 0,
            lifecycle: WisdomLifecycle::Candidate,
            memory_id: None,
            created_ms: now_ms,
            updated_ms: now_ms,
        };

        let idx = self.candidates.len();
        self.candidates.push(candidate);
        self.id_to_index.insert(id, idx);
        self.by_cluster_key.insert(cluster_key, id);
        self.by_domain
            .entry(domain)
            .or_insert_with(Vec::new)
            .push(id);

        id
    }

    /// WAL replay for upsert.
    pub fn replay_upsert(&mut self, candidate: WisdomCandidate) {
        if candidate.id >= self.next_id {
            self.next_id = candidate.id + 1;
        }
        if let Some(&existing_id) = self.by_cluster_key.get(&candidate.cluster_key) {
            if let Some(&idx) = self.id_to_index.get(&existing_id) {
                self.candidates[idx] = candidate;
                return;
            }
        }
        let id = candidate.id;
        let cluster_key = candidate.cluster_key.clone();
        let domain = candidate.domain.clone();
        let idx = self.candidates.len();
        self.candidates.push(candidate);
        self.id_to_index.insert(id, idx);
        self.by_cluster_key.insert(cluster_key, id);
        self.by_domain
            .entry(domain)
            .or_insert_with(Vec::new)
            .push(id);
    }

    /// Transition lifecycle state.
    pub fn update_lifecycle(
        &mut self,
        id: u64,
        new_state: WisdomLifecycle,
        memory_id: Option<u64>,
        contradiction_count: u32,
        now_ms: i64,
    ) -> bool {
        if let Some(&idx) = self.id_to_index.get(&id) {
            let c = &mut self.candidates[idx];
            c.lifecycle = new_state;
            if let Some(mid) = memory_id {
                c.memory_id = Some(mid);
            }
            c.contradiction_count = contradiction_count;
            c.updated_ms = now_ms;
            true
        } else {
            false
        }
    }

    /// WAL replay for lifecycle update.
    pub fn replay_lifecycle(
        &mut self,
        id: u64,
        new_state: u8,
        memory_id: Option<u64>,
        contradiction_count: u32,
        updated_ms: i64,
    ) {
        if let Some(&idx) = self.id_to_index.get(&id) {
            let c = &mut self.candidates[idx];
            c.lifecycle = WisdomLifecycle::from_u8(new_state);
            if let Some(mid) = memory_id {
                c.memory_id = Some(mid);
            }
            c.contradiction_count = contradiction_count;
            c.updated_ms = updated_ms;
        }
    }

    pub fn get(&self, id: u64) -> Option<&WisdomCandidate> {
        self.id_to_index
            .get(&id)
            .map(|&idx| &self.candidates[idx])
    }

    pub fn query(
        &self,
        lifecycle: Option<WisdomLifecycle>,
        domain: Option<&str>,
        limit: usize,
    ) -> Vec<&WisdomCandidate> {
        let iter = self.candidates.iter().filter(|c| {
            if let Some(lc) = lifecycle {
                if c.lifecycle != lc {
                    return false;
                }
            }
            if let Some(d) = domain {
                if c.domain != d {
                    return false;
                }
            }
            true
        });

        let mut results: Vec<_> = iter.collect();
        results.sort_by(|a, b| {
            b.promotion_score
                .partial_cmp(&a.promotion_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        results
    }

    /// Return candidates ready for promotion to next lifecycle stage.
    pub fn promotable(&self, min_score: f32) -> Vec<&WisdomCandidate> {
        self.candidates
            .iter()
            .filter(|c| {
                c.lifecycle == WisdomLifecycle::Candidate
                    && c.support_count >= MIN_SUPPORT_COUNT
                    && c.cross_session_count >= MIN_CROSS_SESSIONS
                    && c.promotion_score >= min_score
                    && c.contradiction_count == 0
            })
            .collect()
    }

    pub fn count(&self) -> usize {
        self.candidates.len()
    }

    pub fn stats(&self) -> WisdomPromotionStats {
        let mut by_lifecycle = HashMap::new();
        let mut promoted = 0usize;
        let mut demoted = 0usize;

        for c in &self.candidates {
            *by_lifecycle
                .entry(c.lifecycle.as_str().to_string())
                .or_insert(0) += 1;
            if c.lifecycle == WisdomLifecycle::Trusted {
                promoted += 1;
            }
            if c.lifecycle == WisdomLifecycle::Demoted {
                demoted += 1;
            }
        }

        WisdomPromotionStats {
            total_candidates: self.candidates.len(),
            by_lifecycle,
            promoted_count: promoted,
            demoted_count: demoted,
        }
    }
}

impl crate::organ::OrganApply for WisdomPromotionStore {
    /// Organ-owned WAL replay (THEORY.md §8 Phase 2).
    fn apply(&mut self, op: crate::ops::Op) -> Option<crate::ops::Op> {
        use crate::ops::Op;
        match op {
            Op::UpsertWisdomCandidate(w) => {
                self.replay_upsert(
                    crate::organ::wisdom_promotion::WisdomCandidate {
                        id: w.candidate_id,
                        cluster_key: w.cluster_key,
                        domain: w.domain,
                        action: w.action,
                        summary: w.summary,
                        episode_ids: w.episode_ids,
                        debt_ids: w.debt_ids,
                        support_count: w.support_count,
                        cross_session_count: w.cross_session_count,
                        mean_surprise: w.mean_surprise,
                        promotion_score: w.promotion_score,
                        contradiction_count: 0,
                        lifecycle: crate::organ::wisdom_promotion::WisdomLifecycle::Candidate,
                        memory_id: None,
                        created_ms: w.created_ms,
                        updated_ms: w.created_ms,
                    },
                );
                    None
                }
            Op::UpdateWisdomLifecycle(l) => {
                self.replay_lifecycle(
                    l.candidate_id,
                    l.new_state,
                    l.memory_id,
                    l.contradiction_count,
                    l.updated_ms,
                );
                    None
                }
            other => Some(other),
        }
    }
}

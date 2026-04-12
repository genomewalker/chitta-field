use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Thresholds ────────────────────────────────────────────────────────────

pub const WATCH_CONTRADICTION_RATIO: f32 = 0.30;
pub const INFLAMED_CONTRADICTION_RATIO: f32 = 0.60;
pub const WATCH_STALENESS: f32 = 0.40;
pub const INFLAMED_STALENESS: f32 = 0.70;
pub const STALENESS_GROWTH_PER_TICK: f32 = 0.05;
pub const STALENESS_DECAY_PER_SUPPORT: f32 = 0.08;
pub const SUPPORT_DELTA_HIT: f32 = 0.12;
pub const CONTRADICTION_DELTA_HIT: f32 = 0.18;
pub const DEFAULT_REDERIVE_TTL_MS: i64 = 604_800_000; // 7 days

// ── State FSM ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineageState {
    Trusted = 0,
    Watch = 1,
    Inflamed = 2,
    Demoted = 3,
}

impl LineageState {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Trusted,
            1 => Self::Watch,
            2 => Self::Inflamed,
            3 => Self::Demoted,
            _ => Self::Trusted,
        }
    }

    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Trusted => "trusted",
            Self::Watch => "watch",
            Self::Inflamed => "inflamed",
            Self::Demoted => "demoted",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RederiveAction {
    Reaffirm = 0,
    Narrow = 1,
    Split = 2,
    Demote = 3,
}

impl RederiveAction {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Reaffirm,
            1 => Self::Narrow,
            2 => Self::Split,
            3 => Self::Demote,
            _ => Self::Demote,
        }
    }

    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

// ── Data Model ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ApplicabilityEnvelope {
    /// Domain this claim applies to. "*" = any domain.
    pub domain: String,
    /// Action types this claim was observed under. Empty = any action.
    pub action_types: Vec<String>,
    /// Free-text precondition tags (e.g. "codebase_rust", "file_exists").
    pub preconditions: Vec<String>,
    /// Source family names that produced the original evidence.
    pub source_families: Vec<String>,
}

impl ApplicabilityEnvelope {
    /// Returns 0..1 overlap score with a query (domain, action).
    /// 1.0 = both match; 0.5 = one dimension matches; 0.0 = no match.
    pub fn overlap(&self, domain: &str, action: &str) -> f32 {
        let domain_score = if self.domain == "*" || self.domain.is_empty() || self.domain == domain {
            1.0f32
        } else {
            0.0f32
        };
        let action_score = if self.action_types.is_empty()
            || self.action_types.iter().any(|a| a == "*" || a == action)
        {
            1.0f32
        } else {
            0.0f32
        };
        (domain_score + action_score) / 2.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengerEvidence {
    pub intervention_id: Option<u64>,
    pub surprise_id: Option<u64>,
    pub outcome_summary: String,
    pub attached_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WisdomLineage {
    pub id: u64,
    /// Links to the WisdomCandidate that was promoted to produce this lineage.
    pub wisdom_candidate_id: u64,
    pub claim: String,
    pub envelope: ApplicabilityEnvelope,
    pub seed_episode_ids: Vec<u64>,
    pub seed_surprise_ids: Vec<u64>,
    pub seed_intervention_ids: Vec<u64>,
    pub seed_debt_ids: Vec<u64>,
    /// Lineage this record was derived from (supersedes / narrows / splits).
    pub ancestor_lineage_id: Option<u64>,
    pub derivation_version: u32,
    /// "supersedes" | "branches_from" | "narrows" | "splits_from"
    pub derivation_relation: Option<String>,
    pub support_mass: f32,
    pub contradiction_mass: f32,
    /// Grows when no fresh support arrives; decays on support events.
    pub staleness_mass: f32,
    pub last_supported_ms: i64,
    pub last_challenged_ms: i64,
    pub state: LineageState,
    pub challengers: Vec<ChallengerEvidence>,
    pub rederive_task_id: Option<u64>,
    pub rederive_opened_ms: Option<i64>,
    pub rederive_ttl_ms: i64,
    pub created_ms: i64,
    pub updated_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WisdomLineageStats {
    pub total: usize,
    pub by_state: HashMap<String, usize>,
    pub rederive_pending: usize,
    pub demoted_ttl_expiry: usize,
    pub total_support_mass: f32,
    pub total_contradiction_mass: f32,
    pub mean_staleness: f32,
}

// ── Store ─────────────────────────────────────────────────────────────────

pub struct WisdomLineageStore {
    next_id: u64,
    lineages: Vec<WisdomLineage>,
    id_to_index: HashMap<u64, usize>,
    by_candidate: HashMap<u64, u64>,
    by_domain: HashMap<String, Vec<u64>>,
    by_state: HashMap<u8, Vec<u64>>,
}

impl WisdomLineageStore {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            lineages: Vec::new(),
            id_to_index: HashMap::new(),
            by_candidate: HashMap::new(),
            by_domain: HashMap::new(),
            by_state: HashMap::new(),
        }
    }

    /// Enroll a new lineage when a WisdomCandidate reaches Trusted state.
    pub fn enroll(
        &mut self,
        wisdom_candidate_id: u64,
        claim: String,
        envelope: ApplicabilityEnvelope,
        seed_episode_ids: Vec<u64>,
        seed_surprise_ids: Vec<u64>,
        seed_intervention_ids: Vec<u64>,
        seed_debt_ids: Vec<u64>,
        ancestor_lineage_id: Option<u64>,
        derivation_relation: Option<String>,
        now_ms: i64,
    ) -> u64 {
        // Idempotent by candidate id
        if let Some(&existing) = self.by_candidate.get(&wisdom_candidate_id) {
            return existing;
        }

        let id = self.next_id;
        self.next_id += 1;

        let lineage = WisdomLineage {
            id,
            wisdom_candidate_id,
            claim,
            envelope: envelope.clone(),
            seed_episode_ids,
            seed_surprise_ids,
            seed_intervention_ids,
            seed_debt_ids,
            ancestor_lineage_id,
            derivation_version: ancestor_lineage_id.map(|_| 1).unwrap_or(0),
            derivation_relation,
            support_mass: 0.0,
            contradiction_mass: 0.0,
            staleness_mass: 0.0,
            last_supported_ms: now_ms,
            last_challenged_ms: 0,
            state: LineageState::Trusted,
            challengers: Vec::new(),
            rederive_task_id: None,
            rederive_opened_ms: None,
            rederive_ttl_ms: DEFAULT_REDERIVE_TTL_MS,
            created_ms: now_ms,
            updated_ms: now_ms,
        };

        self.index_insert(&lineage);
        let idx = self.lineages.len();
        self.lineages.push(lineage);
        self.id_to_index.insert(id, idx);
        id
    }

    /// Replay a WAL-reconstructed lineage (used in open()).
    pub fn replay_upsert(&mut self, lineage: WisdomLineage) {
        if self.id_to_index.contains_key(&lineage.id) {
            if let Some(&idx) = self.id_to_index.get(&lineage.id) {
                self.index_remove(&self.lineages[idx].clone());
                self.index_insert(&lineage);
                self.lineages[idx] = lineage;
            }
        } else {
            if lineage.id >= self.next_id {
                self.next_id = lineage.id + 1;
            }
            self.index_insert(&lineage);
            let idx = self.lineages.len();
            let id = lineage.id;
            self.lineages.push(lineage);
            self.id_to_index.insert(id, idx);
        }
    }

    /// Apply support/contradiction/staleness deltas; returns new state if a transition fired.
    pub fn adjudicate(
        &mut self,
        id: u64,
        support_delta: f32,
        contradiction_delta: f32,
        staleness_delta: f32,
        now_ms: i64,
    ) -> Option<LineageState> {
        let idx = *self.id_to_index.get(&id)?;
        let lineage = &mut self.lineages[idx];

        if matches!(lineage.state, LineageState::Demoted) {
            return None;
        }

        if support_delta > 0.0 {
            lineage.support_mass = (lineage.support_mass + support_delta).min(10.0);
            lineage.staleness_mass = (lineage.staleness_mass - STALENESS_DECAY_PER_SUPPORT).max(0.0);
            lineage.last_supported_ms = now_ms;
        }
        if contradiction_delta > 0.0 {
            lineage.contradiction_mass = (lineage.contradiction_mass + contradiction_delta).min(10.0);
            lineage.last_challenged_ms = now_ms;
        }
        if staleness_delta > 0.0 {
            lineage.staleness_mass = (lineage.staleness_mass + staleness_delta).min(1.0);
        }
        lineage.updated_ms = now_ms;

        let new_state = Self::evaluate_transition(lineage);
        if let Some(ns) = new_state {
            self.transition_state(id, ns, "adjudication", None, now_ms);
        }
        new_state
    }

    /// Replay adjudication from WAL (no transition logic — state came from WAL).
    pub fn replay_adjudicate(
        &mut self,
        id: u64,
        support_mass: f32,
        contradiction_mass: f32,
        staleness_mass: f32,
        last_supported_ms: i64,
        last_challenged_ms: i64,
        updated_ms: i64,
    ) {
        if let Some(&idx) = self.id_to_index.get(&id) {
            let l = &mut self.lineages[idx];
            l.support_mass = support_mass;
            l.contradiction_mass = contradiction_mass;
            l.staleness_mass = staleness_mass;
            l.last_supported_ms = last_supported_ms;
            l.last_challenged_ms = last_challenged_ms;
            l.updated_ms = updated_ms;
        }
    }

    /// Attach a challenger evidence record.
    pub fn record_challenger(&mut self, id: u64, evidence: ChallengerEvidence, now_ms: i64) {
        if let Some(&idx) = self.id_to_index.get(&id) {
            self.lineages[idx].challengers.push(evidence);
            self.lineages[idx].updated_ms = now_ms;
        }
    }

    /// Apply a state transition (called from adjudicate or explicitly from subconscious).
    pub fn transition_state(
        &mut self,
        id: u64,
        new_state: LineageState,
        reason: &str,
        rederive_task_id: Option<u64>,
        now_ms: i64,
    ) -> bool {
        let idx = match self.id_to_index.get(&id) {
            Some(&i) => i,
            None => return false,
        };
        let old_state = self.lineages[idx].state;
        if old_state == new_state {
            return false;
        }
        // Remove from old state index
        if let Some(v) = self.by_state.get_mut(&(old_state.as_u8())) {
            v.retain(|&x| x != id);
        }
        self.lineages[idx].state = new_state;
        if let Some(tid) = rederive_task_id {
            self.lineages[idx].rederive_task_id = Some(tid);
            self.lineages[idx].rederive_opened_ms = Some(now_ms);
        }
        self.lineages[idx].updated_ms = now_ms;
        let _ = reason; // stored in WAL op, not in RAM struct
        self.by_state.entry(new_state.as_u8()).or_default().push(id);
        true
    }

    /// Close a re-derivation contract. Action determines what happens next.
    /// Returns the new fork lineage id if action == Split, else None.
    pub fn close_rederive(
        &mut self,
        id: u64,
        action: RederiveAction,
        new_envelope: Option<ApplicabilityEnvelope>,
        fork_claim: Option<String>,
        fork_lineage_id: Option<u64>,
        now_ms: i64,
    ) -> Option<u64> {
        let idx = *self.id_to_index.get(&id)?;

        match action {
            RederiveAction::Reaffirm => {
                self.lineages[idx].contradiction_mass = 0.0;
                self.lineages[idx].staleness_mass = 0.0;
                self.lineages[idx].rederive_task_id = None;
                self.lineages[idx].rederive_opened_ms = None;
                self.transition_state(id, LineageState::Trusted, "reaffirmed", None, now_ms);
            }
            RederiveAction::Narrow => {
                if let Some(env) = new_envelope {
                    self.lineages[idx].envelope = env;
                }
                self.lineages[idx].contradiction_mass = 0.0;
                self.lineages[idx].staleness_mass = 0.0;
                self.lineages[idx].rederive_task_id = None;
                self.lineages[idx].rederive_opened_ms = None;
                self.lineages[idx].derivation_version += 1;
                self.transition_state(id, LineageState::Trusted, "narrowed", None, now_ms);
            }
            RederiveAction::Split => {
                // Demote this lineage; caller will enroll the fork separately.
                self.transition_state(id, LineageState::Demoted, "split", None, now_ms);
                self.lineages[idx].rederive_task_id = None;
                self.lineages[idx].rederive_opened_ms = None;
                // If fork_lineage_id is pre-allocated, record it.
                if let Some(fork_id) = fork_lineage_id {
                    // The fork is enrolled via a separate enroll() call.
                    return Some(fork_id);
                }
            }
            RederiveAction::Demote => {
                self.transition_state(id, LineageState::Demoted, "demoted_by_rederive", None, now_ms);
                self.lineages[idx].rederive_task_id = None;
                self.lineages[idx].rederive_opened_ms = None;
            }
        }
        let _ = fork_claim;
        None
    }

    /// Grow staleness on all Trusted/Watch lineages that haven't received support recently.
    /// Returns IDs of lineages that transitioned state.
    pub fn tick_staleness(&mut self, now_ms: i64) -> Vec<u64> {
        let candidates: Vec<u64> = self
            .lineages
            .iter()
            .filter(|l| matches!(l.state, LineageState::Trusted | LineageState::Watch))
            .map(|l| l.id)
            .collect();

        let mut transitioned = Vec::new();
        for id in candidates {
            let idx = self.id_to_index[&id];
            let needs_growth = {
                let l = &self.lineages[idx];
                // If last support was more than 3 days ago, grow staleness
                now_ms - l.last_supported_ms > 259_200_000
            };
            if needs_growth {
                let new_state = {
                    let l = &mut self.lineages[idx];
                    l.staleness_mass = (l.staleness_mass + STALENESS_GROWTH_PER_TICK).min(1.0);
                    l.updated_ms = now_ms;
                    Self::evaluate_transition(l)
                };
                if let Some(ns) = new_state {
                    self.transition_state(id, ns, "staleness_tick", None, now_ms);
                    transitioned.push(id);
                }
            }
        }
        transitioned
    }

    /// Return IDs of Inflamed lineages whose rederive TTL has expired.
    pub fn expiry_check(&self, now_ms: i64) -> Vec<u64> {
        self.lineages
            .iter()
            .filter(|l| {
                matches!(l.state, LineageState::Inflamed)
                    && l.rederive_opened_ms
                        .map(|t| now_ms - t > l.rederive_ttl_ms)
                        .unwrap_or(false)
            })
            .map(|l| l.id)
            .collect()
    }

    /// Find lineage IDs whose envelope overlaps (domain, action) above threshold 0.5.
    pub fn find_by_envelope(&self, domain: &str, action: &str) -> Vec<u64> {
        self.lineages
            .iter()
            .filter(|l| {
                !matches!(l.state, LineageState::Demoted)
                    && l.envelope.overlap(domain, action) > 0.5
            })
            .map(|l| l.id)
            .collect()
    }

    /// Query lineages with optional state and domain filters.
    pub fn query(
        &self,
        state_filter: Option<LineageState>,
        domain_filter: Option<&str>,
        limit: usize,
    ) -> Vec<&WisdomLineage> {
        self.lineages
            .iter()
            .filter(|l| {
                state_filter.map(|s| l.state == s).unwrap_or(true)
                    && domain_filter
                        .map(|d| l.envelope.domain == d || l.envelope.domain == "*")
                        .unwrap_or(true)
            })
            .take(if limit == 0 { usize::MAX } else { limit })
            .collect()
    }

    pub fn inflamed(&self) -> Vec<&WisdomLineage> {
        self.query(Some(LineageState::Inflamed), None, 0)
    }

    pub fn get(&self, id: u64) -> Option<&WisdomLineage> {
        self.id_to_index.get(&id).map(|&idx| &self.lineages[idx])
    }

    pub fn get_by_candidate(&self, candidate_id: u64) -> Option<&WisdomLineage> {
        self.by_candidate
            .get(&candidate_id)
            .and_then(|&id| self.get(id))
    }

    pub fn stats(&self) -> WisdomLineageStats {
        let mut by_state: HashMap<String, usize> = HashMap::new();
        let mut rederive_pending = 0usize;
        let mut total_support = 0.0f32;
        let mut total_contradiction = 0.0f32;
        let mut total_staleness = 0.0f32;

        for l in &self.lineages {
            *by_state.entry(l.state.as_str().to_string()).or_insert(0) += 1;
            if l.rederive_task_id.is_some() {
                rederive_pending += 1;
            }
            total_support += l.support_mass;
            total_contradiction += l.contradiction_mass;
            total_staleness += l.staleness_mass;
        }

        let n = self.lineages.len();
        WisdomLineageStats {
            total: n,
            by_state,
            rederive_pending,
            demoted_ttl_expiry: 0,
            total_support_mass: total_support,
            total_contradiction_mass: total_contradiction,
            mean_staleness: if n > 0 { total_staleness / n as f32 } else { 0.0 },
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────

    fn evaluate_transition(l: &WisdomLineage) -> Option<LineageState> {
        let ratio = l.contradiction_mass / (l.support_mass + 0.01);
        match l.state {
            LineageState::Trusted => {
                if ratio > INFLAMED_CONTRADICTION_RATIO || l.staleness_mass > INFLAMED_STALENESS {
                    Some(LineageState::Inflamed)
                } else if ratio > WATCH_CONTRADICTION_RATIO || l.staleness_mass > WATCH_STALENESS {
                    Some(LineageState::Watch)
                } else {
                    None
                }
            }
            LineageState::Watch => {
                if ratio > INFLAMED_CONTRADICTION_RATIO || l.staleness_mass > INFLAMED_STALENESS {
                    Some(LineageState::Inflamed)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn index_insert(&mut self, l: &WisdomLineage) {
        self.by_candidate.insert(l.wisdom_candidate_id, l.id);
        self.by_domain
            .entry(l.envelope.domain.clone())
            .or_default()
            .push(l.id);
        self.by_state
            .entry(l.state.as_u8())
            .or_default()
            .push(l.id);
    }

    fn index_remove(&mut self, l: &WisdomLineage) {
        self.by_candidate.remove(&l.wisdom_candidate_id);
        if let Some(v) = self.by_domain.get_mut(&l.envelope.domain) {
            v.retain(|&x| x != l.id);
        }
        if let Some(v) = self.by_state.get_mut(&(l.state.as_u8())) {
            v.retain(|&x| x != l.id);
        }
    }
}

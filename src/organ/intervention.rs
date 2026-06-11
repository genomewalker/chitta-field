use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

// ── Enums ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionType {
    ToolCall,
    MultiStepPlan,
    Delegation,
    Edit,
    Command,
}

impl ActionType {
    pub fn to_u8(self) -> u8 {
        match self {
            Self::ToolCall => 0,
            Self::MultiStepPlan => 1,
            Self::Delegation => 2,
            Self::Edit => 3,
            Self::Command => 4,
        }
    }
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::ToolCall,
            1 => Self::MultiStepPlan,
            2 => Self::Delegation,
            3 => Self::Edit,
            4 => Self::Command,
            _ => Self::ToolCall,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterventionStatus {
    Open,
    Succeeded,
    Failed,
    Partial,
    Aborted,
}

impl InterventionStatus {
    pub fn to_u8(self) -> u8 {
        match self {
            Self::Open => 0,
            Self::Succeeded => 1,
            Self::Failed => 2,
            Self::Partial => 3,
            Self::Aborted => 4,
        }
    }
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Open,
            1 => Self::Succeeded,
            2 => Self::Failed,
            3 => Self::Partial,
            4 => Self::Aborted,
            _ => Self::Aborted,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReversalCost {
    None,
    Low,
    Medium,
    High,
}

impl ReversalCost {
    pub fn to_u8(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Low => 1,
            Self::Medium => 2,
            Self::High => 3,
        }
    }
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::None,
            1 => Self::Low,
            2 => Self::Medium,
            3 => Self::High,
            _ => Self::None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservationKind {
    Stdout,
    Stderr,
    FileDiff,
    TestResult,
    EnvState,
    UserFeedback,
}

impl ObservationKind {
    pub fn to_u8(self) -> u8 {
        match self {
            Self::Stdout => 0,
            Self::Stderr => 1,
            Self::FileDiff => 2,
            Self::TestResult => 3,
            Self::EnvState => 4,
            Self::UserFeedback => 5,
        }
    }
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Stdout,
            1 => Self::Stderr,
            2 => Self::FileDiff,
            3 => Self::TestResult,
            4 => Self::EnvState,
            5 => Self::UserFeedback,
            _ => Self::Stdout,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttributionClass {
    MemoryRecallError,
    SourceTrustError,
    ProcedureError,
    ToolExecutionError,
    EnvironmentShift,
    HiddenPrecondition,
    AmbiguousState,
    GoalSpecError,
    UserOverride,
    ExternalNondeterminism,
}

impl AttributionClass {
    pub fn to_u8(self) -> u8 {
        match self {
            Self::MemoryRecallError => 0,
            Self::SourceTrustError => 1,
            Self::ProcedureError => 2,
            Self::ToolExecutionError => 3,
            Self::EnvironmentShift => 4,
            Self::HiddenPrecondition => 5,
            Self::AmbiguousState => 6,
            Self::GoalSpecError => 7,
            Self::UserOverride => 8,
            Self::ExternalNondeterminism => 9,
        }
    }
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::MemoryRecallError,
            1 => Self::SourceTrustError,
            2 => Self::ProcedureError,
            3 => Self::ToolExecutionError,
            4 => Self::EnvironmentShift,
            5 => Self::HiddenPrecondition,
            6 => Self::AmbiguousState,
            7 => Self::GoalSpecError,
            8 => Self::UserOverride,
            9 => Self::ExternalNondeterminism,
            _ => Self::ExternalNondeterminism,
        }
    }
}

// ── Records ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionRecord {
    pub id: u64,
    pub realm: String,
    pub session_id: String,
    pub task_id: Option<u64>,
    pub agent_id: String,
    pub domain: String,
    pub intent: String,
    pub action_type: ActionType,
    pub action_ref: String,
    pub preconditions: Vec<String>,
    pub expected_observables: Vec<String>,
    pub reversal_cost: ReversalCost,
    pub started_ms: i64,
    pub closed_ms: Option<i64>,
    pub status: InterventionStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationRecord {
    pub id: u64,
    pub intervention_id: u64,
    pub kind: ObservationKind,
    pub evidence_refs: Vec<u64>,
    pub summary: String,
    pub confidence: f32,
    pub timestamp_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionRecord {
    pub intervention_id: u64,
    pub primary_class: AttributionClass,
    pub secondary_class: Option<AttributionClass>,
    pub confidence_delta: f32,
    pub surprise_id: Option<u64>,
    pub debt_ids: Vec<u64>,
    pub source_memory_ids: Vec<u64>,
    pub skill_memory_ids: Vec<u64>,
    pub note: Option<String>,
    pub timestamp_ms: i64,
}

// ── Config & Stats ─────────────────────────────────────────────────────────

pub struct InterventionConfig {
    /// Interventions open longer than this are considered stale (default 30 min).
    pub stale_threshold_ms: i64,
}

impl Default for InterventionConfig {
    fn default() -> Self {
        Self { stale_threshold_ms: 1_800_000 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionStats {
    pub total_interventions: usize,
    pub open: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub partial: usize,
    pub aborted: usize,
    pub total_observations: usize,
    pub total_attributions: usize,
}

// ── Store ──────────────────────────────────────────────────────────────────

pub struct InterventionStore {
    next_intervention_id: u64,
    next_observation_id: u64,
    interventions: Vec<InterventionRecord>,
    observations: Vec<ObservationRecord>,
    attributions: Vec<AttributionRecord>,
    id_to_index: HashMap<u64, usize>,
    obs_by_intervention: HashMap<u64, Vec<usize>>,
    attr_by_intervention: HashMap<u64, usize>,
    open_interventions: HashSet<u64>,
    by_session: HashMap<String, Vec<u64>>,
    by_realm: HashMap<String, Vec<u64>>,
    #[allow(dead_code)]
    config: InterventionConfig,
}

impl InterventionStore {
    pub fn new() -> Self {
        Self {
            next_intervention_id: 1,
            next_observation_id: 1,
            interventions: Vec::new(),
            observations: Vec::new(),
            attributions: Vec::new(),
            id_to_index: HashMap::new(),
            obs_by_intervention: HashMap::new(),
            attr_by_intervention: HashMap::new(),
            open_interventions: HashSet::new(),
            by_session: HashMap::new(),
            by_realm: HashMap::new(),
            config: InterventionConfig::default(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn start_intervention(
        &mut self,
        realm: String,
        session_id: String,
        task_id: Option<u64>,
        agent_id: String,
        domain: String,
        intent: String,
        action_type: ActionType,
        action_ref: String,
        preconditions: Vec<String>,
        expected_observables: Vec<String>,
        reversal_cost: ReversalCost,
        now_ms: i64,
    ) -> u64 {
        let id = self.next_intervention_id;
        self.next_intervention_id += 1;
        self.insert_intervention(InterventionRecord {
            id,
            realm,
            session_id,
            task_id,
            agent_id,
            domain,
            intent,
            action_type,
            action_ref,
            preconditions,
            expected_observables,
            reversal_cost,
            started_ms: now_ms,
            closed_ms: None,
            status: InterventionStatus::Open,
        });
        id
    }

    pub fn replay_start(&mut self, record: InterventionRecord) {
        if record.id >= self.next_intervention_id {
            self.next_intervention_id = record.id + 1;
        }
        self.insert_intervention(record);
    }

    fn insert_intervention(&mut self, record: InterventionRecord) {
        let id = record.id;
        let idx = self.interventions.len();
        self.open_interventions.insert(id);
        self.by_session
            .entry(record.session_id.clone())
            .or_default()
            .push(id);
        self.by_realm
            .entry(record.realm.clone())
            .or_default()
            .push(id);
        self.id_to_index.insert(id, idx);
        self.interventions.push(record);
    }

    pub fn add_observation(
        &mut self,
        intervention_id: u64,
        kind: ObservationKind,
        evidence_refs: Vec<u64>,
        summary: String,
        confidence: f32,
        now_ms: i64,
    ) -> Option<u64> {
        if !self.id_to_index.contains_key(&intervention_id) {
            return None;
        }
        let id = self.next_observation_id;
        self.next_observation_id += 1;
        let obs_idx = self.observations.len();
        self.obs_by_intervention
            .entry(intervention_id)
            .or_default()
            .push(obs_idx);
        self.observations.push(ObservationRecord {
            id,
            intervention_id,
            kind,
            evidence_refs,
            summary,
            confidence,
            timestamp_ms: now_ms,
        });
        Some(id)
    }

    pub fn replay_observation(&mut self, obs: ObservationRecord) {
        if obs.id >= self.next_observation_id {
            self.next_observation_id = obs.id + 1;
        }
        let obs_idx = self.observations.len();
        self.obs_by_intervention
            .entry(obs.intervention_id)
            .or_default()
            .push(obs_idx);
        self.observations.push(obs);
    }

    pub fn close_intervention(
        &mut self,
        id: u64,
        status: InterventionStatus,
        now_ms: i64,
    ) -> bool {
        let Some(&idx) = self.id_to_index.get(&id) else { return false };
        self.interventions[idx].status = status;
        self.interventions[idx].closed_ms = Some(now_ms);
        self.open_interventions.remove(&id);
        true
    }

    pub fn replay_close(&mut self, id: u64, status: InterventionStatus, closed_ms: i64) {
        if let Some(&idx) = self.id_to_index.get(&id) {
            self.interventions[idx].status = status;
            self.interventions[idx].closed_ms = Some(closed_ms);
            self.open_interventions.remove(&id);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_attribution(
        &mut self,
        intervention_id: u64,
        primary_class: AttributionClass,
        secondary_class: Option<AttributionClass>,
        confidence_delta: f32,
        surprise_id: Option<u64>,
        debt_ids: Vec<u64>,
        source_memory_ids: Vec<u64>,
        skill_memory_ids: Vec<u64>,
        note: Option<String>,
        now_ms: i64,
    ) -> bool {
        if !self.id_to_index.contains_key(&intervention_id) {
            return false;
        }
        let attr_idx = self.attributions.len();
        self.attr_by_intervention.insert(intervention_id, attr_idx);
        self.attributions.push(AttributionRecord {
            intervention_id,
            primary_class,
            secondary_class,
            confidence_delta,
            surprise_id,
            debt_ids,
            source_memory_ids,
            skill_memory_ids,
            note,
            timestamp_ms: now_ms,
        });
        true
    }

    pub fn replay_attribution(&mut self, attr: AttributionRecord) {
        let intervention_id = attr.intervention_id;
        let attr_idx = self.attributions.len();
        self.attr_by_intervention.insert(intervention_id, attr_idx);
        self.attributions.push(attr);
    }

    pub fn get(&self, id: u64) -> Option<&InterventionRecord> {
        self.id_to_index.get(&id).map(|&i| &self.interventions[i])
    }

    pub fn get_observations(&self, intervention_id: u64) -> Vec<&ObservationRecord> {
        self.obs_by_intervention
            .get(&intervention_id)
            .map(|indices| indices.iter().map(|&i| &self.observations[i]).collect())
            .unwrap_or_default()
    }

    pub fn get_attribution(&self, intervention_id: u64) -> Option<&AttributionRecord> {
        self.attr_by_intervention
            .get(&intervention_id)
            .map(|&i| &self.attributions[i])
    }

    pub fn query(
        &self,
        realm: Option<&str>,
        session_id: Option<&str>,
        status: Option<InterventionStatus>,
        limit: usize,
    ) -> Vec<&InterventionRecord> {
        self.interventions
            .iter()
            .rev()
            .filter(|r| {
                realm.map_or(true, |re| r.realm == re)
                    && session_id.map_or(true, |s| r.session_id == s)
                    && status.map_or(true, |st| r.status == st)
            })
            .take(limit)
            .collect()
    }

    pub fn list_open(&self) -> Vec<&InterventionRecord> {
        self.open_interventions
            .iter()
            .filter_map(|id| self.id_to_index.get(id).map(|&i| &self.interventions[i]))
            .collect()
    }

    pub fn stale_open(&self, threshold_ms: i64, now_ms: i64) -> Vec<u64> {
        self.open_interventions
            .iter()
            .filter_map(|&id| {
                let idx = self.id_to_index.get(&id)?;
                let rec = &self.interventions[*idx];
                if now_ms - rec.started_ms > threshold_ms {
                    Some(id)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn stats(&self) -> InterventionStats {
        let mut succeeded = 0usize;
        let mut failed = 0usize;
        let mut partial = 0usize;
        let mut aborted = 0usize;
        for r in &self.interventions {
            match r.status {
                InterventionStatus::Open => {}
                InterventionStatus::Succeeded => succeeded += 1,
                InterventionStatus::Failed => failed += 1,
                InterventionStatus::Partial => partial += 1,
                InterventionStatus::Aborted => aborted += 1,
            }
        }
        InterventionStats {
            total_interventions: self.interventions.len(),
            open: self.open_interventions.len(),
            succeeded,
            failed,
            partial,
            aborted,
            total_observations: self.observations.len(),
            total_attributions: self.attributions.len(),
        }
    }
}

impl crate::organ::OrganApply for InterventionStore {
    /// Organ-owned WAL replay (THEORY.md §8 Phase 2). Consumes this organ's
    /// op variants; everything else passes through to the next organ or the
    /// central multi-structure match in apply_op.
    fn apply(&mut self, op: crate::ops::Op) -> Option<crate::ops::Op> {
        use crate::ops::Op;
        match op {
            Op::StartIntervention(s) => {
                use crate::organ::intervention::{ActionType, InterventionRecord, InterventionStatus, ReversalCost};
                self.replay_start(InterventionRecord {
                    id: s.id,
                    realm: s.realm,
                    session_id: s.session_id,
                    task_id: s.task_id,
                    agent_id: s.agent_id,
                    domain: s.domain,
                    intent: s.intent,
                    action_type: ActionType::from_u8(s.action_type),
                    action_ref: s.action_ref,
                    preconditions: s.preconditions,
                    expected_observables: s.expected_observables,
                    reversal_cost: ReversalCost::from_u8(s.reversal_cost),
                    started_ms: s.started_ms,
                    closed_ms: None,
                    status: InterventionStatus::Open,
                });
                    None
                }
            Op::AddObservation(o) => {
                use crate::organ::intervention::{ObservationKind, ObservationRecord};
                self.replay_observation(ObservationRecord {
                    id: o.id,
                    intervention_id: o.intervention_id,
                    kind: ObservationKind::from_u8(o.kind),
                    evidence_refs: o.evidence_refs,
                    summary: o.summary,
                    confidence: o.confidence,
                    timestamp_ms: o.timestamp_ms,
                });
                    None
                }
            Op::CloseIntervention(c) => {
                use crate::organ::intervention::InterventionStatus;
                self.replay_close(
                    c.intervention_id,
                    InterventionStatus::from_u8(c.status),
                    c.closed_ms,
                );
                    None
                }
            Op::RecordAttribution(a) => {
                use crate::organ::intervention::{AttributionClass, AttributionRecord};
                self.replay_attribution(AttributionRecord {
                    intervention_id: a.intervention_id,
                    primary_class: AttributionClass::from_u8(a.primary_class),
                    secondary_class: a.secondary_class.map(AttributionClass::from_u8),
                    confidence_delta: a.confidence_delta,
                    surprise_id: a.surprise_id,
                    debt_ids: a.debt_ids,
                    source_memory_ids: a.source_memory_ids,
                    skill_memory_ids: a.skill_memory_ids,
                    note: a.note,
                    timestamp_ms: a.timestamp_ms,
                });
                    None
                }
            other => Some(other),
        }
    }
}

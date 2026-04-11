use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

// ── Enums ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Active,
    Blocked,
    Completed,
    Failed,
    Abandoned,
}

impl TaskStatus {
    pub fn to_u8(self) -> u8 {
        match self {
            Self::Active    => 0,
            Self::Blocked   => 1,
            Self::Completed => 2,
            Self::Failed    => 3,
            Self::Abandoned => 4,
        }
    }
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Active,
            1 => Self::Blocked,
            2 => Self::Completed,
            3 => Self::Failed,
            4 => Self::Abandoned,
            _ => Self::Active,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DelegationStatus {
    Active,
    Completed,
    Recalled,
}

impl DelegationStatus {
    pub fn to_u8(self) -> u8 {
        match self {
            Self::Active    => 0,
            Self::Completed => 1,
            Self::Recalled  => 2,
        }
    }
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Active,
            1 => Self::Completed,
            2 => Self::Recalled,
            _ => Self::Active,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceKind {
    Observation,
    Artifact,
    Result,
    Analysis,
    UserFeedback,
}

impl EvidenceKind {
    pub fn to_u8(self) -> u8 {
        match self {
            Self::Observation  => 0,
            Self::Artifact     => 1,
            Self::Result       => 2,
            Self::Analysis     => 3,
            Self::UserFeedback => 4,
        }
    }
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Observation,
            1 => Self::Artifact,
            2 => Self::Result,
            3 => Self::Analysis,
            4 => Self::UserFeedback,
            _ => Self::Observation,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProbeStatus {
    Open,
    Answered,
    Dismissed,
}

impl ProbeStatus {
    pub fn to_u8(self) -> u8 {
        match self {
            Self::Open      => 0,
            Self::Answered  => 1,
            Self::Dismissed => 2,
        }
    }
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Open,
            1 => Self::Answered,
            2 => Self::Dismissed,
            _ => Self::Open,
        }
    }
}

// ── Record structs ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContract {
    pub id: u64,
    pub session_id: String,
    pub realm: String,
    pub goal: String,
    pub constraints: Vec<String>,
    pub acceptance_criteria: Vec<String>,
    pub priority: u8,               // 0=Low 1=Medium 2=High 3=Critical
    pub status: TaskStatus,
    pub parent_task_id: Option<u64>,
    pub intervention_ids: Vec<u64>, // linked Layer 7 interventions
    pub tags: Vec<String>,
    pub created_ms: i64,
    pub updated_ms: i64,
    pub deadline_ms: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegationEdge {
    pub id: u64,
    pub task_id: u64,
    pub from_agent: String,
    pub to_agent: String,
    pub delegated_at: i64,
    pub handoff_note: Option<String>,
    pub status: DelegationStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceLink {
    pub id: u64,
    pub task_id: u64,
    pub memory_id: u64,
    pub produced_by: String,
    pub evidence_kind: EvidenceKind,
    pub relevance: f32,
    pub created_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingProbe {
    pub id: u64,
    pub task_id: u64,
    pub question: String,
    pub expected_answerer: Option<String>,
    pub priority: u8,
    pub status: ProbeStatus,
    pub created_ms: i64,
    pub resolved_ms: Option<i64>,
    pub answer: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionCriterion {
    pub id: u64,
    pub task_id: u64,
    pub criterion: String,
    pub is_met: bool,
    pub checked_ms: Option<i64>,
    pub evidence_note: Option<String>,
}

// ── Full task view (returned by get_task_full) ─────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskFullView {
    pub task: TaskContract,
    pub delegations: Vec<DelegationEdge>,
    pub evidence: Vec<EvidenceLink>,
    pub probes: Vec<PendingProbe>,
    pub criteria: Vec<CompletionCriterion>,
}

// ── Stats ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentProtocolStats {
    pub total_tasks: usize,
    pub active_tasks: usize,
    pub blocked_tasks: usize,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub abandoned_tasks: usize,
    pub total_delegations: usize,
    pub total_evidence_links: usize,
    pub total_probes: usize,
    pub open_probes: usize,
    pub total_criteria: usize,
    pub criteria_met: usize,
}

// ── Store ──────────────────────────────────────────────────────────────────

pub struct AgentProtocolStore {
    next_task_id: u64,
    next_delegation_id: u64,
    next_evidence_id: u64,
    next_probe_id: u64,
    next_criterion_id: u64,

    tasks: Vec<TaskContract>,
    id_to_task: HashMap<u64, usize>,
    active_tasks: HashSet<u64>,
    tasks_by_session: HashMap<String, Vec<u64>>,
    tasks_by_realm: HashMap<String, Vec<u64>>,

    delegations: Vec<DelegationEdge>,
    delegations_by_task: HashMap<u64, Vec<usize>>,

    evidence: Vec<EvidenceLink>,
    evidence_by_task: HashMap<u64, Vec<usize>>,
    // (task_id, memory_id) → evidence_id for idempotency
    evidence_dedup: HashMap<(u64, u64), u64>,

    probes: Vec<PendingProbe>,
    probes_by_task: HashMap<u64, Vec<usize>>,
    open_probes: HashSet<u64>,

    criteria: Vec<CompletionCriterion>,
    criteria_by_task: HashMap<u64, Vec<usize>>,
    // (task_id, criterion_text) → criterion index for upsert
    criteria_dedup: HashMap<(u64, String), usize>,
}

impl AgentProtocolStore {
    pub fn new() -> Self {
        Self {
            next_task_id: 1,
            next_delegation_id: 1,
            next_evidence_id: 1,
            next_probe_id: 1,
            next_criterion_id: 1,
            tasks: Vec::new(),
            id_to_task: HashMap::new(),
            active_tasks: HashSet::new(),
            tasks_by_session: HashMap::new(),
            tasks_by_realm: HashMap::new(),
            delegations: Vec::new(),
            delegations_by_task: HashMap::new(),
            evidence: Vec::new(),
            evidence_by_task: HashMap::new(),
            evidence_dedup: HashMap::new(),
            probes: Vec::new(),
            probes_by_task: HashMap::new(),
            open_probes: HashSet::new(),
            criteria: Vec::new(),
            criteria_by_task: HashMap::new(),
            criteria_dedup: HashMap::new(),
        }
    }

    // ── Task registration ──────────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    pub fn register_task(
        &mut self,
        goal: String,
        constraints: Vec<String>,
        acceptance_criteria: Vec<String>,
        realm: String,
        session_id: String,
        priority: u8,
        parent_task_id: Option<u64>,
        deadline_ms: Option<i64>,
        tags: Vec<String>,
        now_ms: i64,
    ) -> u64 {
        let id = self.next_task_id;
        self.next_task_id += 1;
        self.insert_task(TaskContract {
            id,
            session_id,
            realm,
            goal,
            constraints,
            acceptance_criteria,
            priority,
            status: TaskStatus::Active,
            parent_task_id,
            intervention_ids: Vec::new(),
            tags,
            created_ms: now_ms,
            updated_ms: now_ms,
            deadline_ms,
        });
        id
    }

    pub fn replay_register_task(&mut self, task: TaskContract) {
        if task.id >= self.next_task_id {
            self.next_task_id = task.id + 1;
        }
        self.insert_task(task);
    }

    fn insert_task(&mut self, task: TaskContract) {
        let id = task.id;
        let idx = self.tasks.len();
        self.active_tasks.insert(id);
        self.tasks_by_session
            .entry(task.session_id.clone())
            .or_default()
            .push(id);
        self.tasks_by_realm
            .entry(task.realm.clone())
            .or_default()
            .push(id);
        self.id_to_task.insert(id, idx);
        self.tasks.push(task);
    }

    // ── Task status update ─────────────────────────────────────────────────

    pub fn update_task(
        &mut self,
        id: u64,
        status: Option<TaskStatus>,
        add_intervention_id: Option<u64>,
        add_tag: Option<String>,
        now_ms: i64,
    ) -> bool {
        let Some(&idx) = self.id_to_task.get(&id) else { return false };
        if let Some(s) = status {
            self.tasks[idx].status = s;
            match s {
                TaskStatus::Active | TaskStatus::Blocked => {}
                _ => { self.active_tasks.remove(&id); }
            }
        }
        if let Some(iid) = add_intervention_id {
            if !self.tasks[idx].intervention_ids.contains(&iid) {
                self.tasks[idx].intervention_ids.push(iid);
            }
        }
        if let Some(tag) = add_tag {
            if !self.tasks[idx].tags.contains(&tag) {
                self.tasks[idx].tags.push(tag);
            }
        }
        self.tasks[idx].updated_ms = now_ms;
        true
    }

    pub fn replay_update_task(
        &mut self,
        id: u64,
        status: u8,
        add_intervention_id: Option<u64>,
        add_tag: Option<String>,
        updated_ms: i64,
    ) {
        self.update_task(
            id,
            Some(TaskStatus::from_u8(status)),
            add_intervention_id,
            add_tag,
            updated_ms,
        );
    }

    // ── Delegation ─────────────────────────────────────────────────────────

    pub fn add_delegation(
        &mut self,
        task_id: u64,
        from_agent: String,
        to_agent: String,
        handoff_note: Option<String>,
        now_ms: i64,
    ) -> Option<u64> {
        if !self.id_to_task.contains_key(&task_id) {
            return None;
        }
        let id = self.next_delegation_id;
        self.next_delegation_id += 1;
        let del_idx = self.delegations.len();
        self.delegations_by_task
            .entry(task_id)
            .or_default()
            .push(del_idx);
        self.delegations.push(DelegationEdge {
            id,
            task_id,
            from_agent,
            to_agent,
            delegated_at: now_ms,
            handoff_note,
            status: DelegationStatus::Active,
        });
        // Update task timestamp
        if let Some(&idx) = self.id_to_task.get(&task_id) {
            self.tasks[idx].updated_ms = now_ms;
        }
        Some(id)
    }

    pub fn replay_delegation(&mut self, del: DelegationEdge) {
        if del.id >= self.next_delegation_id {
            self.next_delegation_id = del.id + 1;
        }
        let del_idx = self.delegations.len();
        self.delegations_by_task
            .entry(del.task_id)
            .or_default()
            .push(del_idx);
        self.delegations.push(del);
    }

    // ── Evidence linking ───────────────────────────────────────────────────

    pub fn link_evidence(
        &mut self,
        task_id: u64,
        memory_id: u64,
        produced_by: String,
        evidence_kind: EvidenceKind,
        relevance: f32,
        now_ms: i64,
    ) -> Option<u64> {
        if !self.id_to_task.contains_key(&task_id) {
            return None;
        }
        // Idempotent: same (task_id, memory_id) returns existing evidence ID
        if let Some(&existing_id) = self.evidence_dedup.get(&(task_id, memory_id)) {
            return Some(existing_id);
        }
        let id = self.next_evidence_id;
        self.next_evidence_id += 1;
        let ev_idx = self.evidence.len();
        self.evidence_by_task
            .entry(task_id)
            .or_default()
            .push(ev_idx);
        self.evidence_dedup.insert((task_id, memory_id), id);
        self.evidence.push(EvidenceLink {
            id,
            task_id,
            memory_id,
            produced_by,
            evidence_kind,
            relevance,
            created_ms: now_ms,
        });
        if let Some(&idx) = self.id_to_task.get(&task_id) {
            self.tasks[idx].updated_ms = now_ms;
        }
        Some(id)
    }

    pub fn replay_evidence(&mut self, ev: EvidenceLink) {
        if ev.id >= self.next_evidence_id {
            self.next_evidence_id = ev.id + 1;
        }
        let ev_idx = self.evidence.len();
        self.evidence_by_task
            .entry(ev.task_id)
            .or_default()
            .push(ev_idx);
        self.evidence_dedup.insert((ev.task_id, ev.memory_id), ev.id);
        self.evidence.push(ev);
    }

    // ── Pending probes ─────────────────────────────────────────────────────

    pub fn add_probe(
        &mut self,
        task_id: u64,
        question: String,
        expected_answerer: Option<String>,
        priority: u8,
        now_ms: i64,
    ) -> Option<u64> {
        if !self.id_to_task.contains_key(&task_id) {
            return None;
        }
        let id = self.next_probe_id;
        self.next_probe_id += 1;
        let probe_idx = self.probes.len();
        self.probes_by_task
            .entry(task_id)
            .or_default()
            .push(probe_idx);
        self.open_probes.insert(id);
        self.probes.push(PendingProbe {
            id,
            task_id,
            question,
            expected_answerer,
            priority,
            status: ProbeStatus::Open,
            created_ms: now_ms,
            resolved_ms: None,
            answer: None,
        });
        if let Some(&idx) = self.id_to_task.get(&task_id) {
            self.tasks[idx].updated_ms = now_ms;
        }
        Some(id)
    }

    pub fn resolve_probe(
        &mut self,
        probe_id: u64,
        status: ProbeStatus,
        answer: Option<String>,
        now_ms: i64,
    ) -> bool {
        let Some(probe) = self.probes.iter_mut().find(|p| p.id == probe_id) else {
            return false;
        };
        probe.status = status;
        probe.resolved_ms = Some(now_ms);
        probe.answer = answer;
        self.open_probes.remove(&probe_id);
        true
    }

    pub fn replay_probe(&mut self, probe: PendingProbe) {
        if probe.id >= self.next_probe_id {
            self.next_probe_id = probe.id + 1;
        }
        let probe_idx = self.probes.len();
        self.probes_by_task
            .entry(probe.task_id)
            .or_default()
            .push(probe_idx);
        if probe.status == ProbeStatus::Open {
            self.open_probes.insert(probe.id);
        }
        self.probes.push(probe);
    }

    pub fn replay_resolve_probe(
        &mut self,
        probe_id: u64,
        status: u8,
        answer: Option<String>,
        resolved_ms: i64,
    ) {
        self.resolve_probe(probe_id, ProbeStatus::from_u8(status), answer, resolved_ms);
    }

    // ── Completion criteria ────────────────────────────────────────────────

    /// Upsert: creates new or updates existing criterion by (task_id, criterion_text).
    pub fn set_criterion(
        &mut self,
        task_id: u64,
        criterion: String,
        is_met: bool,
        evidence_note: Option<String>,
        now_ms: i64,
    ) -> Option<u64> {
        if !self.id_to_task.contains_key(&task_id) {
            return None;
        }
        let key = (task_id, criterion.clone());
        if let Some(&existing_idx) = self.criteria_dedup.get(&key) {
            self.criteria[existing_idx].is_met = is_met;
            self.criteria[existing_idx].checked_ms = Some(now_ms);
            self.criteria[existing_idx].evidence_note = evidence_note;
            let id = self.criteria[existing_idx].id;
            if let Some(&tidx) = self.id_to_task.get(&task_id) {
                self.tasks[tidx].updated_ms = now_ms;
            }
            return Some(id);
        }
        let id = self.next_criterion_id;
        self.next_criterion_id += 1;
        let crit_idx = self.criteria.len();
        self.criteria_by_task
            .entry(task_id)
            .or_default()
            .push(crit_idx);
        self.criteria_dedup.insert(key, crit_idx);
        self.criteria.push(CompletionCriterion {
            id,
            task_id,
            criterion,
            is_met,
            checked_ms: Some(now_ms),
            evidence_note,
        });
        if let Some(&tidx) = self.id_to_task.get(&task_id) {
            self.tasks[tidx].updated_ms = now_ms;
        }
        Some(id)
    }

    pub fn replay_criterion(&mut self, crit: CompletionCriterion) {
        if crit.id >= self.next_criterion_id {
            self.next_criterion_id = crit.id + 1;
        }
        let key = (crit.task_id, crit.criterion.clone());
        if let Some(&existing_idx) = self.criteria_dedup.get(&key) {
            self.criteria[existing_idx].is_met = crit.is_met;
            self.criteria[existing_idx].checked_ms = crit.checked_ms;
            self.criteria[existing_idx].evidence_note = crit.evidence_note;
            return;
        }
        let crit_idx = self.criteria.len();
        self.criteria_by_task
            .entry(crit.task_id)
            .or_default()
            .push(crit_idx);
        self.criteria_dedup.insert(key, crit_idx);
        self.criteria.push(crit);
    }

    // ── Queries ────────────────────────────────────────────────────────────

    pub fn get_task(&self, id: u64) -> Option<&TaskContract> {
        self.id_to_task.get(&id).map(|&idx| &self.tasks[idx])
    }

    pub fn get_task_full(&self, id: u64) -> Option<TaskFullView> {
        let task = self.get_task(id)?.clone();
        let delegations = self
            .delegations_by_task
            .get(&id)
            .map(|idxs| idxs.iter().map(|&i| self.delegations[i].clone()).collect())
            .unwrap_or_default();
        let evidence = self
            .evidence_by_task
            .get(&id)
            .map(|idxs| idxs.iter().map(|&i| self.evidence[i].clone()).collect())
            .unwrap_or_default();
        let probes = self
            .probes_by_task
            .get(&id)
            .map(|idxs| idxs.iter().map(|&i| self.probes[i].clone()).collect())
            .unwrap_or_default();
        let criteria = self
            .criteria_by_task
            .get(&id)
            .map(|idxs| idxs.iter().map(|&i| self.criteria[i].clone()).collect())
            .unwrap_or_default();
        Some(TaskFullView { task, delegations, evidence, probes, criteria })
    }

    pub fn query_tasks(
        &self,
        realm: Option<&str>,
        session_id: Option<&str>,
        status: Option<TaskStatus>,
        priority: Option<u8>,
        limit: usize,
    ) -> Vec<&TaskContract> {
        let mut results: Vec<&TaskContract> = self
            .tasks
            .iter()
            .filter(|t| {
                realm.map_or(true, |r| t.realm == r)
                    && session_id.map_or(true, |s| t.session_id == s)
                    && status.map_or(true, |st| t.status == st)
                    && priority.map_or(true, |p| t.priority == p)
            })
            .collect();
        // Sort: active first, then by priority desc, then by created_ms desc
        results.sort_by(|a, b| {
            let a_active = matches!(a.status, TaskStatus::Active | TaskStatus::Blocked);
            let b_active = matches!(b.status, TaskStatus::Active | TaskStatus::Blocked);
            b_active.cmp(&a_active)
                .then(b.priority.cmp(&a.priority))
                .then(b.created_ms.cmp(&a.created_ms))
        });
        results.truncate(limit);
        results
    }

    /// Returns task IDs where all criteria are met — for auto-completion by subconscious.
    pub fn tasks_with_all_criteria_met(&self) -> Vec<u64> {
        self.active_tasks
            .iter()
            .copied()
            .filter(|&tid| {
                let Some(idxs) = self.criteria_by_task.get(&tid) else { return false };
                if idxs.is_empty() { return false }
                idxs.iter().all(|&i| self.criteria[i].is_met)
            })
            .collect()
    }

    pub fn stats(&self) -> AgentProtocolStats {
        let mut s = AgentProtocolStats {
            total_tasks: self.tasks.len(),
            total_delegations: self.delegations.len(),
            total_evidence_links: self.evidence.len(),
            total_probes: self.probes.len(),
            open_probes: self.open_probes.len(),
            total_criteria: self.criteria.len(),
            criteria_met: self.criteria.iter().filter(|c| c.is_met).count(),
            ..Default::default()
        };
        for t in &self.tasks {
            match t.status {
                TaskStatus::Active    => s.active_tasks += 1,
                TaskStatus::Blocked   => s.blocked_tasks += 1,
                TaskStatus::Completed => s.completed_tasks += 1,
                TaskStatus::Failed    => s.failed_tasks += 1,
                TaskStatus::Abandoned => s.abandoned_tasks += 1,
            }
        }
        s
    }
}

impl Default for AgentProtocolStore {
    fn default() -> Self { Self::new() }
}

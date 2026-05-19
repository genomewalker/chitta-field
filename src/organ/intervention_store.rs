/// CEC Phase 9 — Active Memory: Executor Pathway.
///
/// Converts CEC from a pure read path into a control system. Motifs and refutation
/// results compile into typed InterventionPolicies that fire guarded actions at
/// turn-start or session-end. Shadow evaluation gates all promotions; the refutation
/// ledger is the adversarial check that prevents Q-value self-reinforcement.
use serde::{Serialize, Deserialize};
use crate::organ::refutation_ledger::RefutationLedger;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterventionKind {
    /// File a task into the task registry (minimal viable executor).
    OpenTask { title: String, description: String },
    /// Inject a warning at the next turn start (extends Phase 3 hook).
    TurnInjection { message: String, priority: u8 },
    /// Structured guard: check precondition, optionally rewrite or fall back.
    GuardPolicy {
        precondition: String,
        rewrite: Option<String>,
        fallback: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionPolicy {
    pub id:             u32,
    pub source_rule_id: u32,
    pub kind:           InterventionKind,
    pub active:         bool,
    /// Events observed while in shadow mode (must reach SHADOW_MIN before promotion).
    pub shadow_events:  u32,
    /// Cumulative Q-value delta observed during shadow period.
    pub shadow_lift:    f32,
    pub created_ts:     i64,
    pub last_fired_ts:  i64,
    pub fire_count:     u32,
}

impl InterventionPolicy {
    fn eligible_for_promotion(&self) -> bool {
        !self.active
            && self.shadow_events >= SHADOW_MIN
            && self.shadow_lift > LIFT_THRESHOLD
    }
}

const SHADOW_MIN:       u32 = 20;
const LIFT_THRESHOLD:   f32 = 0.15;
const MAX_REFUT_RATIO:  f32 = 0.3;    // suspend policy if source rule is this uncertain

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InterventionStore {
    pub policies: Vec<InterventionPolicy>,
    id_alloc:     u32,
}

impl InterventionStore {
    pub fn new() -> Self { Self::default() }

    /// Register a new policy in shadow mode (not yet active).
    pub fn propose(&mut self, source_rule_id: u32, kind: InterventionKind, ts: i64) -> u32 {
        let id = self.id_alloc;
        self.id_alloc += 1;
        self.policies.push(InterventionPolicy {
            id,
            source_rule_id,
            kind,
            active:         false,
            shadow_events:  0,
            shadow_lift:    0.0,
            created_ts:     ts,
            last_fired_ts:  0,
            fire_count:     0,
        });
        id
    }

    /// Record a Q-value delta observation during shadow evaluation.
    pub fn shadow_observe(&mut self, policy_id: u32, q_delta: f32) {
        if let Some(p) = self.policies.iter_mut().find(|p| p.id == policy_id) {
            if !p.active {
                p.shadow_events += 1;
                p.shadow_lift += q_delta;
            }
        }
    }

    /// Promote all eligible shadow policies. Returns ids of newly-active policies.
    pub fn promote_eligible(&mut self) -> Vec<u32> {
        let mut promoted = Vec::new();
        for p in self.policies.iter_mut() {
            if p.eligible_for_promotion() {
                p.active = true;
                promoted.push(p.id);
            }
        }
        promoted
    }

    /// Suspend policies whose source rule is being refuted above MAX_REFUT_RATIO.
    /// Returns ids of demoted policies.
    pub fn auto_demote_drifted(&mut self, ledger: &RefutationLedger) -> Vec<u32> {
        let mut demoted = Vec::new();
        for p in self.policies.iter_mut() {
            if !p.active { continue; }
            if ledger.refute_ratio_for_rule(p.source_rule_id) > MAX_REFUT_RATIO {
                p.active = false;
                demoted.push(p.id);
            }
        }
        demoted
    }

    /// Return active policies whose source rule is credible (<MAX_REFUT_RATIO).
    /// These are safe to fire. Critical invariant: executor gated behind refutation ledger.
    pub fn fire_active<'a>(&'a self, ledger: &RefutationLedger) -> Vec<&'a InterventionPolicy> {
        self.policies.iter()
            .filter(|p| p.active && ledger.refute_ratio_for_rule(p.source_rule_id) < MAX_REFUT_RATIO)
            .collect()
    }

    /// Stats for the executor_flush tool response.
    pub fn stats_json(&self) -> String {
        let active = self.policies.iter().filter(|p| p.active).count();
        let shadow = self.policies.iter().filter(|p| !p.active && p.shadow_events > 0).count();
        let total_fired: u32 = self.policies.iter().map(|p| p.fire_count).sum();
        format!(
            "{{\"total\":{},\"active\":{},\"shadow\":{},\"total_fired\":{}}}",
            self.policies.len(), active, shadow, total_fired
        )
    }

    pub fn list_json(&self, active_only: bool) -> String {
        let rows: Vec<String> = self.policies.iter()
            .filter(|p| !active_only || p.active)
            .map(|p| {
                let kind = match &p.kind {
                    InterventionKind::OpenTask { title, .. } => format!("OpenTask:{title}"),
                    InterventionKind::TurnInjection { message, .. } => format!("TurnInjection:{}", &message[..message.len().min(40)]),
                    InterventionKind::GuardPolicy { precondition, .. } => format!("GuardPolicy:{precondition}"),
                };
                format!(
                    "{{\"id\":{},\"rule\":{},\"active\":{},\"kind\":\"{}\",\"shadow_events\":{},\"lift\":{:.3},\"fires\":{}}}",
                    p.id, p.source_rule_id, p.active, kind, p.shadow_events, p.shadow_lift, p.fire_count
                )
            })
            .collect();
        format!("[{}]", rows.join(","))
    }
}

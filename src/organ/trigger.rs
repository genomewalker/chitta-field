//! Trigger Tissue — prospective memory as event-driven automata.
//!
//! Implements "remembering to remember": triggers arm on creation, accumulate
//! tension from partial condition matches, and self-fire when conditions are met
//! or tension exceeds threshold.
//!
//! References:
//!   - Einstein & McDaniel (1990). Normal aging and prospective memory.
//!   - GPT-5.4 + Opus brainstorm (2026-04-11) — tension accumulation model

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TriggerStatus {
    Armed,
    Fired,
    Expired,
    Inhibited,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerCondition {
    ConstraintMatch {
        subject: Option<String>,
        predicate: Option<String>,
        object: Option<String>,
    },
    TimeAfter(i64),
    EventMatch {
        domain: String,
        kind: String,
    },
    AllOf(Vec<TriggerCondition>),
    AnyOf(Vec<TriggerCondition>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerAction {
    InjectMemory(String),
    EmitEvent {
        domain: String,
        kind: String,
        payload: String,
    },
    RememberFact {
        subject: String,
        predicate: String,
        object: String,
    },
    Notify(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerAutomaton {
    pub id: u64,
    pub name: String,
    pub condition: TriggerCondition,
    pub action: TriggerAction,
    pub deadline_ms: i64,
    pub tension: f32,
    pub tension_threshold: f32,
    pub gain: f32,
    pub status: TriggerStatus,
    pub created_ms: i64,
    pub fired_ms: i64,
    pub inhibited_by: Vec<u64>,
    pub source_session: Option<String>,
    pub realm: String,
}

#[derive(Debug, Clone)]
pub struct FireResult {
    pub trigger_id: u64,
    pub action: TriggerAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerStore {
    next_id: u64,
    triggers: Vec<TriggerAutomaton>,
    id_to_index: HashMap<u64, usize>,
}

impl TriggerStore {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            triggers: Vec::new(),
            id_to_index: HashMap::new(),
        }
    }

    pub fn count_armed(&self) -> usize {
        self.triggers.iter().filter(|t| t.status == TriggerStatus::Armed).count()
    }

    pub fn add_trigger(
        &mut self,
        name: String,
        condition: TriggerCondition,
        action: TriggerAction,
        deadline_ms: i64,
        tension_threshold: f32,
        gain: f32,
        realm: String,
        source_session: Option<String>,
        now_ms: i64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let idx = self.triggers.len();
        self.triggers.push(TriggerAutomaton {
            id,
            name,
            condition,
            action,
            deadline_ms,
            tension: 0.0,
            tension_threshold,
            gain,
            status: TriggerStatus::Armed,
            created_ms: now_ms,
            fired_ms: 0,
            inhibited_by: Vec::new(),
            source_session,
            realm,
        });
        self.id_to_index.insert(id, idx);
        id
    }

    pub fn replay_add(&mut self, trigger: TriggerAutomaton) {
        let id = trigger.id;
        if id >= self.next_id {
            self.next_id = id + 1;
        }
        let idx = self.triggers.len();
        self.triggers.push(trigger);
        self.id_to_index.insert(id, idx);
    }

    pub fn get(&self, id: u64) -> Option<&TriggerAutomaton> {
        self.id_to_index.get(&id).map(|&idx| &self.triggers[idx])
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut TriggerAutomaton> {
        self.id_to_index.get(&id).copied().map(move |idx| &mut self.triggers[idx])
    }

    pub fn list_armed(&self) -> Vec<&TriggerAutomaton> {
        self.triggers.iter().filter(|t| t.status == TriggerStatus::Armed).collect()
    }

    pub fn list_all(&self) -> &[TriggerAutomaton] {
        &self.triggers
    }

    pub fn fire(&mut self, id: u64, now_ms: i64) -> Option<FireResult> {
        let trigger = self.get_mut(id)?;
        if trigger.status != TriggerStatus::Armed {
            return None;
        }
        trigger.status = TriggerStatus::Fired;
        trigger.fired_ms = now_ms;
        Some(FireResult {
            trigger_id: id,
            action: trigger.action.clone(),
        })
    }

    pub fn dismiss(&mut self, id: u64, now_ms: i64) -> bool {
        if let Some(trigger) = self.get_mut(id) {
            if trigger.status == TriggerStatus::Armed {
                trigger.status = TriggerStatus::Expired;
                trigger.fired_ms = now_ms;
                return true;
            }
        }
        false
    }

    pub fn add_tension(&mut self, id: u64, amount: f32) -> Option<bool> {
        let trigger = self.get_mut(id)?;
        if trigger.status != TriggerStatus::Armed {
            return None;
        }
        trigger.tension = (trigger.tension + amount).min(1.0);
        Some(trigger.tension >= trigger.tension_threshold)
    }

    /// Evaluate time-based triggers. Returns IDs of triggers that should fire.
    pub fn evaluate_time_triggers(&self, now_ms: i64) -> Vec<u64> {
        let mut ready = Vec::new();
        for t in &self.triggers {
            if t.status != TriggerStatus::Armed {
                continue;
            }
            // Check deadline expiry
            if t.deadline_ms > 0 && now_ms >= t.deadline_ms {
                ready.push(t.id);
                continue;
            }
            // Check TimeAfter condition
            if let TriggerCondition::TimeAfter(after_ms) = &t.condition {
                if now_ms >= *after_ms {
                    ready.push(t.id);
                }
            }
            // Check tension threshold
            if t.tension >= t.tension_threshold {
                ready.push(t.id);
            }
        }
        ready
    }

    /// Expire armed triggers past their deadline without firing.
    pub fn expire_overdue(&mut self, now_ms: i64) -> usize {
        let mut count = 0;
        for t in &mut self.triggers {
            if t.status == TriggerStatus::Armed && t.deadline_ms > 0 && now_ms > t.deadline_ms {
                t.status = TriggerStatus::Expired;
                count += 1;
            }
        }
        count
    }

    pub fn replay_update_status(&mut self, id: u64, status: TriggerStatus, fired_ms: i64) {
        if let Some(trigger) = self.get_mut(id) {
            trigger.status = status;
            if fired_ms > 0 {
                trigger.fired_ms = fired_ms;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_fire() {
        let mut store = TriggerStore::new();
        let id = store.add_trigger(
            "test".into(),
            TriggerCondition::TimeAfter(1000),
            TriggerAction::Notify("hello".into()),
            0, 0.8, 0.5, "global".into(), None, 100,
        );
        assert_eq!(store.count_armed(), 1);

        let result = store.fire(id, 1000).unwrap();
        assert_eq!(result.trigger_id, id);
        assert_eq!(store.count_armed(), 0);

        // Can't fire twice
        assert!(store.fire(id, 1001).is_none());
    }

    #[test]
    fn test_tension_accumulation() {
        let mut store = TriggerStore::new();
        let id = store.add_trigger(
            "tension-test".into(),
            TriggerCondition::EventMatch { domain: "git".into(), kind: "push".into() },
            TriggerAction::Notify("run tests".into()),
            0, 0.8, 0.5, "global".into(), None, 100,
        );

        assert_eq!(store.add_tension(id, 0.3), Some(false));
        assert_eq!(store.add_tension(id, 0.3), Some(false));
        assert_eq!(store.add_tension(id, 0.3), Some(true)); // 0.9 >= 0.8
    }

    #[test]
    fn test_time_evaluation() {
        let mut store = TriggerStore::new();
        store.add_trigger(
            "timer".into(),
            TriggerCondition::TimeAfter(5000),
            TriggerAction::Notify("time's up".into()),
            0, 0.8, 0.5, "global".into(), None, 100,
        );

        assert!(store.evaluate_time_triggers(4999).is_empty());
        assert_eq!(store.evaluate_time_triggers(5000).len(), 1);
    }

    #[test]
    fn test_dismiss() {
        let mut store = TriggerStore::new();
        let id = store.add_trigger(
            "dismiss-test".into(),
            TriggerCondition::TimeAfter(5000),
            TriggerAction::Notify("nope".into()),
            0, 0.8, 0.5, "global".into(), None, 100,
        );

        assert!(store.dismiss(id, 200));
        assert_eq!(store.count_armed(), 0);
        assert_eq!(store.get(id).unwrap().status, TriggerStatus::Expired);
    }
}

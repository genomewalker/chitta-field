use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum TaskStatus {
    Pending,
    Running,
    Paused,
    Completed,
    Failed,
}

impl TaskStatus {
    pub fn from_str(s: &str) -> Self {
        match s {
            "running" | "start" => Self::Running,
            "paused" | "pause" => Self::Paused,
            "completed" | "complete" => Self::Completed,
            "failed" | "fail" => Self::Failed,
            _ => Self::Pending,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Paused => "paused",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TaskRecord {
    pub task_id: String,
    pub kind: String,
    pub status: TaskStatus,
    pub payload_json: String,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub fencing_token: u64,
}

#[derive(Debug, Default)]
pub struct TaskRegistry {
    tasks: HashMap<String, TaskRecord>,
}

impl TaskRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create(
        &mut self,
        task_id: String,
        kind: String,
        payload_json: String,
        now_ms: i64,
        fencing_token: u64,
    ) {
        // Fencing: do not overwrite a record with a higher (newer) token.
        if let Some(existing) = self.tasks.get(&task_id) {
            if existing.fencing_token >= fencing_token {
                return;
            }
        }
        self.tasks.insert(
            task_id.clone(),
            TaskRecord {
                task_id,
                kind,
                status: TaskStatus::Pending,
                payload_json,
                created_at_ms: now_ms,
                updated_at_ms: now_ms,
                fencing_token,
            },
        );
    }

    pub fn transition(
        &mut self,
        task_id: &str,
        new_status: &str,
        now_ms: i64,
        fencing_token: u64,
    ) -> bool {
        if let Some(t) = self.tasks.get_mut(task_id) {
            if fencing_token > 0 && fencing_token < t.fencing_token {
                return false; // stale writer: reject
            }
            // Same token: use timestamp as tie-breaker (last-write-wins at ms granularity)
            if fencing_token > 0 && fencing_token == t.fencing_token && now_ms <= t.updated_at_ms {
                return false; // concurrent write with same token but older/equal timestamp
            }
            t.status = TaskStatus::from_str(new_status);
            t.updated_at_ms = now_ms;
            t.fencing_token = fencing_token;
            true
        } else {
            false
        }
    }

    pub fn get(&self, task_id: &str) -> Option<&TaskRecord> {
        self.tasks.get(task_id)
    }

    pub fn list_by_kind(&self, kind: &str) -> Vec<&TaskRecord> {
        self.tasks.values().filter(|t| t.kind == kind).collect()
    }

    pub fn list_active(&self) -> Vec<&TaskRecord> {
        self.tasks
            .values()
            .filter(|t| {
                matches!(
                    t.status,
                    TaskStatus::Pending | TaskStatus::Running | TaskStatus::Paused
                )
            })
            .collect()
    }

    pub fn list_all(&self) -> Vec<&TaskRecord> {
        self.tasks.values().collect()
    }

    pub fn update_payload(&mut self, task_id: &str, payload_json: String, now_ms: i64) -> bool {
        if let Some(t) = self.tasks.get_mut(task_id) {
            // Reject stale payload updates (strictly older timestamp = concurrent writer lost the race)
            if now_ms < t.updated_at_ms {
                return false;
            }
            t.payload_json = payload_json;
            t.updated_at_ms = now_ms;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_registry() -> TaskRegistry {
        let mut r = TaskRegistry::default();
        r.create("t1".to_string(), "job".to_string(), "{}".to_string(), 1000, 1);
        r
    }

    #[test]
    fn test_valid_transition_accepted() {
        let mut r = make_registry();
        assert!(r.transition("t1", "running", 2000, 1));
        assert_eq!(r.get("t1").unwrap().status, TaskStatus::Running);
    }

    #[test]
    fn test_stale_token_rejected() {
        let mut r = make_registry();
        // Advance to token=5
        r.transition("t1", "running", 2000, 5);
        // Stale writer with token=3 must be rejected
        assert!(!r.transition("t1", "completed", 3000, 3));
        assert_eq!(r.get("t1").unwrap().status, TaskStatus::Running);
    }

    #[test]
    fn test_same_token_older_timestamp_rejected() {
        let mut r = make_registry();
        r.transition("t1", "running", 2000, 2);
        // Same token, same timestamp — tie-break rejects it
        assert!(!r.transition("t1", "completed", 2000, 2));
        assert_eq!(r.get("t1").unwrap().status, TaskStatus::Running);
    }

    #[test]
    fn test_same_token_newer_timestamp_accepted() {
        let mut r = make_registry();
        r.transition("t1", "running", 2000, 2);
        // Same token, strictly newer timestamp — accepted
        assert!(r.transition("t1", "completed", 2001, 2));
        assert_eq!(r.get("t1").unwrap().status, TaskStatus::Completed);
    }

    #[test]
    fn test_higher_token_overwrites_regardless_of_timestamp() {
        let mut r = make_registry();
        r.transition("t1", "running", 9000, 5);
        // Higher token, older timestamp — still accepted (token takes priority)
        assert!(r.transition("t1", "completed", 1000, 6));
        assert_eq!(r.get("t1").unwrap().status, TaskStatus::Completed);
    }

    #[test]
    fn test_zero_token_bypasses_fencing() {
        let mut r = make_registry();
        // token=0 disables fencing checks
        assert!(r.transition("t1", "failed", 5000, 0));
        assert_eq!(r.get("t1").unwrap().status, TaskStatus::Failed);
    }

    #[test]
    fn test_transition_unknown_task_returns_false() {
        let mut r = make_registry();
        assert!(!r.transition("no-such-task", "running", 1000, 1));
    }
}

impl crate::organ::OrganApply for TaskRegistry {
    /// Organ-owned WAL replay (THEORY.md §8 Phase 2).
    fn apply(&mut self, op: crate::ops::Op) -> Option<crate::ops::Op> {
        use crate::ops::Op;
        match op {
            Op::TaskEvent(ev) => {
                let payload_str = String::from_utf8(ev.payload_json).unwrap_or_default();
                match ev.kind.as_str() {
                    "create" => {
                        self.create(
                            ev.task_id,
                            ev.task_type,
                            payload_str,
                            ev.ts_ms,
                            ev.fencing_token,
                        );
                    }
                    "start" | "pause" | "resume" | "complete" | "fail" => {
                        self.transition(&ev.task_id, &ev.kind, ev.ts_ms, ev.fencing_token);
                    }
                    _ => {}
                }
                    None
                }
            other => Some(other),
        }
    }
}

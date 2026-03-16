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

    pub fn create(&mut self, task_id: String, kind: String, payload_json: String, now_ms: i64, fencing_token: u64) {
        self.tasks.insert(task_id.clone(), TaskRecord {
            task_id,
            kind,
            status: TaskStatus::Pending,
            payload_json,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
            fencing_token,
        });
    }

    pub fn transition(&mut self, task_id: &str, new_status: &str, now_ms: i64, fencing_token: u64) -> bool {
        if let Some(t) = self.tasks.get_mut(task_id) {
            if fencing_token > 0 && fencing_token < t.fencing_token {
                return false;
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
        self.tasks.values()
            .filter(|t| matches!(t.status, TaskStatus::Pending | TaskStatus::Running | TaskStatus::Paused))
            .collect()
    }

    pub fn list_all(&self) -> Vec<&TaskRecord> {
        self.tasks.values().collect()
    }

    pub fn update_payload(&mut self, task_id: &str, payload_json: String, now_ms: i64) -> bool {
        if let Some(t) = self.tasks.get_mut(task_id) {
            t.payload_json = payload_json;
            t.updated_at_ms = now_ms;
            true
        } else {
            false
        }
    }
}

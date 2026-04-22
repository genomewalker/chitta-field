use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplSession {
    pub id: String,
    pub namespace_json: String,
    pub updated_ms: i64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ReplSessionStore {
    sessions: HashMap<String, ReplSession>,
    #[serde(skip)]
    path: PathBuf,
}

impl ReplSessionStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load(data_dir: &Path) -> Self {
        let path = data_dir.join("repl_sessions.json");
        let mut store: Self = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        store.path = path;
        store
    }

    fn save(&self) {
        if self.path.as_os_str().is_empty() {
            return;
        }
        if let Ok(json) = serde_json::to_string(self) {
            let tmp = self.path.with_extension("json.tmp");
            if std::fs::write(&tmp, &json).is_ok() {
                let _ = std::fs::rename(&tmp, &self.path);
            }
        }
    }

    pub fn get(&self, id: &str) -> Option<&ReplSession> {
        self.sessions.get(id)
    }

    pub fn set(&mut self, id: String, namespace_json: String, updated_ms: i64) {
        self.sessions.insert(
            id.clone(),
            ReplSession { id, namespace_json, updated_ms },
        );
        self.save();
    }

    pub fn delete(&mut self, id: &str) -> bool {
        let removed = self.sessions.remove(id).is_some();
        if removed {
            self.save();
        }
        removed
    }

    pub fn list(&self) -> Vec<&ReplSession> {
        let mut v: Vec<&ReplSession> = self.sessions.values().collect();
        v.sort_by(|a, b| b.updated_ms.cmp(&a.updated_ms));
        v
    }

    pub fn count(&self) -> usize {
        self.sessions.len()
    }
}

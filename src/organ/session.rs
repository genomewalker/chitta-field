use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SessionRecord {
    pub session_id: String,
    pub kind: String,
    pub realm: String,
    pub started_at_ms: i64,
    pub last_heartbeat_ms: i64,
    pub status: String, // "active" | "closed"
}

#[derive(Debug, Default)]
pub struct SessionRegistry {
    sessions: HashMap<String, SessionRecord>,
}

impl SessionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, session_id: String, kind: String, realm: String, now_ms: i64) {
        self.sessions.insert(
            session_id.clone(),
            SessionRecord {
                session_id,
                kind,
                realm,
                started_at_ms: now_ms,
                last_heartbeat_ms: now_ms,
                status: "active".into(),
            },
        );
    }

    pub fn heartbeat(&mut self, session_id: &str, now_ms: i64) {
        if let Some(s) = self.sessions.get_mut(session_id) {
            s.last_heartbeat_ms = now_ms;
        }
    }

    pub fn deregister(&mut self, session_id: &str) {
        if let Some(s) = self.sessions.get_mut(session_id) {
            s.status = "closed".into();
        }
    }

    pub fn get(&self, session_id: &str) -> Option<&SessionRecord> {
        self.sessions.get(session_id)
    }

    pub fn list_active(&self) -> Vec<&SessionRecord> {
        self.sessions
            .values()
            .filter(|s| s.status == "active")
            .collect()
    }

    pub fn list_all(&self) -> Vec<&SessionRecord> {
        self.sessions.values().collect()
    }

    pub fn expire_stale(&mut self, now_ms: i64, ttl_ms: i64) -> Vec<String> {
        let mut expired = Vec::new();
        for s in self.sessions.values_mut() {
            if s.status == "active" && now_ms - s.last_heartbeat_ms > ttl_ms {
                s.status = "closed".into();
                expired.push(s.session_id.clone());
            }
        }
        expired
    }
}

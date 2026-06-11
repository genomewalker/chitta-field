use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct UserModelEntry {
    pub entity_id: String,
    pub entity_type: String,
    pub payload_json: String,
    pub updated_at_ms: i64,
    pub observation_count: u32,
}

#[derive(Debug, Default)]
pub struct UserModelRegistry {
    entries: HashMap<String, UserModelEntry>,
}

impl UserModelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn upsert(
        &mut self,
        entity_id: String,
        entity_type: String,
        payload_json: String,
        now_ms: i64,
    ) {
        self.entries.insert(
            entity_id.clone(),
            UserModelEntry {
                entity_id,
                entity_type,
                payload_json,
                updated_at_ms: now_ms,
                observation_count: 0,
            },
        );
    }

    pub fn observe(&mut self, entity_id: &str, now_ms: i64) {
        if let Some(e) = self.entries.get_mut(entity_id) {
            e.observation_count += 1;
            e.updated_at_ms = now_ms;
        }
    }

    pub fn get(&self, entity_id: &str) -> Option<&UserModelEntry> {
        self.entries.get(entity_id)
    }

    pub fn list_by_type(&self, entity_type: &str) -> Vec<&UserModelEntry> {
        self.entries
            .values()
            .filter(|e| e.entity_type == entity_type)
            .collect()
    }

    pub fn list_all(&self) -> Vec<&UserModelEntry> {
        self.entries.values().collect()
    }
}

impl crate::organ::OrganApply for UserModelRegistry {
    /// Organ-owned WAL replay (THEORY.md §8 Phase 2).
    fn apply(&mut self, op: crate::ops::Op) -> Option<crate::ops::Op> {
        use crate::ops::Op;
        match op {
            Op::UserModelEvent(ev) => {
                let payload_str = String::from_utf8(ev.payload_json).unwrap_or_default();
                match ev.kind.as_str() {
                    "upsert" => {
                        self.upsert(ev.entity_id, ev.entity_type, payload_str, ev.ts_ms);
                    }
                    "observe" | "progress" | "complete" => {
                        self.observe(&ev.entity_id, ev.ts_ms);
                    }
                    _ => {}
                }
                    None
                }
            other => Some(other),
        }
    }
}

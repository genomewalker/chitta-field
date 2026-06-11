#[derive(Debug, Clone)]
pub struct AnalyticsEntry {
    pub id: u64,
    pub kind: String,
    pub entity_id: String,
    pub payload_json: String,
    pub ts_ms: i64,
}

#[derive(Debug, Default)]
pub struct AnalyticsRegistry {
    entries: Vec<AnalyticsEntry>,
    next_id: u64,
}

impl AnalyticsRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn append(
        &mut self,
        kind: String,
        entity_id: String,
        payload_json: String,
        ts_ms: i64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push(AnalyticsEntry {
            id,
            kind,
            entity_id,
            payload_json,
            ts_ms,
        });
        id
    }

    pub fn recent(&self, limit: usize) -> Vec<&AnalyticsEntry> {
        let start = self.entries.len().saturating_sub(limit);
        self.entries[start..].iter().collect()
    }

    pub fn count(&self) -> usize {
        self.entries.len()
    }
}

impl crate::organ::OrganApply for AnalyticsRegistry {
    /// Organ-owned WAL replay (THEORY.md §8 Phase 2).
    fn apply(&mut self, op: crate::ops::Op) -> Option<crate::ops::Op> {
        use crate::ops::Op;
        match op {
            Op::AnalyticsEvent(ev) => {
                let payload_str = String::from_utf8(ev.payload_json).unwrap_or_default();
                self.append(ev.kind, ev.session_id, payload_str, ev.ts_ms);
                    None
                }
            other => Some(other),
        }
    }
}

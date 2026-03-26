use std::collections::HashMap;

/// A single message event stored in the registry.
#[derive(Debug, Clone)]
pub struct MsgEvent {
    pub event_id: u64,
    pub domain: String,
    pub kind: String,
    pub target: String,
    pub payload_json: String,
    pub realm: String,
    pub ts_ms: i64,
}

/// In-memory registry of domain events indexed by (domain, kind, target).
/// Populated during WAL replay and live via cf_emit_event.
#[derive(Debug, Default)]
pub struct MsgRegistry {
    /// Key: target (entity_id), Value: events in insertion order.
    by_target: HashMap<String, Vec<MsgEvent>>,
}

impl MsgRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, event: MsgEvent) {
        self.by_target
            .entry(event.target.clone())
            .or_default()
            .push(event);
    }

    /// Query events by domain, kind, and target. Returns up to `limit` events
    /// in insertion order (oldest first).
    pub fn get_events(
        &self,
        domain: &str,
        kind: &str,
        target: &str,
        limit: usize,
    ) -> Vec<&MsgEvent> {
        let Some(events) = self.by_target.get(target) else {
            return Vec::new();
        };
        events
            .iter()
            .filter(|e| e.domain == domain && e.kind == kind)
            .take(limit)
            .collect()
    }
}

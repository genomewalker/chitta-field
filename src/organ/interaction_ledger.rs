use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EventKind {
    Retrieve = 0,
    Inject   = 1,
    Outcome  = 2,
    Override = 3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservedRef {
    pub memory_id:  u64,
    pub kind:       String,
    pub provenance: String,
    pub score:      f32,
    pub rank:       u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventPayload {
    Retrieve { query: String, strategy: String, limit: u32, refs: Vec<ObservedRef> },
    Inject   { memory_ids: Vec<u64>, token_budget: u32 },
    Outcome  { success: bool, error_kind: Option<String>, turn_count: u32 },
    Override { memory_id: u64, old_text: String, new_text: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ObsMode { Live, Shadow }

impl Default for ObsMode {
    fn default() -> Self { ObsMode::Live }
}

fn default_ts_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEvent {
    #[serde(default)]
    pub event_id:         u64,
    #[serde(default = "default_ts_ms")]
    pub ts_ms:            i64,
    pub session_id:       String,
    #[serde(default)]
    pub thread_id:        Option<String>,
    pub kind:             EventKind,
    pub payload:          EventPayload,
    #[serde(default)]
    pub causal_parent:    Option<u64>,
    #[serde(default)]
    pub observation_mode: ObsMode,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AssertionStatus { Active, Superseded }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedAssertion {
    pub assertion_id:     u64,
    pub subject:          String,
    pub predicate:        String,
    pub object:           String,
    pub version:          u32,
    pub status:           AssertionStatus,
    pub valid_from_ms:    i64,
    pub valid_to_ms:      Option<i64>,
    pub supersedes:       Option<u64>,
    pub source_event_ids: Vec<u64>,
    pub confidence:       f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InteractionLedger {
    pub events:         Vec<InteractionEvent>,
    pub assertions:     Vec<VersionedAssertion>,
    pub next_event_id:  u64,
    pub next_assert_id: u64,
    pub compile_cursor: usize,
}

impl InteractionLedger {
    pub fn append(&mut self, mut ev: InteractionEvent) -> u64 {
        ev.event_id = self.next_event_id;
        self.next_event_id += 1;
        self.events.push(ev);
        self.next_event_id - 1
    }

    pub fn query(
        &self,
        kind: Option<&EventKind>,
        session_id: Option<&str>,
        since_ms: Option<i64>,
        limit: usize,
    ) -> Vec<&InteractionEvent> {
        self.events.iter()
            .filter(|e| kind.map_or(true, |k| &e.kind == k))
            .filter(|e| session_id.map_or(true, |s| e.session_id == s))
            .filter(|e| since_ms.map_or(true, |t| e.ts_ms >= t))
            .rev()
            .take(limit)
            .collect()
    }

    pub fn compile(&mut self, now_ms: i64) {
        for i in self.compile_cursor..self.events.len() {
            if self.events[i].kind != EventKind::Override {
                continue;
            }
            let (memory_id, new_text, ev_id, ev_ts) = match &self.events[i].payload {
                EventPayload::Override { memory_id, new_text, .. } => {
                    (*memory_id, new_text.clone(), self.events[i].event_id, self.events[i].ts_ms)
                }
                _ => continue,
            };
            let subject = memory_id.to_string();
            let prev_version = self.assertions.iter()
                .filter(|a| a.subject == subject)
                .map(|a| a.version)
                .max()
                .unwrap_or(0);
            let mut prev_id: Option<u64> = None;
            for a in self.assertions.iter_mut() {
                if a.subject == subject && a.status == AssertionStatus::Active {
                    a.status = AssertionStatus::Superseded;
                    a.valid_to_ms = Some(now_ms);
                    prev_id = Some(a.assertion_id);
                }
            }
            let new_id = self.next_assert_id;
            self.next_assert_id += 1;
            self.assertions.push(VersionedAssertion {
                assertion_id:     new_id,
                subject,
                predicate:        "content".into(),
                object:           new_text,
                version:          prev_version + 1,
                status:           AssertionStatus::Active,
                valid_from_ms:    ev_ts,
                valid_to_ms:      None,
                supersedes:       prev_id,
                source_event_ids: vec![ev_id],
                confidence:       1.0,
            });
        }
        self.compile_cursor = self.events.len();
    }

    /// Return (subject, predicate, assertion_ids) for pairs with >1 active assertion.
    pub fn contested(&self) -> Vec<(String, String, Vec<u64>)> {
        let mut map: HashMap<(String, String), Vec<u64>> = HashMap::new();
        for a in &self.assertions {
            if a.status == AssertionStatus::Active {
                map.entry((a.subject.clone(), a.predicate.clone()))
                   .or_default()
                   .push(a.assertion_id);
            }
        }
        map.into_iter()
            .filter(|(_, ids)| ids.len() > 1)
            .map(|((s, p), ids)| (s, p, ids))
            .collect()
    }
}

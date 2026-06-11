use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TurnRecord {
    pub id: u64,
    pub role: String,
    pub content: String,
    pub ts_ms: i64,
}

#[derive(Debug, Clone)]
pub struct TranscriptRecord {
    pub transcript_id: String,
    pub session_id: String,
    pub progress_pct: f32,
    pub turns: Vec<TurnRecord>,
}

#[derive(Debug, Default)]
pub struct TranscriptRegistry {
    transcripts: HashMap<String, TranscriptRecord>,
    next_turn_id: u64,
    /// Latest event payload per (session_id, kind) — for cf_get_latest_event("transcript", ...)
    session_events: HashMap<String, HashMap<String, String>>,
}

impl TranscriptRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store the latest event payload for a session+kind pair.
    pub fn set_session_event(&mut self, session_id: &str, kind: &str, payload_json: String) {
        self.session_events
            .entry(session_id.to_string())
            .or_default()
            .insert(kind.to_string(), payload_json);
    }

    /// Retrieve the latest stored payload for a session+kind pair.
    pub fn get_session_event(&self, session_id: &str, kind: &str) -> Option<&str> {
        self.session_events
            .get(session_id)
            .and_then(|m| m.get(kind))
            .map(|s| s.as_str())
    }

    pub fn register(&mut self, transcript_id: String, session_id: String) {
        self.transcripts.insert(
            transcript_id.clone(),
            TranscriptRecord {
                transcript_id,
                session_id,
                progress_pct: 0.0,
                turns: Vec::new(),
            },
        );
    }

    pub fn update_progress(&mut self, transcript_id: &str, pct: f32) {
        if let Some(t) = self.transcripts.get_mut(transcript_id) {
            t.progress_pct = pct;
        }
    }

    pub fn add_turn(
        &mut self,
        transcript_id: &str,
        role: String,
        content: String,
        ts_ms: i64,
    ) -> u64 {
        let id = self.next_turn_id;
        self.next_turn_id += 1;
        if let Some(t) = self.transcripts.get_mut(transcript_id) {
            t.turns.push(TurnRecord {
                id,
                role,
                content,
                ts_ms,
            });
        }
        id
    }

    pub fn get(&self, transcript_id: &str) -> Option<&TranscriptRecord> {
        self.transcripts.get(transcript_id)
    }

    pub fn list_all(&self) -> Vec<&TranscriptRecord> {
        self.transcripts.values().collect()
    }
}

impl crate::organ::OrganApply for TranscriptRegistry {
    /// Organ-owned WAL replay (THEORY.md §8 Phase 2).
    fn apply(&mut self, op: crate::ops::Op) -> Option<crate::ops::Op> {
        use crate::ops::Op;
        match op {
            Op::TranscriptEvent(ev) => {
                // Always store the raw payload for cf_get_latest_event("transcript", kind, session_id)
                let payload_str = String::from_utf8(ev.payload_json.clone()).unwrap_or_default();
                self.set_session_event(&ev.session_id, &ev.kind, payload_str);

                match ev.kind.as_str() {
                    "register" => {
                        let payload =
                            serde_json::from_slice::<serde_json::Value>(&ev.payload_json).ok();
                        let transcript_id = payload
                            .as_ref()
                            .and_then(|v| {
                                v.get("transcript_id")
                                    .and_then(|k| k.as_str())
                                    .map(|s| s.to_string())
                            })
                            .unwrap_or_default();
                        if !transcript_id.is_empty() {
                            self.register(transcript_id, ev.session_id);
                        }
                    }
                    "update_progress" => {
                        let payload =
                            serde_json::from_slice::<serde_json::Value>(&ev.payload_json).ok();
                        let transcript_id = payload
                            .as_ref()
                            .and_then(|v| {
                                v.get("transcript_id")
                                    .and_then(|k| k.as_str())
                                    .map(|s| s.to_string())
                            })
                            .unwrap_or_default();
                        let pct = payload
                            .as_ref()
                            .and_then(|v| v.get("progress_pct").and_then(|k| k.as_f64()))
                            .unwrap_or(0.0) as f32;
                        if !transcript_id.is_empty() {
                            self.update_progress(&transcript_id, pct);
                        }
                    }
                    "add_turn" => {
                        let payload =
                            serde_json::from_slice::<serde_json::Value>(&ev.payload_json).ok();
                        let transcript_id = payload
                            .as_ref()
                            .and_then(|v| {
                                v.get("transcript_id")
                                    .and_then(|k| k.as_str())
                                    .map(|s| s.to_string())
                            })
                            .unwrap_or_default();
                        let role = payload
                            .as_ref()
                            .and_then(|v| {
                                v.get("role")
                                    .and_then(|k| k.as_str())
                                    .map(|s| s.to_string())
                            })
                            .unwrap_or_default();
                        let content = payload
                            .as_ref()
                            .and_then(|v| {
                                v.get("content")
                                    .and_then(|k| k.as_str())
                                    .map(|s| s.to_string())
                            })
                            .unwrap_or_default();
                        if !transcript_id.is_empty() {
                            self.add_turn(&transcript_id, role, content, ev.ts_ms);
                        }
                    }
                    _ => {}
                }
                    None
                }
            other => Some(other),
        }
    }
}

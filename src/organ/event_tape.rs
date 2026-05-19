use serde::{Deserialize, Serialize};
use std::collections::HashMap;

fn fnv1a_64(s: &[u8]) -> u64 {
    s.iter().fold(0xcbf29ce484222325u64, |h, &b| {
        h.wrapping_mul(0x100000001b3) ^ b as u64
    })
}

fn canonicalize_entity(entity: &str) -> String {
    if entity.contains('/') || entity.contains('\\') {
        // File path: strip HOME prefix, keep repo-relative
        if let Ok(home) = std::env::var("HOME") {
            if let Some(rel) = entity.strip_prefix(&*home) {
                return rel.trim_start_matches('/').to_string();
            }
        }
        return entity.trim_start_matches('/').to_string();
    }
    if entity.starts_with("http://") || entity.starts_with("https://") {
        // URL: hostname only
        if let Some(host) = entity.split("//").nth(1).and_then(|s| s.split('/').next()) {
            return host.to_string();
        }
    }
    if entity.contains("::") {
        // FQN: keep lowercase
        return entity.to_lowercase();
    }
    // Freeform: first 40 alphanum chars, lowercase
    entity
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-' || *c == '.')
        .take(40)
        .collect::<String>()
        .to_lowercase()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnEvent {
    pub turn_id:       u32,
    pub tool_id:       u16,
    pub entity_key:    u32,
    pub outcome_class: u8,  // 0=success 1=fail 2=error 3=partial/legacy
    pub session_id:    u64,
    pub ts_ms:         i64,
}

impl TurnEvent {
    pub fn pack(&self) -> u64 {
        ((self.tool_id as u64) << 40)
            | ((self.outcome_class as u64) << 32)
            | (self.entity_key as u64)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EventTape {
    pub events:      Vec<TurnEvent>,
    tool_names:      Vec<String>,
    tool_interner:   HashMap<String, u16>,
    entity_names:    Vec<String>,
    entity_interner: HashMap<u64, u32>,
}

impl EventTape {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern_tool(&mut self, tool: &str) -> u16 {
        if let Some(&id) = self.tool_interner.get(tool) {
            return id;
        }
        let id = self.tool_names.len() as u16;
        self.tool_names.push(tool.to_string());
        self.tool_interner.insert(tool.to_string(), id);
        id
    }

    pub fn intern_entity(&mut self, entity: &str) -> u32 {
        let canon = canonicalize_entity(entity);
        let hash = fnv1a_64(canon.as_bytes());
        if let Some(&id) = self.entity_interner.get(&hash) {
            return id;
        }
        let id = self.entity_names.len() as u32;
        self.entity_names.push(canon);
        self.entity_interner.insert(hash, id);
        id
    }

    /// Append one event, return its packed symbol.
    pub fn log(
        &mut self,
        tool: &str,
        entity: &str,
        outcome: u8,
        session_id: u64,
        ts_ms: i64,
    ) -> u64 {
        let tool_id    = self.intern_tool(tool);
        let entity_key = self.intern_entity(entity);
        let turn_id    = self.events.len() as u32;
        let ev = TurnEvent { turn_id, tool_id, entity_key, outcome_class: outcome, session_id, ts_ms };
        let sym = ev.pack();
        self.events.push(ev);
        sym
    }

    /// Compute the packed symbol without appending (lookup existing interners).
    /// Interns tool/entity if not present (side effect on interner, not tape).
    pub fn symbol_of(&mut self, tool: &str, entity: &str, outcome: u8) -> u64 {
        let tool_id    = self.intern_tool(tool);
        let entity_key = self.intern_entity(entity);
        TurnEvent { turn_id: 0, tool_id, entity_key, outcome_class: outcome, session_id: 0, ts_ms: 0 }.pack()
    }

    /// Seed entity interner from existing triplet subjects/objects.
    pub fn seed_from_triplets<'a>(&mut self, subjects: impl Iterator<Item = &'a str>) {
        for s in subjects {
            self.intern_entity(s);
        }
    }

    /// Synthesize a legacy event for an existing memory (warm-start CDAWG migration).
    pub fn synthesize_legacy(&mut self, realm: &str, ts_ms: i64) {
        self.log("legacy", realm, 3, 0, ts_ms);
    }

    pub fn tool_name(&self, id: u16) -> &str {
        self.tool_names.get(id as usize).map(|s| s.as_str()).unwrap_or("unknown")
    }

    pub fn entity_name(&self, key: u32) -> &str {
        self.entity_names.get(key as usize).map(|s| s.as_str()).unwrap_or("unknown")
    }

    /// Return the last n packed symbols from the tape (oldest first).
    pub fn last_n_syms(&self, n: usize) -> Vec<u64> {
        let start = self.events.len().saturating_sub(n);
        self.events[start..].iter().map(|e| e.pack()).collect()
    }
}

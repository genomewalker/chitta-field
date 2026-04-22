use crate::ids::MemoryId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CorrectionState {
    #[default]
    Emitted,
    Acknowledged,
    Applied,
    Verified,
}

/// A single subject-predicate-object fact.
/// `weight` is the forward (subject→object) strength.
/// `reverse_weight` is the backward (object→subject) strength.
/// Asymmetric weights emerge from sequential observations (FEP §3.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripletEntry {
    pub id: u64,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub weight: f32,
    #[serde(default = "default_reverse_weight")]
    pub reverse_weight: f32,
    pub valid_from_ms: i64,
    pub valid_to_ms: i64, // 0 = still valid
    pub source_memory_id: Option<MemoryId>,
    pub source_file: Option<String>,
}

fn default_reverse_weight() -> f32 {
    -1.0 // sentinel: -1 means "use weight" (backward compat)
}

impl TripletEntry {
    /// Forward weight (subject→object).
    pub fn forward_weight(&self) -> f32 {
        self.weight
    }
    /// Reverse weight (object→subject). Falls back to weight for legacy entries.
    pub fn reverse_weight(&self) -> f32 {
        if self.reverse_weight < 0.0 { self.weight } else { self.reverse_weight }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripletStore {
    next_id: u64,
    entries: Vec<TripletEntry>,
    id_to_index: HashMap<u64, usize>,

    by_subject: HashMap<String, Vec<u64>>,
    by_object: HashMap<String, Vec<u64>>,
    by_predicate: HashMap<String, Vec<u64>>,

    // Sidecar — not serialized (bincode ignores serde(default); skip = ephemeral).
    // Absent entries are implicitly Emitted.
    #[serde(skip)]
    pub correction_states: HashMap<u64, CorrectionState>,
}

impl TripletStore {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            entries: Vec::new(),
            id_to_index: HashMap::new(),
            by_subject: HashMap::new(),
            by_object: HashMap::new(),
            by_predicate: HashMap::new(),
            correction_states: HashMap::new(),
        }
    }

    pub fn correction_state(&self, id: u64) -> CorrectionState {
        self.correction_states.get(&id).copied().unwrap_or_default()
    }

    /// Add a triplet fact, allocating a new ID. Returns the new triplet ID.
    pub fn add(
        &mut self,
        subject: String,
        predicate: String,
        object: String,
        weight: f32,
        valid_from_ms: i64,
        source_memory_id: Option<MemoryId>,
        source_file: Option<String>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.insert_with_id(
            id,
            subject,
            predicate,
            object,
            weight,
            valid_from_ms,
            source_memory_id,
            source_file,
        );
        id
    }

    /// Add a triplet with an explicit ID (used during log replay).
    /// Advances next_id past the given id if necessary.
    pub fn replay_add(
        &mut self,
        id: u64,
        subject: String,
        predicate: String,
        object: String,
        weight: f32,
        valid_from_ms: i64,
        source_memory_id: Option<MemoryId>,
        source_file: Option<String>,
    ) {
        if id >= self.next_id {
            self.next_id = id + 1;
        }
        self.insert_with_id(
            id,
            subject,
            predicate,
            object,
            weight,
            valid_from_ms,
            source_memory_id,
            source_file,
        );
    }

    fn insert_with_id(
        &mut self,
        id: u64,
        subject: String,
        predicate: String,
        object: String,
        weight: f32,
        valid_from_ms: i64,
        source_memory_id: Option<MemoryId>,
        source_file: Option<String>,
    ) {
        let idx = self.entries.len();
        self.id_to_index.insert(id, idx);

        let entry = TripletEntry {
            id,
            subject: subject.clone(),
            predicate: predicate.clone(),
            object: object.clone(),
            weight,
            reverse_weight: weight * 0.3, // asymmetric by default: reverse is attenuated
            valid_from_ms,
            valid_to_ms: 0,
            source_memory_id,
            source_file,
        };
        self.entries.push(entry);

        self.by_subject
            .entry(subject)
            .or_insert_with(Vec::new)
            .push(id);
        self.by_object
            .entry(object)
            .or_insert_with(Vec::new)
            .push(id);
        self.by_predicate
            .entry(predicate)
            .or_insert_with(Vec::new)
            .push(id);
    }

    /// Invalidate a triplet (set valid_to_ms = now_ms).
    pub fn invalidate(&mut self, triplet_id: u64, now_ms: i64) {
        if let Some(&idx) = self.id_to_index.get(&triplet_id) {
            if let Some(entry) = self.entries.get_mut(idx) {
                entry.valid_to_ms = now_ms;
            }
        }
    }

    fn entry_by_id(&self, id: u64) -> Option<&TripletEntry> {
        let &idx = self.id_to_index.get(&id)?;
        self.entries.get(idx)
    }

    fn is_valid(entry: &TripletEntry, at_ms: i64) -> bool {
        // valid_to_ms == 0 means still valid (no expiry set).
        // Otherwise, the triplet is valid while at_ms < valid_to_ms.
        entry.valid_to_ms == 0 || at_ms < entry.valid_to_ms
    }

    fn resolve_ids<'a>(&'a self, ids: &[u64], at_ms: i64) -> Vec<&'a TripletEntry> {
        ids.iter()
            .filter_map(|&id| self.entry_by_id(id))
            .filter(|e| Self::is_valid(e, at_ms))
            .collect()
    }

    /// Query all valid triplets with the given subject.
    pub fn query_subject(&self, subject: &str, at_ms: i64) -> Vec<&TripletEntry> {
        match self.by_subject.get(subject) {
            Some(ids) => self.resolve_ids(ids, at_ms),
            None => Vec::new(),
        }
    }

    /// Query all valid triplets with the given object.
    pub fn query_object(&self, object: &str, at_ms: i64) -> Vec<&TripletEntry> {
        match self.by_object.get(object) {
            Some(ids) => self.resolve_ids(ids, at_ms),
            None => Vec::new(),
        }
    }

    /// Query all valid triplets with the given predicate.
    pub fn query_predicate(&self, predicate: &str, at_ms: i64) -> Vec<&TripletEntry> {
        match self.by_predicate.get(predicate) {
            Some(ids) => self.resolve_ids(ids, at_ms),
            None => Vec::new(),
        }
    }

    /// Query all valid triplets where subject OR object matches the given string.
    pub fn query_entity(&self, entity: &str, at_ms: i64) -> Vec<&TripletEntry> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();

        let subject_ids = self
            .by_subject
            .get(entity)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let object_ids = self
            .by_object
            .get(entity)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        for &id in subject_ids.iter().chain(object_ids.iter()) {
            if seen.insert(id) {
                if let Some(entry) = self.entry_by_id(id) {
                    if Self::is_valid(entry, at_ms) {
                        result.push(entry);
                    }
                }
            }
        }

        result
    }

    /// Find all objects connected to `subject` via `predicate`.
    pub fn objects_of(&self, subject: &str, predicate: &str, at_ms: i64) -> Vec<String> {
        self.query_subject(subject, at_ms)
            .into_iter()
            .filter(|e| e.predicate == predicate)
            .map(|e| e.object.clone())
            .collect()
    }

    /// Find all subjects that have `predicate -> object`.
    pub fn subjects_of(&self, predicate: &str, object: &str, at_ms: i64) -> Vec<String> {
        self.query_object(object, at_ms)
            .into_iter()
            .filter(|e| e.predicate == predicate)
            .map(|e| e.subject.clone())
            .collect()
    }

    /// Invalidate all active triplets whose source_file matches.
    /// Returns the IDs of invalidated triplets.
    pub fn invalidate_by_source_file(&mut self, source_file: &str, now_ms: i64) -> Vec<u64> {
        let mut invalidated = Vec::new();
        for entry in self.entries.iter_mut() {
            if entry.valid_to_ms == 0 {
                if let Some(ref sf) = entry.source_file {
                    if sf == source_file {
                        entry.valid_to_ms = now_ms;
                        invalidated.push(entry.id);
                    }
                }
            }
        }
        invalidated
    }

    pub fn set_correction_state(&mut self, id: u64, state: CorrectionState) -> bool {
        if self.id_to_index.contains_key(&id) {
            self.correction_states.insert(id, state);
            true
        } else {
            false
        }
    }

    pub fn triplet_count(&self) -> usize {
        self.entries.len()
    }

    /// Current next_id — used to seed the TripletIdAllocator after replay.
    pub fn next_id(&self) -> u64 {
        self.next_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_query_subject() {
        let mut store = TripletStore::new();
        store.add(
            "chitta".into(),
            "uses".into(),
            "duckdb".into(),
            1.0,
            0,
            None,
            None,
        );
        store.add(
            "chitta".into(),
            "has".into(),
            "memory".into(),
            0.9,
            0,
            None,
            None,
        );
        store.add(
            "duckdb".into(),
            "is_a".into(),
            "database".into(),
            1.0,
            0,
            None,
            None,
        );

        let results = store.query_subject("chitta", 0);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_temporal_invalidation() {
        let mut store = TripletStore::new();
        let id = store.add(
            "chitta".into(),
            "uses".into(),
            "duckdb".into(),
            1.0,
            0,
            None,
            None,
        );

        // Valid at time 500
        let results = store.query_subject("chitta", 500);
        assert_eq!(results.len(), 1);

        // Invalidate at time 1000
        store.invalidate(id, 1000);

        // Now invalid at time 1500
        let results = store.query_subject("chitta", 1500);
        assert_eq!(results.len(), 0);

        // But was still valid at time 999
        let results = store.query_subject("chitta", 999);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_entity() {
        let mut store = TripletStore::new();
        store.add(
            "alice".into(),
            "knows".into(),
            "bob".into(),
            1.0,
            0,
            None,
            None,
        );
        store.add(
            "charlie".into(),
            "knows".into(),
            "alice".into(),
            1.0,
            0,
            None,
            None,
        );
        store.add(
            "alice".into(),
            "works_at".into(),
            "anthropic".into(),
            1.0,
            0,
            None,
            None,
        );

        let results = store.query_entity("alice", 0);
        assert_eq!(results.len(), 3); // alice appears as subject twice, object once
    }

    #[test]
    fn test_query_object() {
        let mut store = TripletStore::new();
        store.add(
            "chitta".into(),
            "uses".into(),
            "duckdb".into(),
            1.0,
            0,
            None,
            None,
        );
        store.add(
            "amber".into(),
            "uses".into(),
            "duckdb".into(),
            0.8,
            0,
            None,
            None,
        );

        let results = store.query_object("duckdb", 0);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_predicate() {
        let mut store = TripletStore::new();
        store.add(
            "chitta".into(),
            "uses".into(),
            "duckdb".into(),
            1.0,
            0,
            None,
            None,
        );
        store.add(
            "chitta".into(),
            "uses".into(),
            "hnsw".into(),
            0.9,
            0,
            None,
            None,
        );
        store.add(
            "chitta".into(),
            "has".into(),
            "memory".into(),
            1.0,
            0,
            None,
            None,
        );

        let results = store.query_predicate("uses", 0);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_objects_of() {
        let mut store = TripletStore::new();
        store.add(
            "chitta".into(),
            "uses".into(),
            "duckdb".into(),
            1.0,
            0,
            None,
            None,
        );
        store.add(
            "chitta".into(),
            "uses".into(),
            "hnsw".into(),
            0.9,
            0,
            None,
            None,
        );
        store.add(
            "chitta".into(),
            "has".into(),
            "memory".into(),
            1.0,
            0,
            None,
            None,
        );

        let objects = store.objects_of("chitta", "uses", 0);
        assert_eq!(objects.len(), 2);
        assert!(objects.contains(&"duckdb".to_string()));
        assert!(objects.contains(&"hnsw".to_string()));
    }

    #[test]
    fn test_subjects_of() {
        let mut store = TripletStore::new();
        store.add(
            "chitta".into(),
            "uses".into(),
            "duckdb".into(),
            1.0,
            0,
            None,
            None,
        );
        store.add(
            "amber".into(),
            "uses".into(),
            "duckdb".into(),
            0.8,
            0,
            None,
            None,
        );

        let subjects = store.subjects_of("uses", "duckdb", 0);
        assert_eq!(subjects.len(), 2);
        assert!(subjects.contains(&"chitta".to_string()));
        assert!(subjects.contains(&"amber".to_string()));
    }

    #[test]
    fn test_replay_add() {
        let mut store = TripletStore::new();
        // Simulate replay with a specific id
        store.replay_add(42, "a".into(), "b".into(), "c".into(), 1.0, 0, None, None);
        assert_eq!(store.next_id(), 43);
        let results = store.query_subject("a", 0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 42);
    }
}

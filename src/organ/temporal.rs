use std::collections::BTreeMap;
use crate::ids::MemoryId;

/// Entry in the temporal index.
#[derive(Debug, Clone)]
pub struct TemporalEntry {
    pub memory_id: MemoryId,
    pub ts_ms: i64,
    pub kind: String,
    pub realm: String,
    pub strength: f32,
}

/// Temporal index: sorted by (ts_ms, memory_id) for range queries.
pub struct TemporalIndex {
    inner: BTreeMap<(i64, MemoryId), TemporalEntry>,
}

impl TemporalIndex {
    pub fn new() -> Self {
        Self {
            inner: BTreeMap::new(),
        }
    }

    /// Insert or update an entry.
    pub fn upsert(&mut self, entry: TemporalEntry) {
        self.inner.insert((entry.ts_ms, entry.memory_id), entry);
    }

    /// Remove a memory (on forget).
    pub fn remove(&mut self, memory_id: MemoryId, ts_ms: i64) {
        self.inner.remove(&(ts_ms, memory_id));
    }

    /// Query memories in [start_ms, end_ms], optionally filtered by realm.
    /// Returns entries sorted by ts_ms descending (most recent first), limited to `limit`.
    pub fn range_query(
        &self,
        start_ms: i64,
        end_ms: i64,
        realm: Option<&str>,
        limit: usize,
    ) -> Vec<TemporalEntry> {
        self.inner
            .range((start_ms, 0)..=(end_ms, u64::MAX))
            .rev()
            .filter(|(_, e)| realm.map_or(true, |r| e.realm == r))
            .take(limit)
            .map(|(_, e)| e.clone())
            .collect()
    }

    /// Get the N most recent memories, optionally filtered by realm.
    pub fn most_recent(&self, realm: Option<&str>, limit: usize) -> Vec<TemporalEntry> {
        self.inner
            .values()
            .rev()
            .filter(|e| realm.map_or(true, |r| e.realm == r))
            .take(limit)
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_query() {
        let mut idx = TemporalIndex::new();
        for i in 0..10 {
            idx.upsert(TemporalEntry {
                memory_id: i as u64 + 1,
                ts_ms: i * 1000,
                kind: "wisdom".into(),
                realm: "test".into(),
                strength: 1.0,
            });
        }
        let results = idx.range_query(2000, 5000, None, 100);
        assert_eq!(results.len(), 4); // ts 2000, 3000, 4000, 5000
        // most recent first
        assert_eq!(results[0].ts_ms, 5000);
    }

    #[test]
    fn test_realm_filter() {
        let mut idx = TemporalIndex::new();
        idx.upsert(TemporalEntry {
            memory_id: 1,
            ts_ms: 1000,
            kind: "w".into(),
            realm: "a".into(),
            strength: 1.0,
        });
        idx.upsert(TemporalEntry {
            memory_id: 2,
            ts_ms: 2000,
            kind: "w".into(),
            realm: "b".into(),
            strength: 1.0,
        });
        let results = idx.range_query(0, 10000, Some("a"), 100);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory_id, 1);
    }

    #[test]
    fn test_remove() {
        let mut idx = TemporalIndex::new();
        idx.upsert(TemporalEntry {
            memory_id: 1,
            ts_ms: 1000,
            kind: "w".into(),
            realm: "test".into(),
            strength: 1.0,
        });
        assert_eq!(idx.len(), 1);
        idx.remove(1, 1000);
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn test_most_recent() {
        let mut idx = TemporalIndex::new();
        for i in 0..5u64 {
            idx.upsert(TemporalEntry {
                memory_id: i + 1,
                ts_ms: i as i64 * 1000,
                kind: "w".into(),
                realm: "test".into(),
                strength: 1.0,
            });
        }
        let results = idx.most_recent(None, 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].ts_ms, 4000);
        assert_eq!(results[1].ts_ms, 3000);
        assert_eq!(results[2].ts_ms, 2000);
    }
}

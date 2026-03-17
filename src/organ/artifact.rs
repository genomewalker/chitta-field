use crate::ids::{ArtifactId, MemoryId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactEntry {
    pub memory_id: MemoryId,
    pub artifact_id: ArtifactId,
    pub normalized_path: String,
    pub strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactIndex {
    /// path -> list of entries
    by_path: HashMap<String, Vec<ArtifactEntry>>,
    /// memory_id -> list of paths (for reverse lookup / cleanup)
    by_memory: HashMap<MemoryId, Vec<String>>,
}

impl ArtifactIndex {
    pub fn new() -> Self {
        Self {
            by_path: HashMap::new(),
            by_memory: HashMap::new(),
        }
    }

    /// Register an association between a memory and a file path.
    pub fn associate(
        &mut self,
        memory_id: MemoryId,
        artifact_id: ArtifactId,
        path: &str,
        strength: f32,
    ) {
        let entry = ArtifactEntry {
            memory_id,
            artifact_id,
            normalized_path: path.to_string(),
            strength,
        };

        let entries = self
            .by_path
            .entry(path.to_string())
            .or_insert_with(Vec::new);
        // Avoid duplicates: replace if same memory_id already present for this path.
        if let Some(pos) = entries.iter().position(|e| e.memory_id == memory_id) {
            entries[pos] = entry;
        } else {
            entries.push(entry);
        }

        self.by_memory
            .entry(memory_id)
            .or_insert_with(Vec::new)
            .push(path.to_string());
    }

    /// Remove all associations for a memory (on forget).
    pub fn remove_memory(&mut self, memory_id: MemoryId) {
        if let Some(paths) = self.by_memory.remove(&memory_id) {
            for path in paths {
                if let Some(entries) = self.by_path.get_mut(&path) {
                    entries.retain(|e| e.memory_id != memory_id);
                    if entries.is_empty() {
                        self.by_path.remove(&path);
                    }
                }
            }
        }
    }

    /// Query memories associated with a given path (exact match), sorted by strength desc.
    pub fn query_path(&self, path: &str, limit: usize) -> Vec<ArtifactEntry> {
        let Some(entries) = self.by_path.get(path) else {
            return Vec::new();
        };
        let mut result: Vec<ArtifactEntry> = entries.clone();
        result.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result.truncate(limit);
        result
    }

    /// Query memories associated with any path matching a prefix.
    pub fn query_prefix(&self, prefix: &str, limit: usize) -> Vec<ArtifactEntry> {
        let mut result: Vec<ArtifactEntry> = self
            .by_path
            .iter()
            .filter(|(path, _)| path.starts_with(prefix))
            .flat_map(|(_, entries)| entries.iter().cloned())
            .collect();
        result.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result.truncate(limit);
        result
    }

    /// Get all paths associated with a memory.
    pub fn paths_for_memory(&self, memory_id: MemoryId) -> Vec<String> {
        self.by_memory.get(&memory_id).cloned().unwrap_or_default()
    }

    pub fn len(&self) -> usize {
        self.by_path.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_artifact_associate_query() {
        let mut idx = ArtifactIndex::new();
        idx.associate(1, 10, "src/main.cpp", 0.9);
        idx.associate(2, 10, "src/main.cpp", 0.7);
        idx.associate(3, 20, "src/other.cpp", 1.0);
        let hits = idx.query_path("src/main.cpp", 10);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].memory_id, 1); // higher strength first
    }

    #[test]
    fn test_artifact_remove() {
        let mut idx = ArtifactIndex::new();
        idx.associate(1, 10, "src/main.cpp", 0.9);
        idx.remove_memory(1);
        let hits = idx.query_path("src/main.cpp", 10);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_query_prefix() {
        let mut idx = ArtifactIndex::new();
        idx.associate(1, 10, "src/main.cpp", 0.9);
        idx.associate(2, 11, "src/utils.cpp", 0.8);
        idx.associate(3, 12, "include/foo.h", 0.7);
        let hits = idx.query_prefix("src/", 10);
        assert_eq!(hits.len(), 2);
        // sorted by strength desc
        assert_eq!(hits[0].normalized_path, "src/main.cpp");
    }

    #[test]
    fn test_paths_for_memory() {
        let mut idx = ArtifactIndex::new();
        idx.associate(1, 10, "src/a.cpp", 0.9);
        idx.associate(1, 11, "src/b.cpp", 0.8);
        let paths = idx.paths_for_memory(1);
        assert_eq!(paths.len(), 2);
        assert!(paths.contains(&"src/a.cpp".to_string()));
        assert!(paths.contains(&"src/b.cpp".to_string()));
    }
}

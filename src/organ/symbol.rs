use std::collections::HashMap;
use crate::ids::MemoryId;
use serde::{Serialize, Deserialize};

pub type SymbolId = u64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolEntry {
    pub id: SymbolId,
    pub kind: String,
    pub name: String,
    pub signature: String,
    pub file_path: String,
    pub line_start: u32,
    pub line_end: u32,
    pub repo_id: u64,
    pub embedding: Vec<f32>,
    pub description: Option<String>,
    pub memory_id: Option<MemoryId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolIndex {
    by_id: HashMap<SymbolId, SymbolEntry>,
    by_name: HashMap<String, Vec<SymbolId>>,
    dedup: HashMap<(String, String, String, u32), SymbolId>,
}

impl SymbolIndex {
    pub fn new() -> Self {
        Self {
            by_id: HashMap::new(),
            by_name: HashMap::new(),
            dedup: HashMap::new(),
        }
    }

    /// Upsert a symbol. Deduplicates by (kind, name, file_path, line_start).
    /// Returns the SymbolId (existing or new).
    pub fn upsert(&mut self, entry: SymbolEntry) -> SymbolId {
        let key = (
            entry.kind.clone(),
            entry.name.clone(),
            entry.file_path.clone(),
            entry.line_start,
        );

        if let Some(&existing_id) = self.dedup.get(&key) {
            // Update existing entry in place.
            self.by_id.insert(existing_id, SymbolEntry { id: existing_id, ..entry });
            return existing_id;
        }

        let id = entry.id;
        self.dedup.insert(key, id);
        self.by_name.entry(entry.name.clone()).or_insert_with(Vec::new).push(id);
        self.by_id.insert(id, entry);
        id
    }

    pub fn remove(&mut self, id: SymbolId) {
        if let Some(entry) = self.by_id.remove(&id) {
            let key = (entry.kind, entry.name.clone(), entry.file_path, entry.line_start);
            self.dedup.remove(&key);
            if let Some(ids) = self.by_name.get_mut(&entry.name) {
                ids.retain(|&x| x != id);
                if ids.is_empty() {
                    self.by_name.remove(&entry.name);
                }
            }
        }
    }

    pub fn get(&self, id: SymbolId) -> Option<&SymbolEntry> {
        self.by_id.get(&id)
    }

    /// Search by name: exact match first, then prefix match, up to limit results.
    pub fn search_by_name(&self, query: &str, limit: usize) -> Vec<&SymbolEntry> {
        let mut results: Vec<&SymbolEntry> = Vec::new();

        // Exact matches first.
        if let Some(ids) = self.by_name.get(query) {
            for &id in ids {
                if let Some(e) = self.by_id.get(&id) {
                    results.push(e);
                }
            }
        }

        // Prefix matches (excluding exact matches already added).
        if results.len() < limit {
            for (name, ids) in &self.by_name {
                if name != query && name.starts_with(query) {
                    for &id in ids {
                        if let Some(e) = self.by_id.get(&id) {
                            results.push(e);
                            if results.len() >= limit {
                                return results;
                            }
                        }
                    }
                }
            }
        }

        results.truncate(limit);
        results
    }

    /// Semantic search: find k nearest by cosine similarity to query embedding.
    pub fn search_semantic(&self, query: &[f32], k: usize) -> Vec<(SymbolId, f32)> {
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_norm == 0.0 {
            return Vec::new();
        }

        let mut scored: Vec<(SymbolId, f32)> = self
            .by_id
            .values()
            .filter_map(|e| {
                if e.embedding.len() != query.len() {
                    return None;
                }
                let dot: f32 = e.embedding.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
                let emb_norm: f32 = e.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                if emb_norm == 0.0 {
                    return None;
                }
                Some((e.id, dot / (query_norm * emb_norm)))
            })
            .collect();

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Get all symbols in a file path.
    pub fn by_file(&self, file_path: &str) -> Vec<&SymbolEntry> {
        self.by_id
            .values()
            .filter(|e| e.file_path == file_path)
            .collect()
    }

    pub fn count(&self) -> usize {
        self.by_id.len()
    }

    /// Return the maximum symbol ID present, or None if empty.
    pub fn max_id(&self) -> Option<u64> {
        self.by_id.keys().copied().max()
    }
}

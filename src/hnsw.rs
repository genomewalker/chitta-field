use std::collections::{HashMap, HashSet};
use crate::ids::MemoryId;
use crate::ops::EMBED_DIM;

/// A recall hit from semantic search.
#[derive(Debug, Clone)]
pub struct SemanticHit {
    pub memory_id: MemoryId,
    pub cosine_similarity: f32,
}

/// Brute-force cosine similarity semantic index over memory embeddings.
///
/// Holds all embeddings in memory. At 50K memories × 768 dims this is ~148 MB
/// and a full scan takes ~300ms — acceptable for a daemon. Real HNSW can be
/// swapped in later by replacing this struct without changing the call sites.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SemanticIndex {
    /// memory_id -> embedding (flat Vec for cache efficiency)
    embeddings: HashMap<MemoryId, Vec<f32>>,
    /// soft-deleted IDs excluded from search results
    deleted: HashSet<MemoryId>,
}

impl SemanticIndex {
    pub fn new() -> Self {
        Self {
            embeddings: HashMap::new(),
            deleted: HashSet::new(),
        }
    }

    /// Add or update an embedding. Un-deletes the entry if it was soft-deleted.
    pub fn upsert(&mut self, memory_id: MemoryId, embedding: Vec<f32>) {
        self.deleted.remove(&memory_id);
        self.embeddings.insert(memory_id, embedding);
    }

    /// Mark a memory as deleted — excluded from future search results.
    pub fn remove(&mut self, memory_id: MemoryId) {
        self.deleted.insert(memory_id);
        self.embeddings.remove(&memory_id);
    }

    /// Search for k nearest neighbours to `query` by cosine similarity.
    ///
    /// `allowed` is an optional pre-filtered set of MemoryIds (realm filter
    /// applied by the caller). Entries absent from `allowed` are skipped.
    /// Returns hits sorted by cosine similarity descending.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        allowed: Option<&HashSet<MemoryId>>,
    ) -> Vec<SemanticHit> {
        if query.len() != EMBED_DIM {
            return vec![];
        }
        let query_norm = l2_norm(query);
        if query_norm < 1e-9 {
            return vec![];
        }

        let mut scored: Vec<(MemoryId, f32)> = self
            .embeddings
            .iter()
            .filter(|(id, _)| !self.deleted.contains(id))
            .filter(|(id, _)| allowed.map(|a| a.contains(id)).unwrap_or(true))
            .map(|(id, emb)| {
                let sim = cosine_similarity_prenormed(query, query_norm, emb);
                (*id, sim)
            })
            .collect();

        scored.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);

        scored
            .into_iter()
            .map(|(memory_id, cosine_similarity)| SemanticHit {
                memory_id,
                cosine_similarity,
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cosine_similarity_prenormed(query: &[f32], query_norm: f32, other: &[f32]) -> f32 {
    if other.len() != query.len() {
        return 0.0;
    }
    let other_norm = l2_norm(other);
    if other_norm < 1e-9 {
        return 0.0;
    }
    let dot: f32 = query.iter().zip(other.iter()).map(|(a, b)| a * b).sum();
    dot / (query_norm * other_norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_returns_nearest() {
        let mut idx = SemanticIndex::new();
        let mut e1 = vec![0.0f32; 768];
        e1[0] = 1.0;
        let mut e2 = vec![0.0f32; 768];
        e2[1] = 1.0;
        let mut e3 = vec![0.0f32; 768];
        e3[0] = 0.9;
        e3[1] = 0.1;
        idx.upsert(1, e1.clone());
        idx.upsert(2, e2);
        idx.upsert(3, e3);

        let hits = idx.search(&e1, 2, None);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].memory_id, 1); // exact match
        // e3 has a larger component on dim 0 (same as query) than e2
        assert_eq!(hits[1].memory_id, 3);
    }

    #[test]
    fn test_deleted_not_returned() {
        let mut idx = SemanticIndex::new();
        let e = vec![1.0f32; 768];
        idx.upsert(1, e.clone());
        idx.upsert(2, e.clone());
        idx.remove(1);
        let hits = idx.search(&e, 10, None);
        assert!(hits.iter().all(|h| h.memory_id != 1));
    }

    #[test]
    fn test_allowed_filter() {
        let mut idx = SemanticIndex::new();
        let e = vec![1.0f32; 768];
        idx.upsert(1, e.clone());
        idx.upsert(2, e.clone());
        idx.upsert(3, e.clone());

        let allowed: HashSet<MemoryId> = [2u64, 3u64].into_iter().collect();
        let hits = idx.search(&e, 10, Some(&allowed));
        assert!(hits.iter().all(|h| h.memory_id != 1));
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn test_wrong_dim_returns_empty() {
        let idx = SemanticIndex::new();
        let hits = idx.search(&[1.0f32; 64], 5, None);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_upsert_overwrites() {
        let mut idx = SemanticIndex::new();
        let mut e1 = vec![0.0f32; 768];
        e1[0] = 1.0;
        let mut e2 = vec![0.0f32; 768];
        e2[1] = 1.0;

        idx.upsert(1, e1);
        idx.upsert(1, e2.clone()); // overwrite

        // Query aligned with dim-1; id=1 should now match well
        let hits = idx.search(&e2, 1, None);
        assert_eq!(hits[0].memory_id, 1);
    }
}

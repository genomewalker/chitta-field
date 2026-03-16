use std::collections::HashMap;
use crate::ids::MemoryId;
use serde::{Serialize, Deserialize};

/// BM25 parameters (standard defaults).
const K1: f32 = 1.2;
const B: f32 = 0.75;

/// A posting: memory_id + term frequency in that document.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Posting {
    memory_id: MemoryId,
    tf: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordIndex {
    /// term -> postings list
    postings: HashMap<String, Vec<Posting>>,
    /// memory_id -> document length (token count)
    doc_lengths: HashMap<MemoryId, u32>,
    /// running total tokens across all documents
    total_tokens: u64,
    total_docs: u32,
}

#[derive(Debug, Clone)]
pub struct KeywordHit {
    pub memory_id: MemoryId,
    pub bm25_score: f32,
}

impl KeywordIndex {
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            total_tokens: 0,
            total_docs: 0,
        }
    }

    /// Index a memory's text content.
    pub fn index(&mut self, memory_id: MemoryId, content: &str) {
        // Remove old postings for this memory if re-indexing.
        self.remove(memory_id);

        let tokens = tokenize(content);
        let doc_len = tokens.len() as u32;

        if doc_len == 0 {
            return;
        }

        // Count term frequencies.
        let mut tf_map: HashMap<String, u32> = HashMap::new();
        for token in tokens {
            *tf_map.entry(token).or_insert(0) += 1;
        }

        // Insert into postings lists.
        for (term, tf) in tf_map {
            self.postings
                .entry(term)
                .or_insert_with(Vec::new)
                .push(Posting { memory_id, tf });
        }

        self.doc_lengths.insert(memory_id, doc_len);
        self.total_tokens += doc_len as u64;
        self.total_docs += 1;
    }

    /// Remove a memory from the index (on forget).
    pub fn remove(&mut self, memory_id: MemoryId) {
        if let Some(old_len) = self.doc_lengths.remove(&memory_id) {
            self.total_tokens = self.total_tokens.saturating_sub(old_len as u64);
            self.total_docs = self.total_docs.saturating_sub(1);

            // Remove postings referencing this memory_id.
            self.postings.retain(|_, postings| {
                postings.retain(|p| p.memory_id != memory_id);
                !postings.is_empty()
            });
        }
    }

    /// BM25 search. Returns hits sorted by score descending.
    pub fn search(&self, query: &str, k: usize) -> Vec<KeywordHit> {
        if self.total_docs == 0 {
            return Vec::new();
        }

        let query_terms = tokenize(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        let n = self.total_docs as f32;
        let avgdl = self.total_tokens as f32 / n;

        let mut scores: HashMap<MemoryId, f32> = HashMap::new();

        for term in &query_terms {
            let postings = match self.postings.get(term) {
                Some(p) => p,
                None => continue,
            };

            let df = postings.len() as f32;
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

            for posting in postings {
                let doc_len = *self.doc_lengths.get(&posting.memory_id).unwrap_or(&1) as f32;
                let tf = posting.tf as f32;
                let tf_norm = tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * doc_len / avgdl));
                *scores.entry(posting.memory_id).or_insert(0.0) += idf * tf_norm;
            }
        }

        let mut hits: Vec<KeywordHit> = scores
            .into_iter()
            .map(|(memory_id, bm25_score)| KeywordHit { memory_id, bm25_score })
            .collect();

        hits.sort_unstable_by(|a, b| {
            b.bm25_score.partial_cmp(&a.bm25_score).unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);
        hits
    }

    pub fn doc_count(&self) -> usize {
        self.total_docs as usize
    }
}

fn tokenize(text: &str) -> Vec<String> {
    let stop_words = [
        "the", "a", "an", "is", "are", "was", "be", "to", "of", "and", "in", "it", "for",
        "on", "with", "as", "at", "by", "or", "not", "this", "that", "from", "have", "has",
        "had", "but", "its", "my", "your", "we", "i", "he", "she", "they", "do", "did",
        "will", "can", "would", "could", "should",
    ];
    text.split(|c: char| !c.is_alphanumeric())
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() >= 2 && !stop_words.contains(&t.as_str()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_search() {
        let mut idx = KeywordIndex::new();
        idx.index(1, "the quick brown fox jumps over the lazy dog");
        idx.index(2, "rust programming language memory safety");
        idx.index(3, "fox and hound friendship story");

        let hits = idx.search("fox", 10);
        assert_eq!(hits.len(), 2);
        let ids: Vec<u64> = hits.iter().map(|h| h.memory_id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
    }

    #[test]
    fn test_idf_boost() {
        let mut idx = KeywordIndex::new();
        // "rust" appears in only doc 2 — should score higher for "rust" query
        idx.index(1, "programming language safety");
        idx.index(2, "rust programming language memory safety");
        idx.index(3, "programming language design");

        let hits = idx.search("rust", 10);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].memory_id, 2);
    }

    #[test]
    fn test_remove() {
        let mut idx = KeywordIndex::new();
        idx.index(1, "hello world foo bar");
        idx.index(2, "hello world baz");
        idx.remove(1);

        let hits = idx.search("hello", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory_id, 2);
    }

    #[test]
    fn test_multi_term() {
        let mut idx = KeywordIndex::new();
        idx.index(1, "cognitive memory system architecture design");
        idx.index(2, "database system architecture sql");
        idx.index(3, "cognitive science brain research");

        // "cognitive architecture" — doc 1 has both, should rank highest
        let hits = idx.search("cognitive architecture", 10);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].memory_id, 1);
    }

    #[test]
    fn test_empty_index() {
        let idx = KeywordIndex::new();
        let hits = idx.search("anything", 10);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_doc_count() {
        let mut idx = KeywordIndex::new();
        assert_eq!(idx.doc_count(), 0);
        idx.index(1, "hello world");
        assert_eq!(idx.doc_count(), 1);
        idx.index(2, "foo bar");
        assert_eq!(idx.doc_count(), 2);
        idx.remove(1);
        assert_eq!(idx.doc_count(), 1);
    }

    #[test]
    fn test_reindex() {
        let mut idx = KeywordIndex::new();
        idx.index(1, "hello world");
        // Re-index same id with different content.
        idx.index(1, "rust programming");
        assert_eq!(idx.doc_count(), 1);

        // "hello" should no longer match doc 1.
        let hits = idx.search("hello", 10);
        assert!(hits.is_empty());

        // "rust" should match doc 1.
        let hits = idx.search("rust", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory_id, 1);
    }

    #[test]
    fn test_stop_words_filtered() {
        let mut idx = KeywordIndex::new();
        // "the", "is", "a" are stop words and should not be indexed.
        idx.index(1, "the sky is a deep blue");

        // Searching for stop words should return nothing.
        let hits = idx.search("the", 10);
        assert!(hits.is_empty());

        // Non-stop content still matches.
        let hits = idx.search("sky blue", 10);
        assert_eq!(hits.len(), 1);
    }
}

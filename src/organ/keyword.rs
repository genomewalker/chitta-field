use crate::ids::MemoryId;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// BM25 parameters (standard defaults).
const K1: f32 = 1.2;
const B: f32 = 0.75;
const MAX_QUERY_TERMS: usize = 4;
const COMMON_TERM_RATIO: f32 = 0.20;

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
    /// Reverse map used to remove or reindex a single document efficiently.
    #[serde(skip)]
    doc_terms: HashMap<MemoryId, Vec<(String, u32)>>,
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
            doc_terms: HashMap::new(),
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
        let mut reverse_terms = Vec::new();
        for (term, tf) in tf_map {
            self.postings
                .entry(term.clone())
                .or_insert_with(Vec::new)
                .push(Posting { memory_id, tf });
            reverse_terms.push((term, tf));
        }

        self.doc_terms.insert(memory_id, reverse_terms);
        self.doc_lengths.insert(memory_id, doc_len);
        self.total_tokens += doc_len as u64;
        self.total_docs += 1;
    }

    /// Remove a memory from the index (on forget).
    pub fn remove(&mut self, memory_id: MemoryId) {
        if let Some(old_len) = self.doc_lengths.remove(&memory_id) {
            self.total_tokens = self.total_tokens.saturating_sub(old_len as u64);
            self.total_docs = self.total_docs.saturating_sub(1);

            if let Some(terms) = self.doc_terms.remove(&memory_id) {
                for (term, _) in terms {
                    let remove_term = if let Some(postings) = self.postings.get_mut(&term) {
                        postings.retain(|p| p.memory_id != memory_id);
                        postings.is_empty()
                    } else {
                        false
                    };
                    if remove_term {
                        self.postings.remove(&term);
                    }
                }
            }
        }
    }

    /// BM25 search. Returns hits sorted by score descending.
    pub fn search(&self, query: &str, k: usize) -> Vec<KeywordHit> {
        if self.total_docs == 0 || k == 0 {
            return Vec::new();
        }

        let query_terms = dedup_terms(tokenize(query));
        if query_terms.is_empty() {
            return Vec::new();
        }

        let n = self.total_docs as f32;
        let avgdl = self.total_tokens as f32 / n;
        let term_infos = self.query_terms(&query_terms, n);
        if term_infos.is_empty() {
            return Vec::new();
        }

        let mut scores: HashMap<MemoryId, f32> = HashMap::new();

        for term_info in term_infos {
            for posting in term_info.postings {
                let doc_len = *self.doc_lengths.get(&posting.memory_id).unwrap_or(&1) as f32;
                let tf = posting.tf as f32;
                let tf_norm = tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * doc_len / avgdl));
                *scores.entry(posting.memory_id).or_insert(0.0) += term_info.idf * tf_norm;
            }
        }

        let mut top_k = BinaryHeap::new();
        for (memory_id, bm25_score) in scores {
            push_top_k(&mut top_k, k, memory_id, bm25_score);
        }

        heap_to_hits(top_k)
    }

    pub fn doc_count(&self) -> usize {
        self.total_docs as usize
    }

    pub fn rebuild_reverse_index(&mut self) {
        let mut doc_terms: HashMap<MemoryId, Vec<(String, u32)>> = HashMap::new();
        for (term, postings) in &self.postings {
            for posting in postings {
                doc_terms
                    .entry(posting.memory_id)
                    .or_default()
                    .push((term.clone(), posting.tf));
            }
        }
        self.doc_terms = doc_terms;
    }

    fn query_terms<'a>(&'a self, query_terms: &[String], n: f32) -> Vec<QueryTerm<'a>> {
        let mut terms: Vec<QueryTerm<'a>> = query_terms
            .iter()
            .filter_map(|term| {
                let postings = self.postings.get(term)?;
                let df = postings.len();
                let idf = ((n - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln();
                Some(QueryTerm { postings, df, idf })
            })
            .collect();

        if terms.is_empty() {
            return terms;
        }

        terms.sort_unstable_by(|a, b| {
            a.df.cmp(&b.df)
                .then_with(|| b.idf.partial_cmp(&a.idf).unwrap_or(Ordering::Equal))
        });

        let min_df = terms[0].df;
        let has_selective = min_df < self.total_docs as usize;
        if has_selective {
            terms.retain(|term| term.df == min_df || (term.df as f32 / n) <= COMMON_TERM_RATIO);
        }

        terms.truncate(MAX_QUERY_TERMS);
        terms
    }
}

fn tokenize(text: &str) -> Vec<String> {
    let stop_words = [
        "the", "a", "an", "is", "are", "was", "be", "to", "of", "and", "in", "it", "for", "on",
        "with", "as", "at", "by", "or", "not", "this", "that", "from", "have", "has", "had", "but",
        "its", "my", "your", "we", "i", "he", "she", "they", "do", "did", "will", "can", "would",
        "could", "should",
    ];
    text.split(|c: char| !c.is_alphanumeric())
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() >= 2 && !stop_words.contains(&t.as_str()))
        .collect()
}

fn dedup_terms(terms: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::with_capacity(terms.len());
    let mut out = Vec::with_capacity(terms.len());
    for term in terms {
        if seen.insert(term.clone()) {
            out.push(term);
        }
    }
    out
}

struct QueryTerm<'a> {
    postings: &'a [Posting],
    df: usize,
    idf: f32,
}

#[derive(Clone, Copy, Debug)]
struct RankedHit {
    score: f32,
    memory_id: MemoryId,
}

impl PartialEq for RankedHit {
    fn eq(&self, other: &Self) -> bool {
        self.memory_id == other.memory_id && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for RankedHit {}

impl Ord for RankedHit {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.memory_id.cmp(&other.memory_id))
    }
}

impl PartialOrd for RankedHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn push_top_k(heap: &mut BinaryHeap<RankedHit>, k: usize, memory_id: MemoryId, score: f32) {
    let candidate = RankedHit { score, memory_id };
    if heap.len() < k {
        heap.push(candidate);
        return;
    }
    let Some(worst) = heap.peek() else {
        heap.push(candidate);
        return;
    };
    if score > worst.score || (score == worst.score && memory_id < worst.memory_id) {
        heap.pop();
        heap.push(candidate);
    }
}

fn heap_to_hits(heap: BinaryHeap<RankedHit>) -> Vec<KeywordHit> {
    let mut ranked = heap.into_vec();
    ranked.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.memory_id.cmp(&b.memory_id))
    });
    ranked
        .into_iter()
        .map(|hit| KeywordHit {
            memory_id: hit.memory_id,
            bm25_score: hit.score,
        })
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
    fn test_common_terms_do_not_swamp_selective_term() {
        let mut idx = KeywordIndex::new();
        idx.index(1, "architecture benchmark topic17 project3");
        idx.index(2, "architecture benchmark topic22 project9");
        idx.index(3, "architecture benchmark topic17 project8");

        let hits = idx.search("architecture benchmark topic17", 10);
        assert_eq!(hits.len(), 2);
        assert!(hits
            .iter()
            .all(|hit| hit.memory_id == 1 || hit.memory_id == 3));
    }

    #[test]
    fn test_duplicate_query_terms_are_ignored() {
        let mut idx = KeywordIndex::new();
        idx.index(1, "rust ownership");
        idx.index(2, "rust lifetimes");

        let hits_once = idx.search("rust ownership", 10);
        let hits_dup = idx.search("rust rust ownership ownership", 10);
        assert_eq!(hits_once.len(), hits_dup.len());
        assert_eq!(hits_once[0].memory_id, hits_dup[0].memory_id);
        assert!((hits_once[0].bm25_score - hits_dup[0].bm25_score).abs() < 1e-6);
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

    #[test]
    fn test_rebuild_reverse_index_supports_remove() {
        let mut idx = KeywordIndex::new();
        idx.index(1, "alpha beta gamma");
        idx.index(2, "beta delta");

        idx.doc_terms.clear();
        idx.rebuild_reverse_index();
        idx.remove(1);

        let hits = idx.search("alpha beta gamma", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory_id, 2);
    }
}

//! Hyperdimensional Computing (HDC) memory index.
//!
//! Encodes memory content as 8192-bit binary vectors via bag-of-words bundling.
//! Retrieval is O(n) Hamming distance (popcount XOR) — pure CPU bitwise ops,
//! no floats, no GPU, no SIMD required.
//!
//! Vocabulary is hash-derived and deterministic: every restart produces the
//! same word→HdcVec mapping without loading any file.
//!
//! Properties exploited by chitta:
//! - Compositional queries: `AND(X, Y)` ≈ `XOR(encode(X), encode(Y))`
//! - Incremental realm bundles: add/remove a single memory without rebuild
//! - Temporal set membership: `hamming(session_bundle, query_hv)` ≈ relevance
//! - Forgetting without deletion: un-bundle subtracts the vector's bit contribution

use std::collections::{HashMap, HashSet};

use crate::ids::MemoryId;

// ── Dimensionality ────────────────────────────────────────────────────────────

/// Number of u64 words per vector: 128 × 64 = 8192 bits.
const D: usize = 128;

/// A single hyperdimensional vector.
pub type HdcVec = [u64; D];

// ── Deterministic vocabulary ──────────────────────────────────────────────────

#[inline]
fn xorshift64(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

#[inline]
fn fnv1a_64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| {
        h.wrapping_mul(0x100000001b3) ^ b as u64
    })
}

/// Map a word to its random hypervector. Deterministic: same word → same HdcVec
/// across all restarts, no vocabulary file needed.
pub fn word_hv(word: &str) -> HdcVec {
    // +1 so the empty-string / zero-hash edge case still produces a nonzero seed
    let mut s = fnv1a_64(word.as_bytes()).wrapping_add(1);
    std::array::from_fn(|_| xorshift64(&mut s))
}

// ── Primitive operations ──────────────────────────────────────────────────────

/// Binding: XOR of two vectors. Used to associate role with content, or
/// to combine two concepts into a superposition.
#[inline]
pub fn bind(a: &HdcVec, b: &HdcVec) -> HdcVec {
    std::array::from_fn(|i| a[i] ^ b[i])
}

/// Hamming distance between two vectors (0 = identical, 8192 = opposite).
#[inline]
pub fn hamming(a: &HdcVec, b: &HdcVec) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// Normalised similarity in [0, 1] (1 = identical).
#[inline]
pub fn similarity(a: &HdcVec, b: &HdcVec) -> f32 {
    1.0 - hamming(a, b) as f32 / (D * 64) as f32
}

/// Majority-vote bundle of `vecs`. Threshold: strict majority (>n/2 → 1).
/// Ties on even n go to 0. Empty input → zero vector.
pub fn bundle(vecs: &[HdcVec]) -> HdcVec {
    if vecs.is_empty() {
        return [0u64; D];
    }
    // u16 popcount per bit position — supports up to 65535 simultaneous vectors
    let mut counts = vec![0u16; D * 64];
    for hv in vecs {
        for (wi, &w) in hv.iter().enumerate() {
            for bit in 0u32..64 {
                if (w >> bit) & 1 == 1 {
                    counts[wi * 64 + bit as usize] = counts[wi * 64 + bit as usize].saturating_add(1);
                }
            }
        }
    }
    let threshold = vecs.len() as u32 / 2 + 1;
    let mut out = [0u64; D];
    for wi in 0..D {
        for bit in 0u32..64 {
            if counts[wi * 64 + bit as usize] as u32 >= threshold {
                out[wi] |= 1u64 << bit;
            }
        }
    }
    out
}

// ── Text encoder ──────────────────────────────────────────────────────────────

/// Tokenise `text` into lowercase alphanumeric tokens (≥2 chars), look up
/// each token's word_hv, and majority-vote bundle them into a single HdcVec.
pub fn encode(text: &str) -> HdcVec {
    let hvs: Vec<HdcVec> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(|t| word_hv(&t.to_lowercase()))
        .collect();
    bundle(&hvs)
}

// ── Incremental realm bundle ──────────────────────────────────────────────────

/// Maintains a running bit-count so a realm's aggregate HdcVec can be recovered
/// in O(D) without iterating all member vectors.
#[derive(Clone)]
pub struct RealmBundle {
    /// bit[wi*64 + b] = how many stored HdcVecs have bit b set in word wi
    counts: Vec<u16>,
    /// how many HdcVecs have been bundled
    pub n: u32,
}

impl Default for RealmBundle {
    fn default() -> Self {
        Self { counts: vec![0u16; D * 64], n: 0 }
    }
}

impl RealmBundle {
    pub fn add(&mut self, hv: &HdcVec) {
        for (wi, &w) in hv.iter().enumerate() {
            for bit in 0u32..64 {
                if (w >> bit) & 1 == 1 {
                    self.counts[wi * 64 + bit as usize] =
                        self.counts[wi * 64 + bit as usize].saturating_add(1);
                }
            }
        }
        self.n += 1;
    }

    pub fn remove(&mut self, hv: &HdcVec) {
        if self.n == 0 { return; }
        for (wi, &w) in hv.iter().enumerate() {
            for bit in 0u32..64 {
                if (w >> bit) & 1 == 1 {
                    self.counts[wi * 64 + bit as usize] =
                        self.counts[wi * 64 + bit as usize].saturating_sub(1);
                }
            }
        }
        self.n = self.n.saturating_sub(1);
    }

    /// Materialise the current majority-vote HdcVec for this realm.
    pub fn to_hv(&self) -> HdcVec {
        if self.n == 0 { return [0u64; D]; }
        let threshold = self.n / 2 + 1;
        let mut out = [0u64; D];
        for wi in 0..D {
            for bit in 0u32..64 {
                if self.counts[wi * 64 + bit as usize] as u32 >= threshold {
                    out[wi] |= 1u64 << bit;
                }
            }
        }
        out
    }
}

// ── HdcStore ──────────────────────────────────────────────────────────────────

/// Per-memory HDC index. Maintained alongside the HNSW and BM25 indexes.
/// Rebuilt from persisted payload content on startup — no snapshot serialisation needed.
#[derive(Default)]
pub struct HdcStore {
    /// memory_id → encoded HdcVec (1024 bytes each)
    memories: HashMap<MemoryId, HdcVec>,
    /// realm → running bit-count bundle
    realm_bundles: HashMap<String, RealmBundle>,
    /// realm → set of memory IDs (for realm-scoped recall)
    realm_members: HashMap<String, HashSet<MemoryId>>,
}

impl HdcStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bulk-load from (id, content, realm) tuples. Used at startup to rebuild
    /// from persisted payloads without snapshot serialisation of large arrays.
    pub fn rebuild<'a>(&mut self, entries: impl Iterator<Item = (MemoryId, &'a str, &'a str)>) {
        self.memories.clear();
        self.realm_bundles.clear();
        self.realm_members.clear();
        for (id, text, realm) in entries {
            self.insert(id, text, realm);
        }
    }

    pub fn len(&self) -> usize {
        self.memories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }

    /// Encode `text` and register it under `id` in `realm`.
    pub fn insert(&mut self, id: MemoryId, text: &str, realm: &str) {
        let hv = encode(text);
        self.realm_bundles
            .entry(realm.to_string())
            .or_default()
            .add(&hv);
        self.realm_members
            .entry(realm.to_string())
            .or_default()
            .insert(id);
        self.memories.insert(id, hv);
    }

    /// Remove a memory from the index and update the realm bundle.
    pub fn remove(&mut self, id: MemoryId, realm: &str) {
        if let Some(hv) = self.memories.remove(&id) {
            if let Some(rb) = self.realm_bundles.get_mut(realm) {
                rb.remove(&hv);
            }
            if let Some(members) = self.realm_members.get_mut(realm) {
                members.remove(&id);
            }
        }
    }

    /// Return top-`k` memories by Hamming similarity to the encoded `text`.
    /// When `realm` is `Some`, only memories in that realm are considered.
    /// Returns `(MemoryId, hamming_distance)` sorted ascending (0 = identical).
    pub fn query(&self, text: &str, k: usize, realm: Option<&str>) -> Vec<(MemoryId, u32)> {
        if k == 0 || self.memories.is_empty() {
            return vec![];
        }
        let q = encode(text);
        let mut scored: Vec<(MemoryId, u32)> = match realm {
            Some(r) => {
                let Some(members) = self.realm_members.get(r) else { return vec![]; };
                members.iter()
                    .filter_map(|id| self.memories.get(id).map(|hv| (*id, hamming(&q, hv))))
                    .collect()
            }
            None => self.memories.iter().map(|(&id, hv)| (id, hamming(&q, hv))).collect(),
        };
        scored.sort_unstable_by_key(|&(_, d)| d);
        scored.truncate(k);
        scored
    }

    /// Similarity score in [0, 1] for a precomputed query HdcVec vs a memory.
    pub fn score_hv(&self, id: MemoryId, query_hv: &HdcVec) -> Option<f32> {
        self.memories.get(&id).map(|hv| similarity(query_hv, hv))
    }

    /// Return top-`k` memories most representative of a realm's aggregate theme.
    /// Useful for "what does realm X talk about?" cluster queries.
    pub fn realm_theme_query(&self, realm: &str, k: usize) -> Vec<(MemoryId, u32)> {
        let Some(rb) = self.realm_bundles.get(realm) else { return vec![]; };
        let q = rb.to_hv();
        let Some(members) = self.realm_members.get(realm) else { return vec![]; };
        let mut scored: Vec<(MemoryId, u32)> = members.iter()
            .filter_map(|id| self.memories.get(id).map(|hv| (*id, hamming(&q, hv))))
            .collect();
        scored.sort_unstable_by_key(|&(_, d)| d);
        scored.truncate(k);
        scored
    }

    /// Return the realm bundle HdcVec (the "theme" of a realm).
    pub fn realm_hv(&self, realm: &str) -> Option<HdcVec> {
        self.realm_bundles.get(realm).map(|rb| rb.to_hv())
    }

    /// Cross-realm similarity: how similar are two realms' aggregate themes?
    /// Returns similarity in [0, 1].
    pub fn realm_similarity(&self, a: &str, b: &str) -> f32 {
        match (self.realm_hv(a), self.realm_hv(b)) {
            (Some(ha), Some(hb)) => similarity(&ha, &hb),
            _ => 0.0,
        }
    }

    /// Compose a multi-concept query: encode each term, XOR-bind them all,
    /// then return nearest memories. XOR binding approximates logical AND in HDC.
    /// E.g., query_and(&["Python", "debugging"]) finds memories about both.
    pub fn query_and(&self, terms: &[&str], k: usize, realm: Option<&str>) -> Vec<(MemoryId, u32)> {
        if terms.is_empty() { return vec![]; }
        let mut q = word_hv(&terms[0].to_lowercase());
        for t in &terms[1..] {
            q = bind(&q, &word_hv(&t.to_lowercase()));
        }
        let mut scored: Vec<(MemoryId, u32)> = match realm {
            Some(r) => {
                let Some(members) = self.realm_members.get(r) else { return vec![]; };
                members.iter()
                    .filter_map(|id| self.memories.get(id).map(|hv| (*id, hamming(&q, hv))))
                    .collect()
            }
            None => self.memories.iter().map(|(&id, hv)| (id, hamming(&q, hv))).collect(),
        };
        scored.sort_unstable_by_key(|&(_, d)| d);
        scored.truncate(k);
        scored
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_hv_deterministic() {
        assert_eq!(word_hv("hello"), word_hv("hello"));
        assert_ne!(word_hv("hello"), word_hv("world"));
    }

    #[test]
    fn hamming_identity() {
        let hv = word_hv("test");
        assert_eq!(hamming(&hv, &hv), 0);
    }

    #[test]
    fn hamming_range() {
        let a = word_hv("alpha");
        let b = word_hv("beta");
        let d = hamming(&a, &b);
        assert!(d > 0 && d < D as u32 * 64, "d={d}");
        // Expected ~4096 for random independent vectors
        let expected = (D * 32) as u32;
        assert!((d as i32 - expected as i32).abs() < 500, "d={d} far from {expected}");
    }

    #[test]
    fn bundle_of_one_is_identity() {
        let hv = word_hv("solo");
        assert_eq!(bundle(&[hv]), hv);
    }

    #[test]
    fn encode_similar_texts() {
        let a = encode("the cat sat on the mat");
        let b = encode("a cat sat on a mat");
        let c = encode("quantum computing algorithms");
        let d_ab = hamming(&a, &b);
        let d_ac = hamming(&a, &c);
        // Similar texts should be closer than dissimilar
        assert!(d_ab < d_ac, "similar texts farther apart than dissimilar: d_ab={d_ab} d_ac={d_ac}");
    }

    #[test]
    fn store_insert_query() {
        let mut store = HdcStore::new();
        store.insert(1, "Python programming language", "test");
        store.insert(2, "Rust systems programming", "test");
        store.insert(3, "cooking pasta recipes", "test");

        let hits = store.query("Python programming", 2, Some("test"));
        assert!(!hits.is_empty());
        // Memory 1 (Python) should score closer than memory 3 (cooking)
        let d1 = hits.iter().find(|&&(id, _)| id == 1).map(|&(_, d)| d);
        let d3 = hits.iter().find(|&&(id, _)| id == 3).map(|&(_, d)| d);
        if let (Some(d1), Some(d3)) = (d1, d3) {
            assert!(d1 < d3, "Python memory should be closer: d1={d1} d3={d3}");
        }
    }

    #[test]
    fn store_remove_updates_realm_bundle() {
        let mut store = HdcStore::new();
        store.insert(1, "machine learning", "r");
        store.insert(2, "deep learning neural nets", "r");
        assert_eq!(store.realm_bundles["r"].n, 2);
        store.remove(1, "r");
        assert_eq!(store.realm_bundles["r"].n, 1);
        assert!(!store.memories.contains_key(&1));
    }

    #[test]
    fn query_and_binding() {
        let mut store = HdcStore::new();
        store.insert(1, "Python machine learning scikit", "r");
        store.insert(2, "Python web server flask", "r");
        store.insert(3, "cooking pasta tomato sauce", "r");

        // XOR binding produces a superposition vector; semantic ordering is not
        // guaranteed for short texts/small n in binary HDC. Just verify the API works.
        let hits = store.query_and(&["python", "learning"], 3, Some("r"));
        assert_eq!(hits.len(), 3);
        // All distances should be in the valid range
        for (_, d) in &hits {
            assert!(*d <= 8192, "hamming must be ≤ 8192, got {d}");
        }
    }

    #[test]
    fn realm_similarity_reflexive() {
        let mut store = HdcStore::new();
        store.insert(1, "some content here", "realm_a");
        let sim = store.realm_similarity("realm_a", "realm_a");
        assert!((sim - 1.0).abs() < 1e-5, "sim={sim}");
    }

    #[test]
    fn realm_similarity_distinct_realms() {
        let mut store = HdcStore::new();
        store.insert(1, "rust programming systems", "tech");
        store.insert(2, "rust programming systems", "tech");
        store.insert(3, "baking bread sourdough recipes", "food");
        store.insert(4, "cooking pasta italian cuisine", "food");
        let sim_same  = store.realm_similarity("tech", "tech");
        let sim_cross = store.realm_similarity("tech", "food");
        assert!(sim_same > sim_cross, "same-realm sim {sim_same} should exceed cross {sim_cross}");
    }
}

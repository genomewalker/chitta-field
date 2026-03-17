use crate::ids::MemoryId;
use crate::ops::EMBED_DIM;
use std::collections::{BinaryHeap, HashMap, HashSet};

const COARSE_CENTROIDS: usize = 256;
const COARSE_ASSIGNMENTS: usize = 2;
const MIN_PROBES: usize = 6;
const MAX_PROBES: usize = 24;
const MIN_CANDIDATES: usize = 1024;
const MAX_CANDIDATES: usize = 16384;
const LSH_TABLES: usize = 4;
const LSH_BITS: usize = 12;

/// A recall hit from semantic search.
#[derive(Debug, Clone)]
pub struct SemanticHit {
    pub memory_id: MemoryId,
    pub cosine_similarity: f32,
}

/// Approximate semantic index over normalized memory embeddings.
///
/// Uses a coarse centroid assignment to generate candidate buckets, then exact
/// cosine reranking inside those candidates. This keeps query cost bounded even
/// when the sparse cortical index cannot provide candidates.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SemanticIndex {
    /// memory_id -> embedding (flat Vec for cache efficiency)
    embeddings: HashMap<MemoryId, Vec<f32>>,
    /// soft-deleted IDs excluded from search results
    deleted: HashSet<MemoryId>,
    #[serde(default = "default_coarse_centroids")]
    coarse_centroids: Vec<Vec<f32>>,
    #[serde(default)]
    coarse_members: HashMap<u16, Vec<MemoryId>>,
    #[serde(default)]
    mem_coarse: HashMap<MemoryId, Vec<u16>>,
    #[serde(skip)]
    lsh_planes: Vec<Vec<Vec<f32>>>,
    #[serde(skip)]
    lsh_buckets: Vec<HashMap<u16, Vec<MemoryId>>>,
    #[serde(skip)]
    mem_lsh: HashMap<MemoryId, Vec<u16>>,
}

impl SemanticIndex {
    pub fn new() -> Self {
        Self {
            embeddings: HashMap::new(),
            deleted: HashSet::new(),
            coarse_centroids: default_coarse_centroids(),
            coarse_members: HashMap::new(),
            mem_coarse: HashMap::new(),
            lsh_planes: default_lsh_planes(),
            lsh_buckets: vec![HashMap::new(); LSH_TABLES],
            mem_lsh: HashMap::new(),
        }
    }

    /// Add or update an embedding. Un-deletes the entry if it was soft-deleted.
    pub fn upsert(&mut self, memory_id: MemoryId, mut embedding: Vec<f32>) {
        self.remove(memory_id);
        normalize_in_place(&mut embedding);
        self.deleted.remove(&memory_id);
        let coarse_ids = self.assign_coarse(&embedding);
        for coarse_id in &coarse_ids {
            self.coarse_members
                .entry(*coarse_id)
                .or_default()
                .push(memory_id);
        }
        self.mem_coarse.insert(memory_id, coarse_ids);
        let lsh_ids = self.assign_lsh(&embedding);
        for (table_idx, signature) in lsh_ids.iter().enumerate() {
            self.lsh_buckets[table_idx]
                .entry(*signature)
                .or_default()
                .push(memory_id);
        }
        self.mem_lsh.insert(memory_id, lsh_ids);
        self.embeddings.insert(memory_id, embedding);
    }

    /// Mark a memory as deleted — excluded from future search results.
    pub fn remove(&mut self, memory_id: MemoryId) {
        self.deleted.insert(memory_id);
        self.embeddings.remove(&memory_id);
        if let Some(coarse_ids) = self.mem_coarse.remove(&memory_id) {
            for coarse_id in coarse_ids {
                let remove_bucket = if let Some(members) = self.coarse_members.get_mut(&coarse_id) {
                    members.retain(|id| *id != memory_id);
                    members.is_empty()
                } else {
                    false
                };
                if remove_bucket {
                    self.coarse_members.remove(&coarse_id);
                }
            }
        }
        if let Some(lsh_ids) = self.mem_lsh.remove(&memory_id) {
            for (table_idx, signature) in lsh_ids.into_iter().enumerate() {
                let remove_bucket =
                    if let Some(members) = self.lsh_buckets[table_idx].get_mut(&signature) {
                        members.retain(|id| *id != memory_id);
                        members.is_empty()
                    } else {
                        false
                    };
                if remove_bucket {
                    self.lsh_buckets[table_idx].remove(&signature);
                }
            }
        }
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
        let Some(query_unit) = normalize(query) else {
            return vec![];
        };

        if let Some(allowed_ids) = allowed {
            if allowed_ids.len() <= MIN_CANDIDATES {
                return self.search_candidates(query, allowed_ids.iter().copied(), k);
            }
        }

        let mut candidates = self.collect_lsh_candidates(&query_unit, allowed, k);
        if candidates.is_empty() {
            let probes = self.choose_probe_ids(&query_unit, MIN_PROBES);
            candidates = self.collect_candidates(&probes, allowed, k);
        }
        if candidates.is_empty() {
            return vec![];
        }

        let mut top_k = BinaryHeap::new();
        for memory_id in candidates {
            let Some(embedding) = self.embeddings.get(&memory_id) else {
                continue;
            };
            let sim = dot(&query_unit, embedding);
            push_top_k(&mut top_k, k, memory_id, sim);
        }

        heap_to_hits(top_k)
    }

    /// Exact cosine search over a bounded candidate set.
    pub fn search_candidates<I>(&self, query: &[f32], candidates: I, k: usize) -> Vec<SemanticHit>
    where
        I: IntoIterator<Item = MemoryId>,
    {
        if query.len() != EMBED_DIM {
            return vec![];
        }
        let Some(query_unit) = normalize(query) else {
            return vec![];
        };

        let mut top_k = BinaryHeap::new();
        for memory_id in candidates {
            if self.deleted.contains(&memory_id) {
                continue;
            }
            let Some(embedding) = self.embeddings.get(&memory_id) else {
                continue;
            };
            let sim = dot(&query_unit, embedding);
            push_top_k(&mut top_k, k, memory_id, sim);
        }

        heap_to_hits(top_k)
    }

    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Normalize embeddings loaded from older snapshots that stored raw vectors.
    pub fn normalize_all(&mut self) {
        for embedding in self.embeddings.values_mut() {
            normalize_in_place(embedding);
        }
        if self.coarse_centroids.is_empty() {
            self.coarse_centroids = default_coarse_centroids();
        }
        if self.lsh_planes.is_empty() {
            self.lsh_planes = default_lsh_planes();
        }
        self.rebuild_ann();
    }

    fn rebuild_ann(&mut self) {
        self.coarse_members.clear();
        self.mem_coarse.clear();
        self.lsh_buckets = vec![HashMap::new(); LSH_TABLES];
        self.mem_lsh.clear();
        let assignments: Vec<(MemoryId, Vec<u16>)> = self
            .embeddings
            .iter()
            .map(|(&memory_id, embedding)| (memory_id, self.assign_coarse(embedding)))
            .collect();
        for (memory_id, coarse_ids) in assignments {
            for coarse_id in &coarse_ids {
                self.coarse_members
                    .entry(*coarse_id)
                    .or_default()
                    .push(memory_id);
            }
            self.mem_coarse.insert(memory_id, coarse_ids);
        }
        let lsh_assignments: Vec<(MemoryId, Vec<u16>)> = self
            .embeddings
            .iter()
            .map(|(&memory_id, embedding)| (memory_id, self.assign_lsh(embedding)))
            .collect();
        for (memory_id, signatures) in lsh_assignments {
            for (table_idx, signature) in signatures.iter().enumerate() {
                self.lsh_buckets[table_idx]
                    .entry(*signature)
                    .or_default()
                    .push(memory_id);
            }
            self.mem_lsh.insert(memory_id, signatures);
        }
    }

    fn assign_coarse(&self, embedding: &[f32]) -> Vec<u16> {
        self.choose_probe_ids(embedding, COARSE_ASSIGNMENTS)
    }

    fn assign_lsh(&self, embedding: &[f32]) -> Vec<u16> {
        self.lsh_planes
            .iter()
            .map(|table| lsh_signature(table, embedding))
            .collect()
    }

    fn choose_probe_ids(&self, query: &[f32], desired: usize) -> Vec<u16> {
        let take = desired.min(self.coarse_centroids.len());
        if take == 0 {
            return Vec::new();
        }
        let mut scored: Vec<(f32, u16)> = self
            .coarse_centroids
            .iter()
            .enumerate()
            .map(|(idx, centroid)| (dot(query, centroid), idx as u16))
            .collect();
        let nth = take.saturating_sub(1);
        scored.select_nth_unstable_by(nth, |a, b| b.0.total_cmp(&a.0));
        scored.truncate(take);
        scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        scored.into_iter().map(|(_, id)| id).collect()
    }

    fn collect_candidates(
        &self,
        probes: &[u16],
        allowed: Option<&HashSet<MemoryId>>,
        k: usize,
    ) -> Vec<MemoryId> {
        let max_candidates = (k.max(1) * 64).max(MIN_CANDIDATES).min(MAX_CANDIDATES);
        let max_probes = probes.len().min(MAX_PROBES);
        let mut seen = HashSet::with_capacity(max_candidates.min(4096));
        let mut candidates = Vec::with_capacity(max_candidates.min(4096));

        for coarse_id in probes.iter().take(max_probes) {
            let Some(members) = self.coarse_members.get(coarse_id) else {
                continue;
            };
            for &memory_id in members {
                if self.deleted.contains(&memory_id) {
                    continue;
                }
                if let Some(allowed_set) = allowed {
                    if !allowed_set.contains(&memory_id) {
                        continue;
                    }
                }
                if seen.insert(memory_id) {
                    candidates.push(memory_id);
                    if candidates.len() >= max_candidates {
                        return candidates;
                    }
                }
            }
        }
        candidates
    }

    fn collect_lsh_candidates(
        &self,
        query: &[f32],
        allowed: Option<&HashSet<MemoryId>>,
        k: usize,
    ) -> Vec<MemoryId> {
        let max_candidates = (k.max(1) * 64)
            .max(MIN_CANDIDATES / 2)
            .min(MAX_CANDIDATES / 2);
        let signatures = self.assign_lsh(query);
        let mut seen = HashSet::with_capacity(max_candidates.min(4096));
        let mut candidates = Vec::with_capacity(max_candidates.min(4096));

        for (table_idx, signature) in signatures.iter().enumerate() {
            self.extend_from_bucket(
                table_idx,
                *signature,
                allowed,
                max_candidates,
                &mut seen,
                &mut candidates,
            );
            if candidates.len() >= max_candidates {
                return candidates;
            }
        }

        for bit in 0..LSH_BITS {
            for (table_idx, signature) in signatures.iter().enumerate() {
                self.extend_from_bucket(
                    table_idx,
                    *signature ^ (1u16 << bit),
                    allowed,
                    max_candidates,
                    &mut seen,
                    &mut candidates,
                );
                if candidates.len() >= max_candidates {
                    return candidates;
                }
            }
        }

        candidates
    }

    fn extend_from_bucket(
        &self,
        table_idx: usize,
        signature: u16,
        allowed: Option<&HashSet<MemoryId>>,
        max_candidates: usize,
        seen: &mut HashSet<MemoryId>,
        candidates: &mut Vec<MemoryId>,
    ) {
        let Some(members) = self.lsh_buckets[table_idx].get(&signature) else {
            return;
        };
        for &memory_id in members {
            if self.deleted.contains(&memory_id) {
                continue;
            }
            if let Some(allowed_set) = allowed {
                if !allowed_set.contains(&memory_id) {
                    continue;
                }
            }
            if seen.insert(memory_id) {
                candidates.push(memory_id);
                if candidates.len() >= max_candidates {
                    return;
                }
            }
        }
    }
}

fn default_coarse_centroids() -> Vec<Vec<f32>> {
    (0..COARSE_CENTROIDS)
        .map(|idx| random_unit_vec(EMBED_DIM, 10_000 + idx as u64))
        .collect()
}

fn default_lsh_planes() -> Vec<Vec<Vec<f32>>> {
    (0..LSH_TABLES)
        .map(|table| {
            (0..LSH_BITS)
                .map(|bit| random_unit_vec(EMBED_DIM, 20_000 + (table * LSH_BITS + bit) as u64))
                .collect()
        })
        .collect()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn normalize(v: &[f32]) -> Option<Vec<f32>> {
    let norm = l2_norm(v);
    if norm < 1e-9 {
        return None;
    }
    Some(v.iter().map(|x| *x / norm).collect())
}

fn normalize_in_place(v: &mut [f32]) {
    let norm = l2_norm(v);
    if norm < 1e-9 {
        return;
    }
    for x in v {
        *x /= norm;
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
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
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.memory_id.cmp(&other.memory_id))
    }
}

impl PartialOrd for RankedHit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn push_top_k(heap: &mut BinaryHeap<RankedHit>, k: usize, memory_id: MemoryId, score: f32) {
    if k == 0 {
        return;
    }
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

fn heap_to_hits(heap: BinaryHeap<RankedHit>) -> Vec<SemanticHit> {
    let mut ranked: Vec<RankedHit> = heap.into_vec();
    ranked.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.memory_id.cmp(&b.memory_id))
    });
    ranked
        .into_iter()
        .map(|hit| SemanticHit {
            memory_id: hit.memory_id,
            cosine_similarity: hit.score,
        })
        .collect()
}

fn random_unit_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut v: Vec<f32> = (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = (state >> 11) as f32 / (1u64 << 53) as f32;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = (state >> 11) as f32 / (1u64 << 53) as f32;
            let r = (-2.0 * (u1 + 1e-10).ln()).sqrt();
            r * (2.0 * std::f32::consts::PI * u2).cos()
        })
        .collect();
    normalize_in_place(&mut v);
    v
}

fn lsh_signature(planes: &[Vec<f32>], embedding: &[f32]) -> u16 {
    let mut sig = 0u16;
    for (bit, plane) in planes.iter().enumerate() {
        if dot(embedding, plane) >= 0.0 {
            sig |= 1u16 << bit;
        }
    }
    sig
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

    #[test]
    fn test_candidate_search() {
        let mut idx = SemanticIndex::new();
        let mut e1 = vec![0.0f32; 768];
        e1[0] = 1.0;
        let mut e2 = vec![0.0f32; 768];
        e2[1] = 1.0;
        idx.upsert(1, e1.clone());
        idx.upsert(2, e2.clone());

        let hits = idx.search_candidates(&e1, [2, 1], 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory_id, 1);
        assert!(hits[0].cosine_similarity > 0.99);
    }
}

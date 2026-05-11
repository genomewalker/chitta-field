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

// HNSW kicks in above this many memories — below it IVF+LSH is fast enough.
const HNSW_THRESHOLD: usize = 2000;
// HNSW build/search parameters
const HNSW_M: usize = 16;        // neighbors per non-zero layer
const HNSW_M0: usize = 32;       // neighbors at layer 0 (2×M)
const HNSW_EF_CONSTRUCTION: usize = 200; // candidates during insert
const HNSW_EF_SEARCH: usize = 64;        // candidates during search
// 1/ln(M) — controls layer probability distribution
const HNSW_ML: f64 = 0.36067; // 1/ln(16)

// ── HNSW graph ────────────────────────────────────────────────────────────────

/// A layered navigable small-world graph for approximate nearest-neighbour
/// search. Purely in-RAM; rebuilt from `SemanticIndex::embeddings` after load.
///
/// All vectors stored in the parent `SemanticIndex` are unit-normalised, so
/// inner product == cosine similarity throughout.
#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct HnswGraph {
    /// Per-node adjacency lists, indexed by layer.
    /// `neighbors[id][layer]` = Vec of neighbour MemoryIds.
    neighbors: HashMap<MemoryId, Vec<Vec<MemoryId>>>,
    /// Entry point: (node_id, max_layer_of_that_node).
    entry_point: Option<(MemoryId, usize)>,
}

impl HnswGraph {
    fn new() -> Self {
        Self::default()
    }

    fn is_empty(&self) -> bool {
        self.neighbors.is_empty()
    }

    pub(crate) fn len(&self) -> usize {
        self.neighbors.len()
    }

    /// Insert `id` with pre-normalised `embedding`.
    /// `embeddings` is the full store (used to look up neighbour vectors).
    fn insert(&mut self, id: MemoryId, embedding: &[f32], embeddings: &HashMap<MemoryId, Vec<f32>>) {
        let node_level = self.random_level();

        // Determine current max layer
        let current_top = self.entry_point.map(|(_, l)| l).unwrap_or(0);

        // Initialise adjacency list for new node
        self.neighbors.insert(id, vec![Vec::new(); node_level + 1]);

        let ep = match self.entry_point {
            None => {
                // First node
                self.entry_point = Some((id, node_level));
                return;
            }
            Some((ep, _)) => ep,
        };

        // Phase 1: descend from top of graph down to node_level+1
        let mut ep_set = vec![ep];
        for layer in (node_level + 1..=current_top).rev() {
            ep_set = self.search_layer(embedding, &ep_set, 1, layer, embeddings);
        }

        // Phase 2: insert at each layer from min(node_level, current_top) down to 0
        for layer in (0..=node_level.min(current_top)).rev() {
            let candidates = self.search_layer(embedding, &ep_set, HNSW_EF_CONSTRUCTION, layer, embeddings);
            let m = if layer == 0 { HNSW_M0 } else { HNSW_M };
            let selected = self.select_neighbors(embedding, &candidates, m, embeddings);

            // Connect new node to its selected neighbours
            if let Some(node_neighbors) = self.neighbors.get_mut(&id) {
                if layer < node_neighbors.len() {
                    node_neighbors[layer] = selected.clone();
                }
            }

            // Connect each selected neighbour back to new node (shrink if needed)
            for &neighbor_id in &selected {
                let m_max = if layer == 0 { HNSW_M0 } else { HNSW_M };
                // Push new node into neighbour's list, then check if shrink needed
                let needs_shrink = {
                    if let Some(nbr_layers) = self.neighbors.get_mut(&neighbor_id) {
                        if layer < nbr_layers.len() {
                            nbr_layers[layer].push(id);
                            nbr_layers[layer].len() > m_max
                        } else { false }
                    } else { false }
                };
                if needs_shrink {
                    // Compute shrunk list without holding mutable borrow
                    let shrunk = if let (Some(nbr_layers), Some(nbr_emb)) =
                        (self.neighbors.get(&neighbor_id), embeddings.get(&neighbor_id))
                    {
                        if layer < nbr_layers.len() {
                            Some(self.select_neighbors(nbr_emb, &nbr_layers[layer].clone(), m_max, embeddings))
                        } else { None }
                    } else { None };
                    if let (Some(shrunk), Some(nbr_layers)) = (shrunk, self.neighbors.get_mut(&neighbor_id)) {
                        if layer < nbr_layers.len() {
                            nbr_layers[layer] = shrunk;
                        }
                    }
                }
            }

            ep_set = candidates;
        }

        // Update entry point if new node has higher layer
        if node_level > current_top {
            self.entry_point = Some((id, node_level));
        }
    }

    /// Remove a node by clearing its adjacency lists and removing it from
    /// neighbours' lists. This is O(M × L) and acceptable for soft-delete
    /// rates typical in a memory store.
    fn remove(&mut self, id: MemoryId) {
        let Some(node_neighbors) = self.neighbors.remove(&id) else { return };
        // Unlink from neighbours' adjacency lists
        for layer_neighbors in &node_neighbors {
            for &neighbor_id in layer_neighbors {
                if let Some(nbr_layers) = self.neighbors.get_mut(&neighbor_id) {
                    for layer_list in nbr_layers.iter_mut() {
                        layer_list.retain(|&n| n != id);
                    }
                }
            }
        }
        // If this was the entry point, pick any remaining node at the highest layer
        if self.entry_point.map(|(ep, _)| ep) == Some(id) {
            self.entry_point = self
                .neighbors
                .iter()
                .map(|(&nid, layers)| (nid, layers.len().saturating_sub(1)))
                .max_by_key(|&(_, l)| l);
        }
    }

    /// Greedy k-NN search. Returns up to k MemoryIds, filtered by `allowed`.
    fn search(
        &self,
        query: &[f32],
        k: usize,
        allowed: Option<&HashSet<MemoryId>>,
        deleted: &HashSet<MemoryId>,
        embeddings: &HashMap<MemoryId, Vec<f32>>,
    ) -> Vec<(MemoryId, f32)> {
        let Some((ep, top_layer)) = self.entry_point else { return vec![] };

        let mut ep_set = vec![ep];

        // Descend from top layer to layer 1 with ef=1
        for layer in (1..=top_layer).rev() {
            ep_set = self.search_layer(query, &ep_set, 1, layer, embeddings);
        }

        // Final search at layer 0 with ef_search
        let candidates = self.search_layer(query, &ep_set, HNSW_EF_SEARCH.max(k), 0, embeddings);

        // Filter deleted / not-allowed, compute similarities, take top k
        let mut scored: Vec<(MemoryId, f32)> = candidates
            .into_iter()
            .filter(|id| !deleted.contains(id))
            .filter(|id| allowed.map(|a| a.contains(id)).unwrap_or(true))
            .filter_map(|id| embeddings.get(&id).map(|e| (id, dot(query, e))))
            .collect();

        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(k);
        scored
    }

    /// Greedy beam search within one layer. Returns ef best candidates.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[MemoryId],
        ef: usize,
        layer: usize,
        embeddings: &HashMap<MemoryId, Vec<f32>>,
    ) -> Vec<MemoryId> {
        let mut visited: HashSet<MemoryId> = entry_points.iter().copied().collect();
        // candidates: min-heap by score (worst-best at top, so we can prune)
        let mut candidates: BinaryHeap<(OrderedF32, MemoryId)> = entry_points
            .iter()
            .filter_map(|&id| {
                embeddings.get(&id).map(|e| (OrderedF32(-dot(query, e)), id))
            })
            .collect();
        // result: max-heap by score (best at top)
        let mut result: BinaryHeap<(OrderedF32, MemoryId)> = entry_points
            .iter()
            .filter_map(|&id| {
                embeddings.get(&id).map(|e| (OrderedF32(dot(query, e)), id))
            })
            .collect();

        while let Some((neg_score, cid)) = candidates.pop() {
            let c_score = -neg_score.0;
            // Early stop: if worst result is better than best candidate, done
            if result.len() >= ef {
                if let Some(&(OrderedF32(best_result_score), _)) = result.peek() {
                    if c_score < best_result_score {
                        break;
                    }
                }
            }
            // Expand neighbours at this layer
            let Some(node_layers) = self.neighbors.get(&cid) else { continue };
            let Some(layer_neighbors) = node_layers.get(layer) else { continue };
            for &neighbor_id in layer_neighbors {
                if visited.insert(neighbor_id) {
                    let Some(emb) = embeddings.get(&neighbor_id) else { continue };
                    let sim = dot(query, emb);
                    candidates.push((OrderedF32(-sim), neighbor_id));
                    result.push((OrderedF32(sim), neighbor_id));
                    // Trim result to ef
                    while result.len() > ef {
                        result.pop(); // pops the *highest* score because max-heap
                    }
                }
            }
        }

        result.into_iter().map(|(_, id)| id).collect()
    }

    /// Simple greedy neighbour selection (no heuristic pruning for now).
    fn select_neighbors(
        &self,
        query: &[f32],
        candidates: &[MemoryId],
        m: usize,
        embeddings: &HashMap<MemoryId, Vec<f32>>,
    ) -> Vec<MemoryId> {
        let mut scored: Vec<(f32, MemoryId)> = candidates
            .iter()
            .filter_map(|&id| embeddings.get(&id).map(|e| (dot(query, e), id)))
            .collect();
        scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(m);
        scored.into_iter().map(|(_, id)| id).collect()
    }

    /// Draw a random layer for a new node using the standard HNSW distribution.
    /// Uses a simple LCG so there's no rand dependency.
    fn random_level(&self) -> usize {
        // Seed from number of nodes for deterministic but varied levels
        let n = self.neighbors.len() as u64;
        let mut s = n
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s ^= s >> 33;
        s = s.wrapping_mul(0xff51afd7ed558ccd);
        s ^= s >> 33;
        let f = (s >> 11) as f64 / (1u64 << 53) as f64; // uniform [0,1)
        if f < 1e-15 {
            return 0;
        }
        let level = (-f.ln() * HNSW_ML).floor() as usize;
        level.min(16) // cap at 16 layers
    }
}

/// Newtype for f32 that implements Ord via total_cmp. Used in BinaryHeap.
#[derive(Clone, Copy, PartialEq)]
struct OrderedF32(f32);
impl Eq for OrderedF32 {}
impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.0.total_cmp(&other.0) }
}

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
    /// HNSW graph — saved to sidecar .hnsw file; rebuilt when stale.
    /// Active when `embeddings.len() >= HNSW_THRESHOLD`.
    #[serde(skip)]
    hnsw: HnswGraph,
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
            hnsw: HnswGraph::new(),
        }
    }

    /// True when the HNSW graph is the active search path.
    #[inline]
    fn use_hnsw(&self) -> bool {
        self.embeddings.len() >= HNSW_THRESHOLD
    }

    /// Read-only access to an embedding by memory ID.
    pub fn get_embedding(&self, id: MemoryId) -> Option<&[f32]> {
        self.embeddings.get(&id).map(|v| v.as_slice())
    }

    /// Iterator over all memory IDs that have an embedding (including soft-deleted).
    pub fn all_ids(&self) -> impl Iterator<Item = MemoryId> + '_ {
        self.embeddings.keys().copied()
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

        // Insert into HNSW if it is already active, or activate it now that
        // we've just crossed the threshold.
        if self.use_hnsw() {
            let emb = self.embeddings[&memory_id].clone();
            self.hnsw.insert(memory_id, &emb, &self.embeddings);
        }
    }

    /// Mark a memory as deleted — excluded from future search results.
    pub fn remove(&mut self, memory_id: MemoryId) {
        self.hnsw.remove(memory_id);
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

        // Use HNSW when the graph is active (>= HNSW_THRESHOLD memories)
        if self.use_hnsw() && !self.hnsw.is_empty() {
            let hits = self.hnsw.search(&query_unit, k, allowed, &self.deleted, &self.embeddings);
            return hits
                .into_iter()
                .map(|(memory_id, cosine_similarity)| SemanticHit { memory_id, cosine_similarity })
                .collect();
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

    /// Like `search`, but applies a retrieval-signature alignment boost.
    ///
    /// For each candidate, after computing cosine similarity, adds:
    ///   `beta * max(0.0, dot(query_ctx_32, signature_32))`
    /// where `query_ctx_32` is a 32-dim projected query context sketch and
    /// `signature_32` is the cached mean retrieval signature for that memory
    /// (from `RetrievalHistory::signature`).
    ///
    /// Memories with empty signatures receive zero boost.
    pub fn search_with_signature_boost(
        &self,
        query: &[f32],
        k: usize,
        allowed: Option<&HashSet<MemoryId>>,
        query_ctx_32: &[f32],
        signatures: &HashMap<MemoryId, Vec<f32>>,
        beta: f32,
    ) -> Vec<SemanticHit> {
        if query.len() != EMBED_DIM {
            return vec![];
        }
        let Some(query_unit) = normalize(query) else {
            return vec![];
        };

        let candidates = if let Some(allowed_ids) = allowed {
            if allowed_ids.len() <= MIN_CANDIDATES {
                allowed_ids.iter().copied().collect::<Vec<_>>()
            } else {
                let mut cands = self.collect_lsh_candidates(&query_unit, allowed, k);
                if cands.is_empty() {
                    let probes = self.choose_probe_ids(&query_unit, MIN_PROBES);
                    cands = self.collect_candidates(&probes, allowed, k);
                }
                cands
            }
        } else {
            let mut cands = self.collect_lsh_candidates(&query_unit, allowed, k);
            if cands.is_empty() {
                let probes = self.choose_probe_ids(&query_unit, MIN_PROBES);
                cands = self.collect_candidates(&probes, allowed, k);
            }
            cands
        };

        if candidates.is_empty() {
            return vec![];
        }

        let mut top_k = BinaryHeap::new();
        for memory_id in candidates {
            if self.deleted.contains(&memory_id) {
                continue;
            }
            let Some(embedding) = self.embeddings.get(&memory_id) else {
                continue;
            };
            let base_sim = dot(&query_unit, embedding);
            let boost = if let Some(sig) = signatures.get(&memory_id) {
                if !sig.is_empty() && !query_ctx_32.is_empty() {
                    beta * dot_n(query_ctx_32, sig, 32).max(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let adjusted_sim = base_sim + boost;
            push_top_k(&mut top_k, k, memory_id, adjusted_sim);
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


    /// Serialize the HNSW graph to a sidecar file.
    pub fn save_hnsw(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let bytes = bincode::serialize(&self.hnsw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let tmp = path.with_extension("hnsw.tmp");
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.sync_all()?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load HNSW from sidecar file. Returns false if file is missing/stale.
    /// "Stale" means the graph node count doesn't match the embeddings count.
    pub fn load_hnsw(&mut self, path: &std::path::Path) -> bool {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[hnsw] load_hnsw: read {:?} failed: {}", path, e);
                return false;
            }
        };
        let graph: HnswGraph = match bincode::deserialize(&bytes) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("[hnsw] load_hnsw: deserialize failed: {}", e);
                return false;
            }
        };
        if graph.len() != self.embeddings.len() {
            eprintln!("[hnsw] load_hnsw: stale graph ({} nodes) vs embeddings ({} entries) — rebuilding",
                graph.len(), self.embeddings.len());
            return false;
        }
        eprintln!("[hnsw] load_hnsw: loaded {} nodes from {:?}", graph.len(), path);
        self.hnsw = graph;
        true
    }

    fn rebuild_ann(&mut self) {
        let hnsw_valid = !self.hnsw.is_empty() && self.hnsw.len() == self.embeddings.len();
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

        // Rebuild HNSW if collection is large enough
        if !hnsw_valid && self.use_hnsw() {
            self.rebuild_hnsw();
        }
    }

    fn rebuild_hnsw(&mut self) {
        self.hnsw = HnswGraph::new();
        // Insert in insertion order (by id for determinism) to build graph
        let mut ids: Vec<MemoryId> = self.embeddings.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            let emb = self.embeddings[&id].clone();
            self.hnsw.insert(id, &emb, &self.embeddings);
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

fn dot_n(a: &[f32], b: &[f32], n: usize) -> f32 {
    a.iter().zip(b.iter()).take(n).map(|(x, y)| x * y).sum()
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

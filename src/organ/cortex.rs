use std::collections::{HashMap, HashSet};
use crate::ids::MemoryId;

// ── Sparse Code ──────────────────────────────────────────────────────────────

/// Sparse representation of a memory: K=64 active features out of N=16,384.
#[derive(Debug, Clone, Default)]
pub struct SparseCode {
    pub feature_ids: Vec<u32>,   // exactly K entries, sorted ascending
    pub activations: Vec<f32>,   // parallel to feature_ids, normalized
}

impl SparseCode {
    pub fn is_empty(&self) -> bool { self.feature_ids.is_empty() }
}

// ── Sparse Encoder (product-key) ─────────────────────────────────────────────

pub const EMBED_DIM: usize = 768;
pub const HALF_DIM: usize = 384;          // EMBED_DIM / 2
pub const N_LEFT: usize = 128;             // centroids for left half
pub const N_RIGHT: usize = 128;           // centroids for right half
pub const N_ATOMS: usize = N_LEFT * N_RIGHT; // 16,384 total
pub const K_ACTIVE: usize = 64;           // active features per memory
pub const ENCODER_LR: f32 = 5e-4;
pub const SHORTLIST_PER_HALF: usize = 16; // top-16 from each half → 256 candidates

/// Online product-key sparse encoder.
/// Dictionary: N_LEFT × N_RIGHT = 16,384 atoms, each EMBED_DIM-dimensional.
/// Represented as two half-dictionaries for O(√N) top-K selection.
pub struct SparseEncoder {
    left_atoms: Vec<Vec<f32>>,    // N_LEFT × HALF_DIM
    right_atoms: Vec<Vec<f32>>,   // N_RIGHT × HALF_DIM
}

impl Default for SparseEncoder {
    fn default() -> Self { Self::new() }
}

impl SparseEncoder {
    /// Initialize with random unit vectors from LCG seeded by index.
    pub fn new() -> Self {
        let mut left_atoms = Vec::with_capacity(N_LEFT);
        let mut right_atoms = Vec::with_capacity(N_RIGHT);
        for i in 0..N_LEFT {
            left_atoms.push(random_unit_vec(HALF_DIM, i as u64));
        }
        for i in 0..N_RIGHT {
            right_atoms.push(random_unit_vec(HALF_DIM, (N_LEFT + i) as u64));
        }
        Self { left_atoms, right_atoms }
    }

    /// Encode a 768-dim embedding into a sparse code of K=64 active features.
    /// Product-key top-K: O(128×384 + 128×384 + 256) instead of O(16384×768).
    pub fn encode(&self, embedding: &[f32]) -> SparseCode {
        assert_eq!(embedding.len(), EMBED_DIM);
        let (e_left, e_right) = embedding.split_at(HALF_DIM);

        // Score each half-dictionary
        let mut left_scores: Vec<(f32, usize)> = self.left_atoms.iter().enumerate()
            .map(|(i, a)| (dot(e_left, a), i))
            .collect();
        let mut right_scores: Vec<(f32, usize)> = self.right_atoms.iter().enumerate()
            .map(|(i, a)| (dot(e_right, a), i))
            .collect();

        // Partial sort: top SHORTLIST_PER_HALF from each half
        left_scores.select_nth_unstable_by(SHORTLIST_PER_HALF, |a, b| b.0.partial_cmp(&a.0).unwrap());
        right_scores.select_nth_unstable_by(SHORTLIST_PER_HALF, |a, b| b.0.partial_cmp(&a.0).unwrap());

        // Build shortlist of SHORTLIST_PER_HALF² = 256 candidates
        let mut candidates: Vec<(f32, u32)> = Vec::with_capacity(SHORTLIST_PER_HALF * SHORTLIST_PER_HALF);
        for &(ls, li) in &left_scores[..SHORTLIST_PER_HALF] {
            for &(rs, ri) in &right_scores[..SHORTLIST_PER_HALF] {
                let atom_id = (li * N_RIGHT + ri) as u32;
                candidates.push((ls + rs, atom_id));
            }
        }

        // Top-K from candidates
        let k = K_ACTIVE.min(candidates.len());
        candidates.select_nth_unstable_by(k, |a, b| b.0.partial_cmp(&a.0).unwrap());
        candidates.truncate(k);
        candidates.sort_by(|a, b| a.1.cmp(&b.1)); // sort by feature_id ascending

        // Normalize activations (relu + L1 norm)
        let sum: f32 = candidates.iter().map(|(s, _)| s.max(0.0)).sum();
        let norm = if sum > 1e-9 { sum } else { 1.0 };

        let feature_ids: Vec<u32> = candidates.iter().map(|(_, id)| *id).collect();
        let activations: Vec<f32> = candidates.iter().map(|(s, _)| s.max(0.0) / norm).collect();

        SparseCode { feature_ids, activations }
    }

    /// Online Hebbian update: adjust atoms toward the encoded memory.
    /// Called after put_memory to adapt the dictionary.
    pub fn update(&mut self, embedding: &[f32], code: &SparseCode) {
        if code.is_empty() { return; }
        let (e_left, e_right) = embedding.split_at(HALF_DIM);

        for (&fid, &act) in code.feature_ids.iter().zip(code.activations.iter()) {
            let li = (fid as usize) / N_RIGHT;
            let ri = (fid as usize) % N_RIGHT;

            // Left atom update
            let la = &mut self.left_atoms[li];
            for (w, &x) in la.iter_mut().zip(e_left) {
                *w += ENCODER_LR * act * (x - act * *w);
            }
            normalize(la);

            // Right atom update
            let ra = &mut self.right_atoms[ri];
            for (w, &x) in ra.iter_mut().zip(e_right) {
                *w += ENCODER_LR * act * (x - act * *w);
            }
            normalize(ra);
        }
    }
}

// ── Cortical Posting Index ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PostingEntry {
    pub mem_id: MemoryId,
    pub activation_q: u8,    // activation quantized to u8 (0.0-1.0 → 0-255)
    pub strength_q: u8,      // strength quantized to u8
}

/// Inverted posting index over sparse codes. O(K) posting lookups per query.
pub struct CorticalIndex {
    /// feature_id → posting list (sorted by mem_id for merge efficiency)
    postings: HashMap<u32, Vec<PostingEntry>>,
    /// document frequency: how many memories have each feature active
    df: HashMap<u32, u64>,
    /// total indexed memories
    pub n_memories: u64,
    /// mem_id → its sparse code (for removal)
    pub(crate) mem_codes: HashMap<MemoryId, SparseCode>,
    /// mem_id → ts_ms (for recency scoring)
    mem_ts: HashMap<MemoryId, i64>,
    /// mem_id → kind (for recency weighting: only "episode" kind gets decay)
    mem_kind: HashMap<MemoryId, String>,
}

impl Default for CorticalIndex {
    fn default() -> Self { Self::new() }
}

impl CorticalIndex {
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            df: HashMap::new(),
            n_memories: 0,
            mem_codes: HashMap::new(),
            mem_ts: HashMap::new(),
            mem_kind: HashMap::new(),
        }
    }

    pub fn index(&mut self, mem_id: MemoryId, code: &SparseCode, strength: f32, ts_ms: i64, kind: &str) {
        if code.is_empty() { return; }
        // Remove old code if re-indexing
        if self.mem_codes.contains_key(&mem_id) {
            self.remove(mem_id);
        }

        let strength_q = (strength.clamp(0.0, 1.0) * 255.0) as u8;

        for (&fid, &act) in code.feature_ids.iter().zip(code.activations.iter()) {
            let entry = PostingEntry {
                mem_id,
                activation_q: (act * 255.0) as u8,
                strength_q,
            };
            self.postings.entry(fid).or_default().push(entry);
            *self.df.entry(fid).or_insert(0) += 1;
        }

        self.mem_codes.insert(mem_id, code.clone());
        self.mem_ts.insert(mem_id, ts_ms);
        self.mem_kind.insert(mem_id, kind.to_string());
        self.n_memories += 1;
    }

    pub fn remove(&mut self, mem_id: MemoryId) {
        if let Some(code) = self.mem_codes.remove(&mem_id) {
            for fid in &code.feature_ids {
                if let Some(list) = self.postings.get_mut(fid) {
                    list.retain(|e| e.mem_id != mem_id);
                    if let Some(df) = self.df.get_mut(fid) {
                        *df = df.saturating_sub(1);
                    }
                }
            }
            self.n_memories = self.n_memories.saturating_sub(1);
        }
        self.mem_ts.remove(&mem_id);
        self.mem_kind.remove(&mem_id);
    }

    /// Update strength for a memory (called on reconsolidation).
    pub fn update_strength(&mut self, mem_id: MemoryId, strength: f32) {
        if let Some(code) = self.mem_codes.get(&mem_id) {
            let sq = (strength.clamp(0.0, 1.0) * 255.0) as u8;
            for fid in &code.feature_ids.clone() {
                if let Some(list) = self.postings.get_mut(fid) {
                    for e in list.iter_mut() {
                        if e.mem_id == mem_id { e.strength_q = sq; }
                    }
                }
            }
        }
    }

    /// Search: IDF-weighted sparse overlap scoring.
    /// Returns up to k results as (mem_id, score), sorted by score descending.
    pub fn search(
        &self,
        query_code: &SparseCode,
        k: usize,
        allowed: Option<&HashSet<MemoryId>>,
    ) -> Vec<(MemoryId, f32)> {
        if query_code.is_empty() || self.n_memories == 0 { return Vec::new(); }

        let n = self.n_memories as f32;
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        // Accumulate scores per candidate memory
        let mut scores: HashMap<MemoryId, f32> = HashMap::new();
        let mut query_norm = 0.0f32;

        for (&fid, &q_act) in query_code.feature_ids.iter().zip(query_code.activations.iter()) {
            let df = self.df.get(&fid).copied().unwrap_or(1) as f32;
            let idf = ((n + 1.0) / (df + 1.0)).ln() + 1.0;
            query_norm += idf * q_act;

            let Some(list) = self.postings.get(&fid) else { continue };

            for entry in list {
                if let Some(allowed_set) = allowed {
                    if !allowed_set.contains(&entry.mem_id) { continue; }
                }
                let c_act = entry.activation_q as f32 / 255.0;
                let term = idf * (q_act * c_act).sqrt();
                *scores.entry(entry.mem_id).or_insert(0.0) += term;
            }
        }

        if query_norm < 1e-9 { query_norm = 1.0; }

        // Finalize scores with strength + recency
        let mut results: Vec<(MemoryId, f32)> = scores.into_iter().map(|(mem_id, sparse_raw)| {
            let sparse = sparse_raw / query_norm;

            // Get strength from a posting entry
            let strength = self.mem_codes.get(&mem_id)
                .and_then(|code| code.feature_ids.first())
                .and_then(|&fid| self.postings.get(&fid))
                .and_then(|list| list.iter().find(|e| e.mem_id == mem_id))
                .map(|e| e.strength_q as f32 / 255.0)
                .unwrap_or(0.5);

            // Recency: only for episodic memories
            let is_episodic = self.mem_kind.get(&mem_id)
                .map(|k| k == "episode" || k == "observation")
                .unwrap_or(false);
            let recency = if is_episodic {
                let age_days = (now_ms - self.mem_ts.get(&mem_id).copied().unwrap_or(0))
                    .max(0) as f32 / (86400.0 * 1000.0);
                (-age_days / 30.0).exp()
            } else { 0.0 };

            let score = 0.85 * sparse + 0.10 * strength + 0.05 * recency;
            (mem_id, score)
        }).collect();

        // Top-K
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    pub fn len(&self) -> usize { self.n_memories as usize }
    pub fn is_empty(&self) -> bool { self.n_memories == 0 }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn normalize(v: &mut Vec<f32>) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 { for x in v.iter_mut() { *x /= norm; } }
}

/// Generate a random unit vector using a simple LCG seeded by index.
fn random_unit_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut v: Vec<f32> = (0..dim).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Box-Muller for normal distribution
        let u1 = (state >> 11) as f32 / (1u64 << 53) as f32;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (state >> 11) as f32 / (1u64 << 53) as f32;
        let r = (-2.0 * (u1 + 1e-10).ln()).sqrt();
        r * (2.0 * std::f32::consts::PI * u2).cos()
    }).collect();
    normalize(&mut v);
    v
}

use crate::ids::{ChunkHash, MemoryId};
use crate::ops::StateDeltaOp;
use serde::{Deserialize, Serialize};

/// Compact quantized sketch of a query that retrieved this memory.
/// 32 i8 values + scale reconstruct a ~32-dim projected embedding.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RetrievalContext {
    pub centroid_q: Vec<i8>,   // 32 dims, quantized
    pub scale: f32,
    pub context_hash: u64,     // hash of original query embedding + realm
    pub ts_ms: i64,
}

/// Rolling history of up to 8 query contexts that retrieved a memory.
/// `signature` is the cached mean of the stored centroids (full f32, 32 dims).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RetrievalHistory {
    pub entries: Vec<RetrievalContext>,  // capped at 8, FIFO oldest-drop
    pub signature: Vec<f32>,             // cached mean centroid (32 dims)
}

impl RetrievalHistory {
    pub const MAX_ENTRIES: usize = 8;
    pub const CENTROID_DIMS: usize = 32;

    /// Append a new retrieval context; if full, drop the oldest.
    /// Recomputes the cached mean signature.
    pub fn push(&mut self, ctx: RetrievalContext) {
        if self.entries.len() >= Self::MAX_ENTRIES {
            self.entries.remove(0);
        }
        self.entries.push(ctx);
        self.recompute_signature();
    }

    fn recompute_signature(&mut self) {
        if self.entries.is_empty() {
            self.signature.clear();
            return;
        }
        let n = self.entries.len() as f32;
        let dims = Self::CENTROID_DIMS;
        let mut sig = vec![0f32; dims];
        for entry in &self.entries {
            for (i, &q) in entry.centroid_q.iter().take(dims).enumerate() {
                sig[i] += (q as f32 * entry.scale) / n;
            }
        }
        self.signature = sig;
    }
}

/// Mutable overlay state for a memory — lives separately from the immutable payload.
/// First-class memory lifecycle status.
/// `Active` is the default; `Superseded` is set when a correction replaces this memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum MemoryStatus {
    #[default]
    Active,
    Superseded,
    Contradicted,
    Archived,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryState {
    pub memory_id: MemoryId,
    pub current_version: u32,
    pub current_chunk_hash: ChunkHash,
    pub deleted: bool,

    pub strength: f32,
    pub decay_rate: f32,
    pub confidence: f32,

    pub access_count: u32,
    pub last_accessed_ms: i64,
    pub last_strengthened_ms: i64,
    pub created_at_ms: i64,

    pub pinned: bool,
    pub tier: u8, // 0=L1 (hippocampus), 1=L2 (cortex), 2=L3 (archive)
    #[serde(default)]
    pub last_state_op_ts_ms: i64,
    #[serde(default)]
    pub retrieval_history: RetrievalHistory,
    #[serde(default)]
    pub embed_pending: bool,
    #[serde(default)]
    pub status: MemoryStatus,
}

impl MemoryState {
    pub fn new(memory_id: MemoryId, chunk_hash: ChunkHash, created_at_ms: i64) -> Self {
        Self {
            memory_id,
            current_version: 0,
            current_chunk_hash: chunk_hash,
            deleted: false,
            strength: 1.0,
            decay_rate: 0.001,
            confidence: 1.0,
            access_count: 0,
            last_accessed_ms: created_at_ms,
            last_strengthened_ms: created_at_ms,
            created_at_ms,
            pinned: false,
            tier: 0,
            last_state_op_ts_ms: 0,
            retrieval_history: RetrievalHistory::default(),
            embed_pending: false,
            status: MemoryStatus::Active,
        }
    }

    pub fn apply_delta(&mut self, delta: &StateDeltaOp, now_ms: i64) {
        if delta.op_ts_ms > 0 && delta.op_ts_ms <= self.last_state_op_ts_ms {
            return;
        }
        if delta.op_ts_ms > 0 {
            self.last_state_op_ts_ms = delta.op_ts_ms;
        }
        if let Some(d) = delta.strength_delta {
            self.strength = (self.strength + d).clamp(0.0, 1.0);
            if d > 0.0 {
                self.last_strengthened_ms = now_ms;
            }
        }
        if let Some(d) = delta.confidence_delta {
            self.confidence = (self.confidence + d).clamp(0.0, 1.0);
        }
        if let Some(r) = delta.decay_rate {
            self.decay_rate = r.max(0.0);
        }
        if delta.touch {
            self.access_count += 1;
            self.last_accessed_ms = now_ms;
        }
        if let Some(p) = delta.pin {
            self.pinned = p;
        }
        if let Some(s) = delta.status {
            self.status = match s {
                1 => MemoryStatus::Superseded,
                2 => MemoryStatus::Contradicted,
                3 => MemoryStatus::Archived,
                _ => MemoryStatus::Active,
            };
        }
    }

    /// Compute effective strength accounting for time-based decay.
    pub fn effective_strength(&self, now_ms: i64) -> f32 {
        let age_days = (now_ms - self.last_strengthened_ms).max(0) as f64 / 86_400_000.0;
        let decayed = self.strength as f64 * (-self.decay_rate as f64 * age_days).exp();
        decayed.clamp(0.0, 1.0) as f32
    }
}

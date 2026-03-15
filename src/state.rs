use crate::ids::{MemoryId, ChunkHash};
use crate::ops::StateDeltaOp;

/// Mutable overlay state for a memory — lives separately from the immutable payload.
#[derive(Debug, Clone)]
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
        }
    }

    pub fn apply_delta(&mut self, delta: &StateDeltaOp, now_ms: i64) {
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
    }

    /// Compute effective strength accounting for time-based decay.
    pub fn effective_strength(&self, now_ms: i64) -> f32 {
        let age_days = (now_ms - self.last_strengthened_ms).max(0) as f64 / 86_400_000.0;
        let decayed = self.strength as f64 * (-self.decay_rate as f64 * age_days).exp();
        decayed.clamp(0.0, 1.0) as f32
    }
}

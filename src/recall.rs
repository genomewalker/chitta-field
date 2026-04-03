use crate::ids::MemoryId;

#[derive(Debug, Clone)]
pub struct RecallHit {
    pub memory_id: MemoryId,
    pub score: f32,
    pub semantic_score: f32,
    pub ts_ms: i64,
    pub kind: String,
    pub realm: String,
    pub strength: f32,
    pub confidence: f32,
    pub access_count: u32,
    /// UTF-8 decoded content (empty string if content is not valid UTF-8).
    pub content: String,
    /// Explain fields: decompose how the final score was computed.
    pub semantic_weight: f32,
    pub status_mul: f32,
    pub epistemic_mul: f32,
    pub strength_factor: f32,
    /// Affect dimensions (Anthropic emotion vectors 2026)
    pub affect_valence: f32,
    pub affect_arousal: f32,
}

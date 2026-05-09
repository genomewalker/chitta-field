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
    /// Cognitive scoring decomposition (ACT-R / FEP / flashbulb)
    pub actr_activation: f32,
    pub surprise_boost: f32,
    pub arousal_boost: f32,
    /// Mood-congruent recall (Bower 1981): boost when query affect matches memory affect
    pub mood_congruence: f32,
    /// Frustration-escalation: extra boost for corrections when caller is frustrated
    pub frustration_boost: f32,
    /// Interference density: penalty from local competitor crowding (Price of Meaning)
    pub interference_factor: f32,
    /// Spacing boost: reward for well-spaced retrieval intervals (Geometry of Forgetting)
    pub spacing_boost: f32,
}

/// Session-level recall hit: evidence from multiple chunks aggregated per source_session.
#[derive(Debug, Clone)]
pub struct SessionRecallHit {
    pub session_id: String,
    /// Noisy-OR combined evidence score
    pub score: f32,
    /// Number of chunks contributing evidence
    pub chunk_count: u32,
    /// Best single-chunk score (max semantic)
    pub max_chunk_score: f32,
    /// Content of the highest-scoring chunk
    pub best_evidence: String,
    pub realm: String,
}

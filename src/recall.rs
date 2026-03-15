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
}

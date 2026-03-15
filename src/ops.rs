use serde::{Deserialize, Serialize};
use crate::ids::{MemoryId, ChunkHash, ArtifactId};

pub const EMBED_DIM: usize = 768; // BGE-base-en-v1.5

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Op {
    PutPayload(PutPayloadOp),
    UpdateState(StateDeltaOp),
    DeleteMemory(DeleteMemoryOp),
    AddAssocEdge(AddAssocEdgeOp),
    UpsertArtifact(UpsertArtifactOp),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PutPayloadOp {
    pub memory_id: MemoryId,
    pub version: u32,
    pub chunk_hash: ChunkHash,
    pub created_at_ms: i64,
    pub authored_at_ms: i64,
    pub kind: String,
    pub realm: String,
    pub content: Vec<u8>,
    pub embedding_model: String,
    pub embedding: Vec<f32>, // len == EMBED_DIM
    pub artifact_refs: Vec<ArtifactRef>,
    pub source_session: Option<String>,
    pub source_tool: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDeltaOp {
    pub memory_id: MemoryId,
    pub strength_delta: Option<f32>,
    pub confidence_delta: Option<f32>,
    pub decay_rate: Option<f32>,
    pub touch: bool, // update last_accessed_ms
    pub pin: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteMemoryOp {
    pub memory_id: MemoryId,
    pub deleted_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeType {
    DerivedFrom,
    SameSession,
    SameArtifact,
    CoRetrieved,
    Contradicts,
    Supports,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddAssocEdgeOp {
    pub src: MemoryId,
    pub dst: MemoryId,
    pub edge_type: EdgeType,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertArtifactOp {
    pub artifact_id: ArtifactId,
    pub normalized_path: String,
    pub repo_root: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub artifact_id: ArtifactId,
    pub relation: ArtifactRelation,
    pub line_start: u32,
    pub line_end: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactRelation {
    Touched,
    Edited,
    Created,
    Read,
    Mentioned,
}

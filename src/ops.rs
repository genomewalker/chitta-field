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
    AddTriplet(AddTripletOp),
    InvalidateTriplet(InvalidateTripletOp),
    UpsertSymbol(UpsertSymbolOp),
    RemoveSymbol(RemoveSymbolOp),
    AddSymCallEdge(AddSymCallEdgeOp),
    RemoveSymCallEdge(RemoveSymCallEdgeOp),
    UpsertCodeFile(UpsertCodeFileOp),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddTripletOp {
    pub triplet_id: u64,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub weight: f32,
    pub valid_from_ms: i64,
    pub source_memory_id: Option<u64>,
    pub source_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvalidateTripletOp {
    pub triplet_id: u64,
    pub invalidated_at_ms: i64,
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
pub struct UpsertSymbolOp {
    pub symbol_id: u64,
    pub kind: String,
    pub name: String,
    pub signature: String,
    pub file_path: String,
    pub line_start: u32,
    pub line_end: u32,
    pub repo_id: u64,
    pub embedding: Vec<f32>,
    pub description: Option<String>,
    pub memory_id: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoveSymbolOp {
    pub symbol_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddSymCallEdgeOp {
    pub caller_id: u64,
    pub callee_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoveSymCallEdgeOp {
    pub caller_id: u64,
    pub callee_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertCodeFileOp {
    pub file_id: u64,
    pub path: String,
    pub project: String,
    pub mtime: i64,
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

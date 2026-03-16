pub use crate::ops::{PutPayloadOp, ArtifactRef, ArtifactRelation};
use crate::ids::{MemoryId, ChunkHash};
use serde::{Serialize, Deserialize};

/// In-memory representation of the latest payload for a memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPayload {
    pub memory_id: MemoryId,
    pub version: u32,
    pub chunk_hash: ChunkHash,
    pub created_at_ms: i64,
    pub authored_at_ms: i64,
    pub kind: String,
    pub realm: String,
    pub content: Vec<u8>,
    pub embedding_model: String,
    pub embedding: Vec<f32>,
    pub artifact_refs: Vec<ArtifactRef>,
    pub source_session: Option<String>,
    pub source_tool: Option<String>,
}

impl From<PutPayloadOp> for MemoryPayload {
    fn from(op: PutPayloadOp) -> Self {
        Self {
            memory_id: op.memory_id,
            version: op.version,
            chunk_hash: op.chunk_hash,
            created_at_ms: op.created_at_ms,
            authored_at_ms: op.authored_at_ms,
            kind: op.kind,
            realm: op.realm,
            content: op.content,
            embedding_model: op.embedding_model,
            embedding: op.embedding,
            artifact_refs: op.artifact_refs,
            source_session: op.source_session,
            source_tool: op.source_tool,
        }
    }
}

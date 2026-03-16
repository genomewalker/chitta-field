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
    UpdateSparseCode(UpdateSparseCodeOp),
    DemoteMemory(DemoteMemoryOp),
    TrainPQ(TrainPQOp),
    UpdateResidualPQ(UpdateResidualPQOp),
    SessionEvent(SessionEventOp),
    TranscriptEvent(TranscriptEventOp),
    TaskEvent(TaskEventOp),
    UserModelEvent(UserModelEventOp),
    ThemeEvent(ThemeEventOp),
    AnalyticsEvent(AnalyticsEventOp),
    ClearProject(ClearProjectOp),
    UpdateSymbolDescription(UpdateSymbolDescriptionOp),
    UpdateMemoryContent(UpdateMemoryContentOp),
}

pub const OP_SESSION_EVENT: u8 = 16;
pub const OP_TRANSCRIPT_EVENT: u8 = 17;
pub const OP_TASK_EVENT: u8 = 18;
pub const OP_USER_MODEL_EVENT: u8 = 19;
pub const OP_THEME_EVENT: u8 = 20;
pub const OP_ANALYTICS_EVENT: u8 = 21;
pub const OP_CLEAR_PROJECT: u8 = 22;
pub const OP_UPDATE_SYMBOL_DESCRIPTION: u8 = 23;
pub const OP_UPDATE_MEMORY_CONTENT: u8 = 24;

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
    #[serde(default)]
    pub op_ts_ms: i64,
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
pub struct UpdateSparseCodeOp {
    pub memory_id: u64,
    pub feature_ids: Vec<u32>,
    pub activations: Vec<f32>,
    pub ts_ms: i64,  // when encoded
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoteMemoryOp {
    pub memory_id: MemoryId,
    pub new_tier: u8,
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

/// Persists a trained ProductQuantizer (bincode-serialized codebooks).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainPQOp {
    pub codebook_bytes: Vec<u8>,
}

/// Persists PQ codes for the residual of a single memory.
/// pq_bytes stored as Vec<u8> (length 32) for serde compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateResidualPQOp {
    pub memory_id: MemoryId,
    pub pq_bytes: Vec<u8>,
}

/// Domain event envelope for session lifecycle events.
/// kind: "register", "heartbeat", "deregister", "message_send", "message_ack"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEventOp {
    pub event_id: u64,
    pub session_id: String,
    pub kind: String,
    pub payload_json: Vec<u8>,
    pub realm: String,
    pub ts_ms: i64,
}

/// Domain event envelope for transcript lifecycle events.
/// kind: "register", "update_progress", "add_turn", "create_episode"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptEventOp {
    pub event_id: u64,
    pub session_id: String,
    pub kind: String,
    pub payload_json: Vec<u8>,
    pub realm: String,
    pub ts_ms: i64,
}

/// Domain event envelope for task lifecycle events.
/// task_type: "sadhana", "long_task", "dream", "background"
/// kind: "create", "start", "pause", "resume", "complete", "fail"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEventOp {
    pub event_id: u64,
    pub task_type: String,
    pub task_id: String,
    pub kind: String,
    pub payload_json: Vec<u8>,
    pub realm: String,
    pub ts_ms: i64,
    pub fencing_token: u64,
}

/// Domain event envelope for user model entity events.
/// entity_type: "profile", "goal", "habit", "anticipation", "calibration"
/// kind: "upsert", "observe", "progress", "complete"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserModelEventOp {
    pub event_id: u64,
    pub entity_type: String,
    pub entity_id: String,
    pub kind: String,
    pub payload_json: Vec<u8>,
    pub ts_ms: i64,
}

/// Domain event envelope for theme graph events.
/// kind: "create", "update_centroid", "assign_member", "remove_member"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeEventOp {
    pub event_id: u64,
    pub kind: String,
    pub theme_id: u64,
    pub payload_json: Vec<u8>,
    pub ts_ms: i64,
}

/// Remove all code files and associated symbols for a project.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearProjectOp {
    pub project: String,
}

/// Update the description string for a symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateSymbolDescriptionOp {
    pub symbol_id: u64,
    pub description: String,
}

/// Update the content and/or embedding for an existing memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateMemoryContentOp {
    pub memory_id: MemoryId,
    pub content: Vec<u8>,
    pub embedding: Vec<f32>,  // empty = no embedding change
}

/// Domain event envelope for analytics events.
/// kind: "exposure", "recall_query", "correction", "usage_outcome"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsEventOp {
    pub event_id: u64,
    pub kind: String,
    pub session_id: String,
    pub payload_json: Vec<u8>,
    pub ts_ms: i64,
}

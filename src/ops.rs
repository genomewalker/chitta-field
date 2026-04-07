use crate::ids::{ArtifactId, ChunkHash, MemoryId};
use serde::{Deserialize, Serialize};

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
    RecordRecallBatch(RecordRecallBatchOp),
    StrengthenAssocEdge(StrengthenAssocEdgeOp),
    MsgEvent(MsgEventOp),
    SkillUpload(SkillUploadOp),
    SkillDeprecate(SkillDeprecateOp),
    AgentUpsert(AgentUpsertOp),
    AgentDisable(AgentDisableOp),
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
pub const OP_RECORD_RECALL_BATCH: u8 = 25;
pub const OP_STRENGTHEN_ASSOC_EDGE: u8 = 26;
pub const OP_MSG_EVENT: u8 = 27;
pub const OP_SKILL_UPLOAD: u8 = 28;
pub const OP_SKILL_DEPRECATE: u8 = 29;
pub const OP_AGENT_UPSERT: u8 = 30;
pub const OP_AGENT_DISABLE: u8 = 31;

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
    /// Status change (0=Active, 1=Superseded, 2=Contradicted, 3=Archived, 4=Proposed, 5=Observed, 6=Verified); None = no change
    #[serde(default)]
    pub status: Option<u8>,
    /// Epistemic status (0=UserStated, 1=ToolDerived, 2=ModelInferred, 3=AutonomousSynthesis); None = no change
    #[serde(default)]
    pub epistemic_status: Option<u8>,
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
    pub ts_ms: i64, // when encoded
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
    pub embedding: Vec<f32>, // empty = no embedding change
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

/// Durably records a completed recall event: updates access state for each
/// retrieved memory, appends retrieval context to their history, and updates
/// pairwise co-activation stats for all (src, dst) pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordRecallBatchOp {
    /// IDs of memories returned as the final result of this recall.
    pub memory_ids: Vec<MemoryId>,
    /// 32-dim quantized sketch of the (possibly refined) query centroid.
    pub centroid_q: Vec<i8>,
    pub centroid_scale: f32,
    /// Stable hash of the original user query embedding + realm.
    /// Used to measure context diversity across co-activations.
    pub context_hash: u64,
    pub ts_ms: i64,
    /// Base weight delta to apply to each co-retrieved assoc edge.
    /// Actual delta is scaled by pair stats (sim_count * diversity_count).
    pub base_assoc_delta: f32,
}

/// Upsert an assoc edge: if one already exists between (src, dst, edge_type),
/// add `delta` to its weight. Otherwise insert a new edge with weight = `delta`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrengthenAssocEdgeOp {
    pub src: MemoryId,
    pub dst: MemoryId,
    pub edge_type: EdgeType,
    pub delta: f32,
}

/// Domain event for cross-session messaging.
/// domain: always "msg"
/// kind: "send", "ack", "ack_all"
/// target: recipient session_id (or message_id for ack)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MsgEventOp {
    pub event_id: u64,
    pub domain: String,
    pub kind: String,
    pub target: String,
    pub payload_json: Vec<u8>,
    pub realm: String,
    pub ts_ms: i64,
}

/// Upload a new version of a skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillUploadOp {
    pub skill_id: String,
    pub content: String,
    pub uploaded_by: String,
    pub tags: Vec<String>,
    pub ts_ms: i64,
}

/// Deprecate a skill (marks latest version).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDeprecateOp {
    pub skill_id: String,
}

/// Register or update an agent identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentUpsertOp {
    pub agent_id: String,
    pub display_name: String,
    pub description: String,
    pub ts_ms: i64,
}

/// Disable (revoke) an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDisableOp {
    pub agent_id: String,
}

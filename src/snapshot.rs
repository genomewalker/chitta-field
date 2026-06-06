use crate::error::{FieldError, Result};
use crate::field::{AssocEdge, CoActivationStats};
use crate::hnsw::SemanticIndex;
use crate::ids::{ArtifactId, ChunkHash, MemoryId};
use crate::organ::artifact::ArtifactIndex;
use crate::organ::callgraph::CallGraph;
use crate::organ::codefile::CodeFileIndex;
use crate::organ::keyword::KeywordIndex;
use crate::organ::symbol::SymbolIndex;
use crate::organ::temporal::TemporalIndex;
use crate::organ::event_tape::EventTape;
use crate::organ::decision_tape::DecisionTape;
use crate::organ::turiya_monitor::TuriyaMonitor;
use crate::organ::observer::ObserverState;
use crate::organ::interaction_ledger::InteractionLedger;
use crate::organ::predicate_store::PredicateStore;
use crate::organ::triplet::{CorrectionState, TripletStore};
use crate::payload::MemoryPayload;
use crate::state::{EpistemicStatus, MemoryState, MemoryStatus, RetrievalHistory};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Magic for snapshots written before the ANN SemanticIndex rewrite (v1.0.3 and earlier).
const FULL_SNAPSHOT_MAGIC_V1: u64 = 0xF011_5741_7E00_0003;
/// Magic for v1.0.4 snapshots (post-ANN, old MemoryState, no coactivation_stats).
const FULL_SNAPSHOT_MAGIC_V4: u64 = 0xF011_5741_7E00_0004;
/// Magic for v1.0.5 snapshots (with coactivation_stats, but 14-field MemoryState).
/// These were written before RetrievalHistory/embed_pending/MemoryStatus/EpistemicStatus
/// were added to MemoryState on 2026-03-26.
const FULL_SNAPSHOT_MAGIC_V5: u64 = 0xF011_5741_7E00_0005;
/// Magic for v1.0.6 snapshots: full MemoryState with RetrievalHistory, embed_pending,
/// MemoryStatus, EpistemicStatus, surprise, affect — but NO access_timestamps.
const FULL_SNAPSHOT_MAGIC_V6: u64 = 0xF011_5741_7E00_0006;
/// Magic for v1.0.7 snapshots: adds access_timestamps but NO interference fields.
const FULL_SNAPSHOT_MAGIC_V7: u64 = 0xF011_5741_7E00_0007;
/// Magic for v1.0.8 snapshots: adds competitive_weight, lure_risk, spacing_quality.
/// FullSnapshot had no top-level sidecars (ack_scores / correction_states).
const FULL_SNAPSHOT_MAGIC_V8: u64 = 0xF011_5741_7E00_0008;
/// v1.0.9: FullSnapshot with ack_scores + correction_states; embeddings in bincode.
const FULL_SNAPSHOT_MAGIC_V9: u64 = 0xF011_5741_7E00_0009;
/// v1.0.10: embeddings moved to .emb sidecar; content still in bincode.
const FULL_SNAPSHOT_MAGIC_V10: u64 = 0xF011_5741_7E00_000A;
/// v1.0.11: content moved to .pld sidecar; bincode field is empty Vec.
/// MemoryState did NOT yet have `staged`/`invalidated_by`.
const FULL_SNAPSHOT_MAGIC_V11: u64 = 0xF011_5741_7E00_000B;
/// V12: MemoryState gains `staged` + `invalidated_by`. MemoryPayload unchanged.
const FULL_SNAPSHOT_MAGIC_V12: u64 = 0xF011_5741_7E00_000C;
/// V13: MemoryPayload gains `harness: Option<String>`.
const FULL_SNAPSHOT_MAGIC_V13: u64 = 0xF011_5741_7E00_000D;
/// V14: FullSnapshot gains `event_tape: EventTape` (CEC Phase 1).
const FULL_SNAPSHOT_MAGIC_V14: u64 = 0xF011_5741_7E00_000E;
/// V15: FullSnapshot gains `decision_tape: DecisionTape` (CEC Phase 10).
const FULL_SNAPSHOT_MAGIC_V15: u64 = 0xF011_5741_7E00_000F;
/// V16 magic (v5.35–5.41): FullSnapshot gains `turiya_monitor: TuriyaMonitor` (CEC Phase 11).
const FULL_SNAPSHOT_MAGIC_V16: u64 = 0xF011_5741_7E00_0010;
/// V17 magic: MemoryPayload gains `provenance: String` + `candidate: bool` (CEC Phase 17).
const FULL_SNAPSHOT_MAGIC_V17: u64 = 0xF011_5741_7E00_0011;
/// V18 magic: FullSnapshot gains `observer_state: ObserverState` (Tier-0 observation extraction).
pub const FULL_SNAPSHOT_MAGIC_V18: u64 = 0xF011_5741_7E00_0012;
/// V19 magic: MemoryPayload gains `embedding_model_id` + `embedding_dim`; EMBED_DIM 256→768.
const FULL_SNAPSHOT_MAGIC_V19: u64     = 0xF011_5741_7E00_0013;
/// V20 magic: FullSnapshot gains `interaction_ledger: InteractionLedger`.
const FULL_SNAPSHOT_MAGIC_V20: u64     = 0xF011_5741_7E00_0014;
/// V21 magic: FullSnapshot gains `predicate_store: PredicateStore`.
const FULL_SNAPSHOT_MAGIC: u64         = 0xF011_5741_7E00_0015; // V21

/// Magic for the payload content sidecar (.pld).
const PLD_MAGIC: u64 = 0x504C_4400_0000_0001; // "PLD\0\0\0\0\x01"

/// Magic for the store-identity sidecar (.shdr).
const SHDR_MAGIC: u64 = 0x5348_4452_0000_0001; // "SHDR\0\0\0\x01"
/// On-disk store format version (independent of the snapshot bincode magic).
pub const STORE_FORMAT_VERSION: u32 = 1;

/// Store identity recorded alongside each snapshot family in a `.shdr` sidecar.
///
/// Purpose: make snapshot selection and WAL replay aware of the *vector space* a store
/// was written in, so a foreign-dim/foreign-model snapshot can never win on max-seqno
/// (the root cause of the 768→1536 migration contamination). `vector_space_id` is a
/// stable hash of (model_id, embed_dim, text_format_version) — deterministic from the
/// compiled constants, so it can be computed without loading anything.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoreHeader {
    pub format_version: u32,
    pub embed_dim: u32,
    pub model_id: String,
    pub text_format_version: u32,
    /// Monotonic per-write-session counter; bumped each time the store is opened for
    /// writing. Lets a reader detect a newer writer (PR4 single-writer hygiene).
    pub lineage_epoch: u64,
    /// Identity of the writer that first stamped this lineage (entropy from instance_id
    /// + open time). Stable across the lineage; carried forward on each save.
    pub writer_uuid: u128,
    /// Stable hash of (model_id, embed_dim, text_format_version).
    pub vector_space_id: u64,
}

impl StoreHeader {
    /// FNV-1a over model_id + embed_dim + text_format_version. Stable across processes.
    pub fn compute_vector_space_id(model_id: &str, embed_dim: u32, text_format_version: u32) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |b: u8| { h ^= b as u64; h = h.wrapping_mul(0x0000_0100_0000_01b3); };
        for b in model_id.bytes() { mix(b); }
        for b in embed_dim.to_le_bytes() { mix(b); }
        for b in text_format_version.to_le_bytes() { mix(b); }
        h
    }

    /// The vector_space_id this binary compiles for.
    pub fn compiled_vector_space_id() -> u64 {
        Self::compute_vector_space_id(
            crate::ops::EMBED_MODEL_ID,
            crate::ops::EMBED_DIM as u32,
            crate::ops::TEXT_FORMAT_VERSION,
        )
    }

    /// Build a header describing the compiled vector space with the given lineage.
    pub fn current(lineage_epoch: u64, writer_uuid: u128) -> Self {
        let model_id = crate::ops::EMBED_MODEL_ID.to_string();
        let embed_dim = crate::ops::EMBED_DIM as u32;
        let text_format_version = crate::ops::TEXT_FORMAT_VERSION;
        StoreHeader {
            format_version: STORE_FORMAT_VERSION,
            embed_dim,
            vector_space_id: Self::compute_vector_space_id(&model_id, embed_dim, text_format_version),
            model_id,
            text_format_version,
            lineage_epoch,
            writer_uuid,
        }
    }

    /// True when this header's vector space matches the binary's compiled constants.
    pub fn matches_compiled(&self) -> bool {
        self.embed_dim == crate::ops::EMBED_DIM as u32
            && self.model_id == crate::ops::EMBED_MODEL_ID
            && self.vector_space_id == Self::compiled_vector_space_id()
    }

    /// Persist to `path` (magic + bincode body) via tmp+rename+fsync.
    pub fn save(&self, path: &Path) -> Result<()> {
        let tmp = path.with_extension("shdr.tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&SHDR_MAGIC.to_le_bytes())?;
            let body = bincode::serialize(self).map_err(|e| FieldError::Serialization(e.to_string()))?;
            f.write_all(&body)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        if let Some(dir) = path.parent() {
            if let Ok(d) = std::fs::File::open(dir) { let _ = d.sync_all(); }
        }
        Ok(())
    }

    /// Load from `path`. Returns None if missing/corrupt/wrong-magic (legacy store).
    pub fn load(path: &Path) -> Option<StoreHeader> {
        let bytes = std::fs::read(path).ok()?;
        if bytes.len() < 8 { return None; }
        let magic = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
        if magic != SHDR_MAGIC { return None; }
        bincode::deserialize(&bytes[8..]).ok()
    }
}

// ── Compile-time guard: MemoryState size must match snapshot magic ────────────
// If you add/remove fields to MemoryState, this assert will fail.
// You MUST: (1) bump FULL_SNAPSHOT_MAGIC, (2) add a LegacyMemoryStateVN,
// (3) add migration in FullSnapshot::load(), (4) update this constant.
// bincode is positional — #[serde(default)] does NOT work with it.
// V12: added `staged: bool` + `invalidated_by: Option<String>`. Exact size depends on
// compiler field-reordering (repr(Rust) may pack bools together). Guard is a range check.
const _: () = assert!(
    std::mem::size_of::<MemoryState>() >= 200,
    "MemoryState shrank below V9 baseline — check for accidental field removal"
);

// ── Legacy SemanticIndex (pre-ANN rewrite) ───────────────────────────────────

#[derive(Serialize, Deserialize)]
struct LegacySemanticIndex {
    embeddings: std::collections::HashMap<MemoryId, Vec<f32>>,
    deleted: std::collections::HashSet<MemoryId>,
}

// ── Legacy MemoryState (V4 and V5 snapshots, 14-field layout) ────────────────

/// MemoryPayload as serialized in V1–V12 snapshots (pre-Phase-4).
/// Lacks `harness: Option<String>`.
#[derive(Serialize, Deserialize)]
struct LegacyMemoryPayloadV12 {
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
    pub artifact_refs: Vec<crate::ops::ArtifactRef>,
    pub source_session: Option<String>,
    pub source_tool: Option<String>,
}
impl LegacyMemoryPayloadV12 {
    fn upgrade(self) -> MemoryPayload {
        let harness = self.source_tool.as_deref().map(|t| {
            if t.starts_with("codex") { "codex".to_string() } else { "claude-code".to_string() }
        });
        MemoryPayload {
            memory_id: self.memory_id,
            version: self.version,
            chunk_hash: self.chunk_hash,
            created_at_ms: self.created_at_ms,
            authored_at_ms: self.authored_at_ms,
            kind: self.kind,
            realm: self.realm,
            content: self.content,
            embedding_model: self.embedding_model,
            embedding: self.embedding,
            artifact_refs: self.artifact_refs,
            source_session: self.source_session,
            source_tool: self.source_tool,
            harness,
            provenance: "human".to_string(),
            candidate: false,
            embedding_model_id: "legacy-256".to_string(),
            embedding_dim: 256,
        }
    }
}

fn upgrade_payloads(m: std::collections::HashMap<MemoryId, LegacyMemoryPayloadV12>) -> std::collections::HashMap<MemoryId, MemoryPayload> {
    m.into_iter().map(|(id, p)| (id, p.upgrade())).collect()
}

/// MemoryState as serialized in V4 snapshots (pre-CTM, 14 fields).
/// Lacks RetrievalHistory, embed_pending, MemoryStatus, and EpistemicStatus.
#[derive(Serialize, Deserialize)]
struct LegacyMemoryStateV4 {
    pub memory_id: MemoryId,
    pub current_version: u32,
    pub current_chunk_hash: crate::ids::ChunkHash,
    pub deleted: bool,
    pub strength: f32,
    pub decay_rate: f32,
    pub confidence: f32,
    pub access_count: u32,
    pub last_accessed_ms: i64,
    pub last_strengthened_ms: i64,
    pub created_at_ms: i64,
    pub pinned: bool,
    pub tier: u8,
    pub last_state_op_ts_ms: i64,
}

impl LegacyMemoryStateV4 {
    fn upgrade(self) -> MemoryState {
        MemoryState {
            memory_id: self.memory_id,
            current_version: self.current_version,
            current_chunk_hash: self.current_chunk_hash,
            deleted: self.deleted,
            strength: self.strength,
            decay_rate: self.decay_rate,
            confidence: self.confidence,
            access_count: self.access_count,
            last_accessed_ms: self.last_accessed_ms,
            last_strengthened_ms: self.last_strengthened_ms,
            created_at_ms: self.created_at_ms,
            pinned: self.pinned,
            tier: self.tier,
            last_state_op_ts_ms: self.last_state_op_ts_ms,
            retrieval_history: RetrievalHistory::default(),
            embed_pending: false,
            status: MemoryStatus::Active,
            epistemic_status: EpistemicStatus::ToolDerived,
            surprise: 0.0,
            affect_valence: 0.0,
            affect_arousal: 0.0,
            access_timestamps: Vec::new(),
            competitive_weight: 0.0,
            lure_risk: 0.0,
            spacing_quality: 0.0,
            staged: false,
            invalidated_by: None,
        }
    }
}

/// MemoryState as serialized in V5 snapshots (15 fields: added RetrievalHistory).
/// Written by binaries from 2026-03-17 (commit c61a955) through 2026-03-25.
/// Lacks embed_pending, MemoryStatus, and EpistemicStatus.
#[derive(Serialize, Deserialize)]
struct LegacyMemoryStateV5 {
    pub memory_id: MemoryId,
    pub current_version: u32,
    pub current_chunk_hash: crate::ids::ChunkHash,
    pub deleted: bool,
    pub strength: f32,
    pub decay_rate: f32,
    pub confidence: f32,
    pub access_count: u32,
    pub last_accessed_ms: i64,
    pub last_strengthened_ms: i64,
    pub created_at_ms: i64,
    pub pinned: bool,
    pub tier: u8,
    pub last_state_op_ts_ms: i64,
    pub retrieval_history: RetrievalHistory,
}

impl LegacyMemoryStateV5 {
    fn upgrade(self) -> MemoryState {
        MemoryState {
            memory_id: self.memory_id,
            current_version: self.current_version,
            current_chunk_hash: self.current_chunk_hash,
            deleted: self.deleted,
            strength: self.strength,
            decay_rate: self.decay_rate,
            confidence: self.confidence,
            access_count: self.access_count,
            last_accessed_ms: self.last_accessed_ms,
            last_strengthened_ms: self.last_strengthened_ms,
            created_at_ms: self.created_at_ms,
            pinned: self.pinned,
            tier: self.tier,
            last_state_op_ts_ms: self.last_state_op_ts_ms,
            retrieval_history: self.retrieval_history,
            embed_pending: false,
            status: MemoryStatus::Active,
            epistemic_status: EpistemicStatus::ToolDerived,
            surprise: 0.0,
            affect_valence: 0.0,
            affect_arousal: 0.0,
            access_timestamps: Vec::new(),
            competitive_weight: 0.0,
            lure_risk: 0.0,
            spacing_quality: 0.0,
            staged: false,
            invalidated_by: None,
        }
    }
}

/// MemoryState as serialized in V6 snapshots (added affect + surprise, no access_timestamps).
/// Written by v5.10.0–v5.10.1 (2026-04-04 through 2026-04-10).
#[derive(Serialize, Deserialize)]
struct LegacyMemoryStateV6 {
    pub memory_id: MemoryId,
    pub current_version: u32,
    pub current_chunk_hash: crate::ids::ChunkHash,
    pub deleted: bool,
    pub strength: f32,
    pub decay_rate: f32,
    pub confidence: f32,
    pub access_count: u32,
    pub last_accessed_ms: i64,
    pub last_strengthened_ms: i64,
    pub created_at_ms: i64,
    pub pinned: bool,
    pub tier: u8,
    pub last_state_op_ts_ms: i64,
    pub retrieval_history: RetrievalHistory,
    pub embed_pending: bool,
    pub status: MemoryStatus,
    pub epistemic_status: EpistemicStatus,
    pub surprise: f32,
    pub affect_valence: f32,
    pub affect_arousal: f32,
}

impl LegacyMemoryStateV6 {
    fn upgrade(self) -> MemoryState {
        MemoryState {
            memory_id: self.memory_id,
            current_version: self.current_version,
            current_chunk_hash: self.current_chunk_hash,
            deleted: self.deleted,
            strength: self.strength,
            decay_rate: self.decay_rate,
            confidence: self.confidence,
            access_count: self.access_count,
            last_accessed_ms: self.last_accessed_ms,
            last_strengthened_ms: self.last_strengthened_ms,
            created_at_ms: self.created_at_ms,
            pinned: self.pinned,
            tier: self.tier,
            last_state_op_ts_ms: self.last_state_op_ts_ms,
            retrieval_history: self.retrieval_history,
            embed_pending: self.embed_pending,
            status: self.status,
            epistemic_status: self.epistemic_status,
            surprise: self.surprise,
            affect_valence: self.affect_valence,
            affect_arousal: self.affect_arousal,
            access_timestamps: Vec::new(),
            competitive_weight: 0.0,
            lure_risk: 0.0,
            spacing_quality: 0.0,
            staged: false,
            invalidated_by: None,
        }
    }
}

/// MemoryState as serialized in V7 snapshots (added access_timestamps, no interference fields).
/// Written by v5.11.0–v5.11.2 (2026-04-10).
#[derive(Serialize, Deserialize)]
struct LegacyMemoryStateV7 {
    pub memory_id: MemoryId,
    pub current_version: u32,
    pub current_chunk_hash: crate::ids::ChunkHash,
    pub deleted: bool,
    pub strength: f32,
    pub decay_rate: f32,
    pub confidence: f32,
    pub access_count: u32,
    pub last_accessed_ms: i64,
    pub last_strengthened_ms: i64,
    pub created_at_ms: i64,
    pub pinned: bool,
    pub tier: u8,
    pub last_state_op_ts_ms: i64,
    pub retrieval_history: RetrievalHistory,
    pub embed_pending: bool,
    pub status: MemoryStatus,
    pub epistemic_status: EpistemicStatus,
    pub surprise: f32,
    pub affect_valence: f32,
    pub affect_arousal: f32,
    pub access_timestamps: Vec<i64>,
}

impl LegacyMemoryStateV7 {
    fn upgrade(self) -> MemoryState {
        MemoryState {
            memory_id: self.memory_id,
            current_version: self.current_version,
            current_chunk_hash: self.current_chunk_hash,
            deleted: self.deleted,
            strength: self.strength,
            decay_rate: self.decay_rate,
            confidence: self.confidence,
            access_count: self.access_count,
            last_accessed_ms: self.last_accessed_ms,
            last_strengthened_ms: self.last_strengthened_ms,
            created_at_ms: self.created_at_ms,
            pinned: self.pinned,
            tier: self.tier,
            last_state_op_ts_ms: self.last_state_op_ts_ms,
            retrieval_history: self.retrieval_history,
            embed_pending: self.embed_pending,
            status: self.status,
            epistemic_status: self.epistemic_status,
            surprise: self.surprise,
            affect_valence: self.affect_valence,
            affect_arousal: self.affect_arousal,
            access_timestamps: self.access_timestamps,
            competitive_weight: 0.0,
            lure_risk: 0.0,
            spacing_quality: 0.0,
            staged: false,
            invalidated_by: None,
        }
    }
}

// ── Legacy snapshot structs ───────────────────────────────────────────────────

/// V1 snapshot: pre-ANN SemanticIndex + 14-field MemoryState.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV1 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, LegacyMemoryPayloadV12>,
    pub states: HashMap<MemoryId, LegacyMemoryStateV4>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: LegacySemanticIndex,
}

/// V4 snapshot: SemanticIndex (ANN), 14-field MemoryState, no coactivation_stats.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV4 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, LegacyMemoryPayloadV12>,
    pub states: HashMap<MemoryId, LegacyMemoryStateV4>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
}

/// V5 snapshot: SemanticIndex (ANN), 15-field MemoryState (with RetrievalHistory), with coactivation_stats.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV5 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, LegacyMemoryPayloadV12>,
    pub states: HashMap<MemoryId, LegacyMemoryStateV5>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
}

/// V6 snapshot: full MemoryState with affect/surprise but no access_timestamps.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV6 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, LegacyMemoryPayloadV12>,
    pub states: HashMap<MemoryId, LegacyMemoryStateV6>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
}

/// V7 snapshot: full MemoryState with access_timestamps but no interference fields.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV7 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, LegacyMemoryPayloadV12>,
    pub states: HashMap<MemoryId, LegacyMemoryStateV7>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
}

// ── V8 snapshot struct (no top-level sidecars) ───────────────────────────────

/// MemoryState as it existed before Phase 5 write-gate (v1.0.12).
/// Identical to current MemoryState minus `staged` and `invalidated_by`.
/// Used to deserialise V8–V11 snapshots written by older binaries.
#[derive(Serialize, Deserialize)]
struct LegacyMemoryStateV11 {
    pub memory_id: MemoryId,
    pub current_version: u32,
    pub current_chunk_hash: crate::ids::ChunkHash,
    pub deleted: bool,
    pub strength: f32,
    pub decay_rate: f32,
    pub confidence: f32,
    pub access_count: u32,
    pub last_accessed_ms: i64,
    pub last_strengthened_ms: i64,
    pub created_at_ms: i64,
    pub pinned: bool,
    pub tier: u8,
    pub last_state_op_ts_ms: i64,
    pub retrieval_history: RetrievalHistory,
    pub embed_pending: bool,
    pub status: MemoryStatus,
    pub epistemic_status: EpistemicStatus,
    pub surprise: f32,
    pub affect_valence: f32,
    pub affect_arousal: f32,
    pub access_timestamps: Vec<i64>,
    pub competitive_weight: f32,
    pub lure_risk: f32,
    pub spacing_quality: f32,
}

impl LegacyMemoryStateV11 {
    fn upgrade(self) -> MemoryState {
        MemoryState {
            memory_id: self.memory_id,
            current_version: self.current_version,
            current_chunk_hash: self.current_chunk_hash,
            deleted: self.deleted,
            strength: self.strength,
            decay_rate: self.decay_rate,
            confidence: self.confidence,
            access_count: self.access_count,
            last_accessed_ms: self.last_accessed_ms,
            last_strengthened_ms: self.last_strengthened_ms,
            created_at_ms: self.created_at_ms,
            pinned: self.pinned,
            tier: self.tier,
            last_state_op_ts_ms: self.last_state_op_ts_ms,
            retrieval_history: self.retrieval_history,
            embed_pending: self.embed_pending,
            status: self.status,
            epistemic_status: self.epistemic_status,
            surprise: self.surprise,
            affect_valence: self.affect_valence,
            affect_arousal: self.affect_arousal,
            access_timestamps: self.access_timestamps,
            competitive_weight: self.competitive_weight,
            lure_risk: self.lure_risk,
            spacing_quality: self.spacing_quality,
            staged: false,
            invalidated_by: None,
        }
    }
}

/// V11 snapshot layout (magic 0xF011_5741_7E00_000B): same sidecar layout as
/// current but MemoryState lacks `staged`/`invalidated_by`.
/// Also used for V9 and V10 which share the same MemoryState binary layout.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV11 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, LegacyMemoryPayloadV12>,
    pub states: HashMap<MemoryId, LegacyMemoryStateV11>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores: HashMap<MemoryId, i32>,
    pub correction_states: HashMap<u64, CorrectionState>,
}

#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV8 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, LegacyMemoryPayloadV12>,
    pub states: HashMap<MemoryId, LegacyMemoryStateV11>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
}

// ── Current snapshot struct (v9+) ────────────────────────────────────────────

/// V12 snapshot layout: same sidecar layout as V11; MemoryState has staged+invalidated_by;
/// MemoryPayload lacks `harness`.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV12 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, LegacyMemoryPayloadV12>,
    pub states: HashMap<MemoryId, MemoryState>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores: HashMap<MemoryId, i32>,
    pub correction_states: HashMap<u64, CorrectionState>,
}

#[derive(Serialize, Deserialize)]
pub struct FullSnapshot {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, MemoryPayload>,
    pub states: HashMap<MemoryId, MemoryState>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
    /// Persistent ack/nack usage scores keyed by MemoryId.
    pub ack_scores: HashMap<MemoryId, i32>,
    /// Persistent correction lifecycle states keyed by triplet id.
    pub correction_states: HashMap<u64, CorrectionState>,
    /// CEC Phase 1: append-only structured event tape for CDAWG reconstruction.
    pub event_tape: EventTape,
    /// CEC Phase 10: branch-point memory — chosen actions and rejected alternatives.
    pub decision_tape: DecisionTape,
    /// CEC Phase 11: Turīya Monitor — rolling health samples across CEC organs.
    pub turiya_monitor: TuriyaMonitor,
    /// Tier-0 observation extraction state — structured facts extracted from conversational turns.
    pub observer_state: ObserverState,
    /// V20: Interaction Ledger — reads/writes/outcomes as first-class events + versioned assertions.
    pub interaction_ledger: InteractionLedger,
    /// V21: Predicate Store — executable checks attached to memories for falsifiability.
    pub predicate_store: PredicateStore,
}

/// MemoryPayload as serialized in V16 snapshots (pre-Phase-17).
/// Lacks `provenance: String` and `candidate: bool`.
#[derive(Serialize, Deserialize)]
struct LegacyMemoryPayloadV16 {
    pub memory_id:       crate::ids::MemoryId,
    pub version:         u32,
    pub chunk_hash:      crate::ids::ChunkHash,
    pub created_at_ms:   i64,
    pub authored_at_ms:  i64,
    pub kind:            String,
    pub realm:           String,
    pub content:         Vec<u8>,
    pub embedding_model: String,
    pub embedding:       Vec<f32>,
    pub artifact_refs:   Vec<crate::ops::ArtifactRef>,
    pub source_session:  Option<String>,
    pub source_tool:     Option<String>,
    pub harness:         Option<String>,
}
impl LegacyMemoryPayloadV16 {
    fn upgrade(self) -> crate::payload::MemoryPayload {
        crate::payload::MemoryPayload {
            memory_id:       self.memory_id,
            version:         self.version,
            chunk_hash:      self.chunk_hash,
            created_at_ms:   self.created_at_ms,
            authored_at_ms:  self.authored_at_ms,
            kind:            self.kind,
            realm:           self.realm,
            content:         self.content,
            embedding_model: self.embedding_model,
            embedding:       self.embedding,
            artifact_refs:   self.artifact_refs,
            source_session:  self.source_session,
            source_tool:     self.source_tool,
            harness:         self.harness,
            provenance:          "human".to_string(),
            candidate:           false,
            embedding_model_id:  "legacy-256".to_string(),
            embedding_dim:       256,
        }
    }
}

fn upgrade_payloads_v16(
    m: std::collections::HashMap<crate::ids::MemoryId, LegacyMemoryPayloadV16>,
) -> std::collections::HashMap<crate::ids::MemoryId, crate::payload::MemoryPayload> {
    m.into_iter().map(|(id, p)| (id, p.upgrade())).collect()
}

/// MemoryPayload as serialized in V18 snapshots (pre-V19).
/// Lacks `embedding_model_id: String` and `embedding_dim: u32`.
#[derive(Serialize, Deserialize)]
struct LegacyMemoryPayloadV18 {
    pub memory_id:       crate::ids::MemoryId,
    pub version:         u32,
    pub chunk_hash:      crate::ids::ChunkHash,
    pub created_at_ms:   i64,
    pub authored_at_ms:  i64,
    pub kind:            String,
    pub realm:           String,
    pub content:         Vec<u8>,
    pub embedding_model: String,
    pub embedding:       Vec<f32>,
    pub artifact_refs:   Vec<crate::ops::ArtifactRef>,
    pub source_session:  Option<String>,
    pub source_tool:     Option<String>,
    pub harness:         Option<String>,
    pub provenance:      String,
    pub candidate:       bool,
}
impl LegacyMemoryPayloadV18 {
    fn upgrade(self) -> crate::payload::MemoryPayload {
        crate::payload::MemoryPayload {
            memory_id:          self.memory_id,
            version:            self.version,
            chunk_hash:         self.chunk_hash,
            created_at_ms:      self.created_at_ms,
            authored_at_ms:     self.authored_at_ms,
            kind:               self.kind,
            realm:              self.realm,
            content:            self.content,
            embedding_model:    self.embedding_model,
            embedding:          vec![],   // cleared — stale 256-d vectors must not enter 768-d HNSW
            artifact_refs:      self.artifact_refs,
            source_session:     self.source_session,
            source_tool:        self.source_tool,
            harness:            self.harness,
            provenance:         self.provenance,
            candidate:          self.candidate,
            embedding_model_id: "legacy-256".to_string(),
            embedding_dim:      256,
        }
    }
}

fn upgrade_payloads_v18(
    m: std::collections::HashMap<crate::ids::MemoryId, LegacyMemoryPayloadV18>,
) -> std::collections::HashMap<crate::ids::MemoryId, crate::payload::MemoryPayload> {
    m.into_iter().map(|(id, p)| (id, p.upgrade())).collect()
}

/// V16 snapshot layout — FullSnapshot without provenance/candidate on MemoryPayload.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV16 {
    pub snapshot_seqno:      u64,
    pub payloads:            HashMap<MemoryId, LegacyMemoryPayloadV16>,
    pub states:              HashMap<MemoryId, MemoryState>,
    pub assoc_edges:         HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts:           HashMap<String, ArtifactId>,
    pub artifact_paths:      HashMap<ArtifactId, String>,
    pub time_idx:            TemporalIndex,
    pub keyword_idx:         KeywordIndex,
    pub artifact_idx:        ArtifactIndex,
    pub triplet_store:       TripletStore,
    pub symbol_idx:          SymbolIndex,
    pub call_graph:          CallGraph,
    pub code_files:          CodeFileIndex,
    pub semantic_idx:        SemanticIndex,
    pub coactivation_stats:  HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores:          HashMap<MemoryId, i32>,
    pub correction_states:   HashMap<u64, CorrectionState>,
    pub event_tape:          EventTape,
    pub decision_tape:       DecisionTape,
    pub turiya_monitor:      TuriyaMonitor,
}

/// V18 snapshot layout — FullSnapshot with observer_state but without embedding_model_id/embedding_dim on payloads.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV18 {
    pub snapshot_seqno:      u64,
    pub payloads:            HashMap<MemoryId, LegacyMemoryPayloadV18>,
    pub states:              HashMap<MemoryId, MemoryState>,
    pub assoc_edges:         HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts:           HashMap<String, ArtifactId>,
    pub artifact_paths:      HashMap<ArtifactId, String>,
    pub time_idx:            TemporalIndex,
    pub keyword_idx:         KeywordIndex,
    pub artifact_idx:        ArtifactIndex,
    pub triplet_store:       TripletStore,
    pub symbol_idx:          SymbolIndex,
    pub call_graph:          CallGraph,
    pub code_files:          CodeFileIndex,
    pub semantic_idx:        SemanticIndex,
    pub coactivation_stats:  HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores:          HashMap<MemoryId, i32>,
    pub correction_states:   HashMap<u64, CorrectionState>,
    pub event_tape:          EventTape,
    pub decision_tape:       DecisionTape,
    pub turiya_monitor:      TuriyaMonitor,
    pub observer_state:      ObserverState,
}

/// V19 snapshot layout — FullSnapshot without interaction_ledger.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV19 {
    pub snapshot_seqno:      u64,
    pub payloads:            HashMap<MemoryId, MemoryPayload>,
    pub states:              HashMap<MemoryId, MemoryState>,
    pub assoc_edges:         HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts:           HashMap<String, ArtifactId>,
    pub artifact_paths:      HashMap<ArtifactId, String>,
    pub time_idx:            TemporalIndex,
    pub keyword_idx:         KeywordIndex,
    pub artifact_idx:        ArtifactIndex,
    pub triplet_store:       TripletStore,
    pub symbol_idx:          SymbolIndex,
    pub call_graph:          CallGraph,
    pub code_files:          CodeFileIndex,
    pub semantic_idx:        SemanticIndex,
    pub coactivation_stats:  HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores:          HashMap<MemoryId, i32>,
    pub correction_states:   HashMap<u64, CorrectionState>,
    pub event_tape:          EventTape,
    pub decision_tape:       DecisionTape,
    pub turiya_monitor:      TuriyaMonitor,
    pub observer_state:      ObserverState,
}

/// V20 snapshot layout — FullSnapshot with interaction_ledger but without predicate_store.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV20 {
    pub snapshot_seqno:      u64,
    pub payloads:            HashMap<MemoryId, MemoryPayload>,
    pub states:              HashMap<MemoryId, MemoryState>,
    pub assoc_edges:         HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts:           HashMap<String, ArtifactId>,
    pub artifact_paths:      HashMap<ArtifactId, String>,
    pub time_idx:            TemporalIndex,
    pub keyword_idx:         KeywordIndex,
    pub artifact_idx:        ArtifactIndex,
    pub triplet_store:       TripletStore,
    pub symbol_idx:          SymbolIndex,
    pub call_graph:          CallGraph,
    pub code_files:          CodeFileIndex,
    pub semantic_idx:        SemanticIndex,
    pub coactivation_stats:  HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores:          HashMap<MemoryId, i32>,
    pub correction_states:   HashMap<u64, CorrectionState>,
    pub event_tape:          EventTape,
    pub decision_tape:       DecisionTape,
    pub turiya_monitor:      TuriyaMonitor,
    pub observer_state:      ObserverState,
    pub interaction_ledger:  InteractionLedger,
}

/// V17 snapshot layout — FullSnapshot without observer_state.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV17 {
    pub snapshot_seqno:      u64,
    pub payloads:            HashMap<MemoryId, MemoryPayload>,
    pub states:              HashMap<MemoryId, MemoryState>,
    pub assoc_edges:         HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts:           HashMap<String, ArtifactId>,
    pub artifact_paths:      HashMap<ArtifactId, String>,
    pub time_idx:            TemporalIndex,
    pub keyword_idx:         KeywordIndex,
    pub artifact_idx:        ArtifactIndex,
    pub triplet_store:       TripletStore,
    pub symbol_idx:          SymbolIndex,
    pub call_graph:          CallGraph,
    pub code_files:          CodeFileIndex,
    pub semantic_idx:        SemanticIndex,
    pub coactivation_stats:  HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores:          HashMap<MemoryId, i32>,
    pub correction_states:   HashMap<u64, CorrectionState>,
    pub event_tape:          EventTape,
    pub decision_tape:       DecisionTape,
    pub turiya_monitor:      TuriyaMonitor,
}

/// V15 snapshot layout — FullSnapshot without turiya_monitor.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV15 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, MemoryPayload>,
    pub states: HashMap<MemoryId, MemoryState>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores: HashMap<MemoryId, i32>,
    pub correction_states: HashMap<u64, CorrectionState>,
    pub event_tape: EventTape,
    pub decision_tape: DecisionTape,
}

/// V14 snapshot layout — FullSnapshot without decision_tape.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV14 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, MemoryPayload>,
    pub states: HashMap<MemoryId, MemoryState>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores: HashMap<MemoryId, i32>,
    pub correction_states: HashMap<u64, CorrectionState>,
    pub event_tape: EventTape,
}

/// V13 snapshot layout — FullSnapshot without event_tape.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV13 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, MemoryPayload>,
    pub states: HashMap<MemoryId, MemoryState>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores: HashMap<MemoryId, i32>,
    pub correction_states: HashMap<u64, CorrectionState>,
}

impl FullSnapshot {
    pub fn save(&self, path: &Path) -> Result<()> {
        let tmp = path.with_extension("snapshot.tmp");
        {
            let f = std::fs::File::create(&tmp)?;
            let mut w = BufWriter::new(f);
            w.write_all(&FULL_SNAPSHOT_MAGIC.to_le_bytes())?;
            bincode::serialize_into(&mut w, self)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            w.flush()?;
            // fsync data+magic to disk before the rename commits the file, so a crash
            // can't leave a renamed-but-truncated snapshot whose magic still reads valid.
            w.into_inner()
                .map_err(|e| FieldError::Manifest(e.to_string()))?
                .sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        // fsync the parent directory so the rename entry itself survives a crash.
        if let Some(dir) = path.parent() {
            if let Ok(d) = std::fs::File::open(dir) { let _ = d.sync_all(); }
        }
        Ok(())
    }

    /// Save payload content to a `.pld` sidecar.
    /// Format: [PLD_MAGIC:u64][count:u64]([id:u64][len:u32][bytes:u8×len])×count
    pub fn save_payload_sidecar(path: &Path, payloads: &HashMap<MemoryId, crate::payload::MemoryPayload>) -> std::io::Result<()> {
        let tmp = path.with_extension("pld.tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&PLD_MAGIC.to_le_bytes())?;
            f.write_all(&(payloads.len() as u64).to_le_bytes())?;
            for (&id, payload) in payloads {
                f.write_all(&id.to_le_bytes())?;
                f.write_all(&(payload.content.len() as u32).to_le_bytes())?;
                f.write_all(&payload.content)?;
            }
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load payload content from a `.pld` sidecar into an existing payload map.
    /// Returns false if the file is missing or corrupt (caller falls back to bincode content).
    pub fn load_payload_sidecar(path: &Path, payloads: &mut HashMap<MemoryId, crate::payload::MemoryPayload>) -> bool {
        let bytes = match std::fs::read(path) { Ok(b) => b, Err(_) => return false };
        if bytes.len() < 16 { return false; }
        let magic = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        if magic != PLD_MAGIC { return false; }
        let count = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let mut off = 16usize;
        for _ in 0..count {
            if off + 12 > bytes.len() { return false; }
            let id = u64::from_le_bytes(bytes[off..off+8].try_into().unwrap());
            let len = u32::from_le_bytes(bytes[off+8..off+12].try_into().unwrap()) as usize;
            off += 12;
            if off + len > bytes.len() { return false; }
            if let Some(payload) = payloads.get_mut(&id) {
                payload.content = bytes[off..off+len].to_vec();
            }
            off += len;
        }
        eprintln!("[chitta-field] loaded .pld sidecar: {} content entries", count);
        true
    }

    /// Read only the magic and snapshot_seqno without deserializing the full snapshot.
    pub fn peek_seqno(path: &Path) -> Result<u64> {
        let f = std::fs::File::open(path)?;
        let mut r = BufReader::new(f);
        let mut buf = [0u8; 16];
        r.read_exact(&mut buf)?;
        let magic = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        if magic != FULL_SNAPSHOT_MAGIC
            && magic != FULL_SNAPSHOT_MAGIC_V18
            && magic != FULL_SNAPSHOT_MAGIC_V17
            && magic != FULL_SNAPSHOT_MAGIC_V16
            && magic != FULL_SNAPSHOT_MAGIC_V15
            && magic != FULL_SNAPSHOT_MAGIC_V14
            && magic != FULL_SNAPSHOT_MAGIC_V13
            && magic != FULL_SNAPSHOT_MAGIC_V12
            && magic != FULL_SNAPSHOT_MAGIC_V11
            && magic != FULL_SNAPSHOT_MAGIC_V10
            && magic != FULL_SNAPSHOT_MAGIC_V9
            && magic != FULL_SNAPSHOT_MAGIC_V8
            && magic != FULL_SNAPSHOT_MAGIC_V7
            && magic != FULL_SNAPSHOT_MAGIC_V6
            && magic != FULL_SNAPSHOT_MAGIC_V5
            && magic != FULL_SNAPSHOT_MAGIC_V4
            && magic != FULL_SNAPSHOT_MAGIC_V1
            && magic != FULL_SNAPSHOT_MAGIC_V19
        && magic != FULL_SNAPSHOT_MAGIC_V20
        {
            return Err(FieldError::Manifest("invalid full snapshot magic".to_string()));
        }
        Ok(u64::from_le_bytes(buf[8..16].try_into().unwrap()))
    }

    /// Load a full snapshot from disk. Transparently migrates all legacy formats.
    /// Uses a streaming BufReader — never reads the whole file into a Vec<u8>.
    pub fn load(path: &Path) -> Result<Self> {
        let mut snap = Self::load_inner(path)?;
        snap.triplet_store.rebuild_indexes();
        Ok(snap)
    }

    fn load_inner(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| FieldError::Manifest(e.to_string()))?;
        let mut r = BufReader::with_capacity(1 << 20, file);
        let mut magic_buf = [0u8; 8];
        r.read_exact(&mut magic_buf)
            .map_err(|_| FieldError::Manifest("snapshot too short".to_string()))?;
        let magic = u64::from_le_bytes(magic_buf);

        if magic == FULL_SNAPSHOT_MAGIC {
            // V21: current format with predicate_store.
            let mut snap: Self = bincode::deserialize_from(&mut r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            for state in snap.states.values_mut() { state.sanitize(); }
            return Ok(snap);
        }

        if magic == FULL_SNAPSHOT_MAGIC_V20 {
            // V20→V21: add predicate_store (default empty).
            eprintln!("[chitta-field] migrating v20 snapshot → v21 (predicate store)");
            let leg: LegacyFullSnapshotV20 = bincode::deserialize_from(&mut r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut states = leg.states;
            for state in states.values_mut() { state.sanitize(); }
            return Ok(FullSnapshot {
                snapshot_seqno:     leg.snapshot_seqno,
                payloads:           leg.payloads,
                states,
                assoc_edges:        leg.assoc_edges,
                artifacts:          leg.artifacts,
                artifact_paths:     leg.artifact_paths,
                time_idx:           leg.time_idx,
                keyword_idx:        leg.keyword_idx,
                artifact_idx:       leg.artifact_idx,
                triplet_store:      leg.triplet_store,
                symbol_idx:         leg.symbol_idx,
                call_graph:         leg.call_graph,
                code_files:         leg.code_files,
                semantic_idx:       leg.semantic_idx,
                coactivation_stats: leg.coactivation_stats,
                ack_scores:         leg.ack_scores,
                correction_states:  leg.correction_states,
                event_tape:         leg.event_tape,
                decision_tape:      leg.decision_tape,
                turiya_monitor:     leg.turiya_monitor,
                observer_state:     leg.observer_state,
                interaction_ledger: leg.interaction_ledger,
                predicate_store:    PredicateStore::default(),
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V19 {
            // V19→V20: add interaction_ledger (default empty).
            eprintln!("[chitta-field] migrating v19 snapshot → v20 (interaction ledger)");
            let leg: LegacyFullSnapshotV19 = bincode::deserialize_from(&mut r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut states = leg.states;
            for state in states.values_mut() { state.sanitize(); }
            return Ok(FullSnapshot {
                snapshot_seqno:     leg.snapshot_seqno,
                payloads:           leg.payloads,
                states,
                assoc_edges:        leg.assoc_edges,
                artifacts:          leg.artifacts,
                artifact_paths:     leg.artifact_paths,
                time_idx:           leg.time_idx,
                keyword_idx:        leg.keyword_idx,
                artifact_idx:       leg.artifact_idx,
                triplet_store:      leg.triplet_store,
                symbol_idx:         leg.symbol_idx,
                call_graph:         leg.call_graph,
                code_files:         leg.code_files,
                semantic_idx:       leg.semantic_idx,
                coactivation_stats: leg.coactivation_stats,
                ack_scores:         leg.ack_scores,
                correction_states:  leg.correction_states,
                event_tape:         leg.event_tape,
                decision_tape:      leg.decision_tape,
                turiya_monitor:     leg.turiya_monitor,
                observer_state:     leg.observer_state,
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V18 {
            // V18→V19: add embedding_model_id/embedding_dim; clear stale 256-d embeddings;
            // mark all non-deleted memories with content as embed_pending for 768-d re-embedding.
            eprintln!("[chitta-field] migrating v18 snapshot → v19 (Ollama 768-d re-embedding)");
            let leg: LegacyFullSnapshotV18 = bincode::deserialize_from(&mut r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let payloads = upgrade_payloads_v18(leg.payloads);
            let mut states = leg.states;
            // Mark every non-deleted memory with content as embed_pending so the
            // backfill thread re-embeds with nomic-embed-text:v1.5 at 768 dims.
            for (id, state) in states.iter_mut() {
                if state.deleted { continue; }
                let has_content = payloads.get(id)
                    .map(|p| p.content.len() >= 10)
                    .unwrap_or(false);
                if has_content {
                    state.embed_pending = true;
                }
            }
            for state in states.values_mut() { state.sanitize(); }
            return Ok(FullSnapshot {
                snapshot_seqno:     leg.snapshot_seqno,
                payloads,
                states,
                assoc_edges:        leg.assoc_edges,
                artifacts:          leg.artifacts,
                artifact_paths:     leg.artifact_paths,
                time_idx:           leg.time_idx,
                keyword_idx:        leg.keyword_idx,
                artifact_idx:       leg.artifact_idx,
                triplet_store:      leg.triplet_store,
                symbol_idx:         leg.symbol_idx,
                call_graph:         leg.call_graph,
                code_files:         leg.code_files,
                semantic_idx:       leg.semantic_idx,
                coactivation_stats: leg.coactivation_stats,
                ack_scores:         leg.ack_scores,
                correction_states:  leg.correction_states,
                event_tape:         leg.event_tape,
                decision_tape:      leg.decision_tape,
                turiya_monitor:     leg.turiya_monitor,
                observer_state:     leg.observer_state,
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V17 {
            // V17→V18: add empty ObserverState.
            eprintln!("[chitta-field] migrating v17 snapshot → v18 (adding observer_state)");
            let leg: LegacyFullSnapshotV17 = bincode::deserialize_from(&mut r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut snap = FullSnapshot {
                snapshot_seqno:     leg.snapshot_seqno,
                payloads:           leg.payloads,
                states:             leg.states,
                assoc_edges:        leg.assoc_edges,
                artifacts:          leg.artifacts,
                artifact_paths:     leg.artifact_paths,
                time_idx:           leg.time_idx,
                keyword_idx:        leg.keyword_idx,
                artifact_idx:       leg.artifact_idx,
                triplet_store:      leg.triplet_store,
                symbol_idx:         leg.symbol_idx,
                call_graph:         leg.call_graph,
                code_files:         leg.code_files,
                semantic_idx:       leg.semantic_idx,
                coactivation_stats: leg.coactivation_stats,
                ack_scores:         leg.ack_scores,
                correction_states:  leg.correction_states,
                event_tape:         leg.event_tape,
                decision_tape:      leg.decision_tape,
                turiya_monitor:     leg.turiya_monitor,
                observer_state:     ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
            };
            for state in snap.states.values_mut() { state.sanitize(); }
            return Ok(snap);
        }

        if magic == FULL_SNAPSHOT_MAGIC_V16 {
            // V16→V17: MemoryPayload gains provenance + candidate.
            eprintln!("[chitta-field] migrating v16 snapshot → v17 (adding provenance+candidate to payloads)");
            let leg: LegacyFullSnapshotV16 = bincode::deserialize_from(&mut r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut snap = FullSnapshot {
                snapshot_seqno:     leg.snapshot_seqno,
                payloads:           upgrade_payloads_v16(leg.payloads),
                states:             leg.states,
                assoc_edges:        leg.assoc_edges,
                artifacts:          leg.artifacts,
                artifact_paths:     leg.artifact_paths,
                time_idx:           leg.time_idx,
                keyword_idx:        leg.keyword_idx,
                artifact_idx:       leg.artifact_idx,
                triplet_store:      leg.triplet_store,
                symbol_idx:         leg.symbol_idx,
                call_graph:         leg.call_graph,
                code_files:         leg.code_files,
                semantic_idx:       leg.semantic_idx,
                coactivation_stats: leg.coactivation_stats,
                ack_scores:         leg.ack_scores,
                correction_states:  leg.correction_states,
                event_tape:         leg.event_tape,
                decision_tape:      leg.decision_tape,
                turiya_monitor:     leg.turiya_monitor,
                observer_state:     ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
            };
            for state in snap.states.values_mut() { state.sanitize(); }
            return Ok(snap);
        }

        if magic == FULL_SNAPSHOT_MAGIC_V15 {
            // V15→V16: add empty TuriyaMonitor.
            eprintln!("[chitta-field] migrating v15 snapshot → v16 (adding turiya_monitor)");
            let leg: LegacyFullSnapshotV15 = bincode::deserialize_from(&mut r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut snap = FullSnapshot {
                snapshot_seqno:    leg.snapshot_seqno,
                payloads:          leg.payloads,
                states:            leg.states,
                assoc_edges:       leg.assoc_edges,
                artifacts:         leg.artifacts,
                artifact_paths:    leg.artifact_paths,
                time_idx:          leg.time_idx,
                keyword_idx:       leg.keyword_idx,
                artifact_idx:      leg.artifact_idx,
                triplet_store:     leg.triplet_store,
                symbol_idx:        leg.symbol_idx,
                call_graph:        leg.call_graph,
                code_files:        leg.code_files,
                semantic_idx:      leg.semantic_idx,
                coactivation_stats:leg.coactivation_stats,
                ack_scores:        leg.ack_scores,
                correction_states: leg.correction_states,
                event_tape:        leg.event_tape,
                decision_tape:     leg.decision_tape,
                turiya_monitor:    TuriyaMonitor::new(),
                observer_state:    ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
            };
            for state in snap.states.values_mut() { state.sanitize(); }
            return Ok(snap);
        }

        if magic == FULL_SNAPSHOT_MAGIC_V14 {
            // V14→V15: add empty DecisionTape.
            eprintln!("[chitta-field] migrating v14 snapshot → v15 (adding decision_tape)");
            let leg: LegacyFullSnapshotV14 = bincode::deserialize_from(&mut r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut snap = FullSnapshot {
                snapshot_seqno:    leg.snapshot_seqno,
                payloads:          leg.payloads,
                states:            leg.states,
                assoc_edges:       leg.assoc_edges,
                artifacts:         leg.artifacts,
                artifact_paths:    leg.artifact_paths,
                time_idx:          leg.time_idx,
                keyword_idx:       leg.keyword_idx,
                artifact_idx:      leg.artifact_idx,
                triplet_store:     leg.triplet_store,
                symbol_idx:        leg.symbol_idx,
                call_graph:        leg.call_graph,
                code_files:        leg.code_files,
                semantic_idx:      leg.semantic_idx,
                coactivation_stats:leg.coactivation_stats,
                ack_scores:        leg.ack_scores,
                correction_states: leg.correction_states,
                event_tape:        leg.event_tape,
                decision_tape:     DecisionTape::new(),
                turiya_monitor:    TuriyaMonitor::new(),
                observer_state:    ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
            };
            for state in snap.states.values_mut() { state.sanitize(); }
            return Ok(snap);
        }

        if magic == FULL_SNAPSHOT_MAGIC_V13 {
            // V13→V14: add empty EventTape.
            eprintln!("[chitta-field] migrating v13 snapshot → v14 (adding event_tape)");
            let leg: LegacyFullSnapshotV13 = bincode::deserialize_from(&mut r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut snap = FullSnapshot {
                snapshot_seqno:     leg.snapshot_seqno,
                payloads:           leg.payloads,
                states:             leg.states,
                assoc_edges:        leg.assoc_edges,
                artifacts:          leg.artifacts,
                artifact_paths:     leg.artifact_paths,
                time_idx:           leg.time_idx,
                keyword_idx:        leg.keyword_idx,
                artifact_idx:       leg.artifact_idx,
                triplet_store:      leg.triplet_store,
                symbol_idx:         leg.symbol_idx,
                call_graph:         leg.call_graph,
                code_files:         leg.code_files,
                semantic_idx:       leg.semantic_idx,
                coactivation_stats: leg.coactivation_stats,
                ack_scores:         leg.ack_scores,
                correction_states:  leg.correction_states,
                event_tape:         EventTape::new(),
                decision_tape:      DecisionTape::new(),
                turiya_monitor:     TuriyaMonitor::new(),
                observer_state:     ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
            };
            for state in snap.states.values_mut() { state.sanitize(); }
            return Ok(snap);
        }

        if magic == FULL_SNAPSHOT_MAGIC_V12 {
            // v12: MemoryState has staged+invalidated_by; MemoryPayload lacks harness.
            eprintln!("[chitta-field] migrating v12 snapshot → v13 (adding harness to payloads)");
            let r = &mut r;
            let leg: LegacyFullSnapshotV12 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut snap = FullSnapshot {
                snapshot_seqno: leg.snapshot_seqno,
                payloads: upgrade_payloads(leg.payloads),
                states: leg.states,
                assoc_edges: leg.assoc_edges,
                artifacts: leg.artifacts,
                artifact_paths: leg.artifact_paths,
                time_idx: leg.time_idx,
                keyword_idx: leg.keyword_idx,
                artifact_idx: leg.artifact_idx,
                triplet_store: leg.triplet_store,
                symbol_idx: leg.symbol_idx,
                call_graph: leg.call_graph,
                code_files: leg.code_files,
                semantic_idx: leg.semantic_idx,
                coactivation_stats: leg.coactivation_stats,
                ack_scores: leg.ack_scores,
                correction_states: leg.correction_states,
                event_tape: EventTape::new(),
                decision_tape: DecisionTape::new(),
                turiya_monitor: TuriyaMonitor::new(),
                observer_state: ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
            };
            for state in snap.states.values_mut() { state.sanitize(); }
            return Ok(snap);
        }

        if magic == FULL_SNAPSHOT_MAGIC_V11
            || magic == FULL_SNAPSHOT_MAGIC_V10
            || magic == FULL_SNAPSHOT_MAGIC_V9
        {
            // v9/v10/v11: MemoryState lacks staged/invalidated_by; MemoryPayload lacks harness.
            if magic == FULL_SNAPSHOT_MAGIC_V11 {
                eprintln!("[chitta-field] migrating v11 snapshot → v13");
            } else {
                eprintln!("[chitta-field] migrating v9/v10 snapshot → v13");
            }
            let r = &mut r;
            let leg: LegacyFullSnapshotV11 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let states: HashMap<MemoryId, MemoryState> = leg.states
                .into_iter()
                .map(|(id, s)| { let mut m = s.upgrade(); m.sanitize(); (id, m) })
                .collect();
            return Ok(FullSnapshot {
                snapshot_seqno: leg.snapshot_seqno,
                payloads: upgrade_payloads(leg.payloads),
                states,
                assoc_edges: leg.assoc_edges,
                artifacts: leg.artifacts,
                artifact_paths: leg.artifact_paths,
                time_idx: leg.time_idx,
                keyword_idx: leg.keyword_idx,
                artifact_idx: leg.artifact_idx,
                triplet_store: leg.triplet_store,
                symbol_idx: leg.symbol_idx,
                call_graph: leg.call_graph,
                code_files: leg.code_files,
                semantic_idx: leg.semantic_idx,
                coactivation_stats: leg.coactivation_stats,
                ack_scores: leg.ack_scores,
                correction_states: leg.correction_states,
                event_tape: EventTape::new(),
                decision_tape: DecisionTape::new(),
                turiya_monitor: TuriyaMonitor::new(),
                observer_state: ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V8 {
            // V8: no top-level sidecars; MemoryState lacks staged/invalidated_by; payload lacks harness.
            eprintln!("[chitta-field] migrating v8 snapshot → v13");
            let r = &mut r;
            let v8: LegacyFullSnapshotV8 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let states: HashMap<MemoryId, MemoryState> = v8.states
                .into_iter()
                .map(|(id, s)| { let mut m = s.upgrade(); m.sanitize(); (id, m) })
                .collect();
            return Ok(FullSnapshot {
                snapshot_seqno: v8.snapshot_seqno,
                payloads: upgrade_payloads(v8.payloads),
                states,
                assoc_edges: v8.assoc_edges,
                artifacts: v8.artifacts,
                artifact_paths: v8.artifact_paths,
                time_idx: v8.time_idx,
                keyword_idx: v8.keyword_idx,
                artifact_idx: v8.artifact_idx,
                triplet_store: v8.triplet_store,
                symbol_idx: v8.symbol_idx,
                call_graph: v8.call_graph,
                code_files: v8.code_files,
                semantic_idx: v8.semantic_idx,
                coactivation_stats: v8.coactivation_stats,
                ack_scores: HashMap::new(),
                correction_states: HashMap::new(),
                event_tape: EventTape::new(),
                decision_tape: DecisionTape::new(),
                turiya_monitor: TuriyaMonitor::new(),
                observer_state: ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V7 {
            // V7: access_timestamps but no interference fields.
            eprintln!("[chitta-field] migrating v7 snapshot → v9");
            let r = &mut r;
            let v7: LegacyFullSnapshotV7 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let states = v7.states.into_iter().map(|(id, s)| { let mut m = s.upgrade(); m.sanitize(); (id, m) }).collect();
            return Ok(FullSnapshot {
                snapshot_seqno: v7.snapshot_seqno,
                payloads: upgrade_payloads(v7.payloads),
                states,
                assoc_edges: v7.assoc_edges,
                artifacts: v7.artifacts,
                artifact_paths: v7.artifact_paths,
                time_idx: v7.time_idx,
                keyword_idx: v7.keyword_idx,
                artifact_idx: v7.artifact_idx,
                triplet_store: v7.triplet_store,
                symbol_idx: v7.symbol_idx,
                call_graph: v7.call_graph,
                code_files: v7.code_files,
                semantic_idx: v7.semantic_idx,
                coactivation_stats: v7.coactivation_stats,
                ack_scores: HashMap::new(),
                correction_states: HashMap::new(),
                event_tape: EventTape::new(),
                decision_tape: DecisionTape::new(),
                turiya_monitor: TuriyaMonitor::new(),
                observer_state: ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V6 {
            // V6: MemoryState with affect/surprise but no access_timestamps.
            eprintln!("[chitta-field] migrating v6 snapshot → v9");
            let r = &mut r;
            let v6: LegacyFullSnapshotV6 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let states = v6.states.into_iter().map(|(id, s)| { let mut m = s.upgrade(); m.sanitize(); (id, m) }).collect();
            return Ok(FullSnapshot {
                snapshot_seqno: v6.snapshot_seqno,
                payloads: upgrade_payloads(v6.payloads),
                states,
                assoc_edges: v6.assoc_edges,
                artifacts: v6.artifacts,
                artifact_paths: v6.artifact_paths,
                time_idx: v6.time_idx,
                keyword_idx: v6.keyword_idx,
                artifact_idx: v6.artifact_idx,
                triplet_store: v6.triplet_store,
                symbol_idx: v6.symbol_idx,
                call_graph: v6.call_graph,
                code_files: v6.code_files,
                semantic_idx: v6.semantic_idx,
                coactivation_stats: v6.coactivation_stats,
                ack_scores: HashMap::new(),
                correction_states: HashMap::new(),
                event_tape: EventTape::new(),
                decision_tape: DecisionTape::new(),
                turiya_monitor: TuriyaMonitor::new(),
                observer_state: ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V5 {
            // V5: 14-field MemoryState + coactivation_stats.
            eprintln!("[chitta-field] migrating v5 snapshot → v9");
            let r = &mut r;
            let v5: LegacyFullSnapshotV5 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let states = v5.states.into_iter().map(|(id, s)| { let mut m = s.upgrade(); m.sanitize(); (id, m) }).collect();
            return Ok(FullSnapshot {
                snapshot_seqno: v5.snapshot_seqno,
                payloads: upgrade_payloads(v5.payloads),
                states,
                assoc_edges: v5.assoc_edges,
                artifacts: v5.artifacts,
                artifact_paths: v5.artifact_paths,
                time_idx: v5.time_idx,
                keyword_idx: v5.keyword_idx,
                artifact_idx: v5.artifact_idx,
                triplet_store: v5.triplet_store,
                symbol_idx: v5.symbol_idx,
                call_graph: v5.call_graph,
                code_files: v5.code_files,
                semantic_idx: v5.semantic_idx,
                coactivation_stats: v5.coactivation_stats,
                ack_scores: HashMap::new(),
                correction_states: HashMap::new(),
                event_tape: EventTape::new(),
                decision_tape: DecisionTape::new(),
                turiya_monitor: TuriyaMonitor::new(),
                observer_state: ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V4 {
            // V4: 14-field MemoryState, no coactivation_stats.
            eprintln!("[chitta-field] migrating v4 snapshot → v9");
            let r = &mut r;
            let v4: LegacyFullSnapshotV4 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let states = v4.states.into_iter().map(|(id, s)| { let mut m = s.upgrade(); m.sanitize(); (id, m) }).collect();
            return Ok(FullSnapshot {
                snapshot_seqno: v4.snapshot_seqno,
                payloads: upgrade_payloads(v4.payloads),
                states,
                assoc_edges: v4.assoc_edges,
                artifacts: v4.artifacts,
                artifact_paths: v4.artifact_paths,
                time_idx: v4.time_idx,
                keyword_idx: v4.keyword_idx,
                artifact_idx: v4.artifact_idx,
                triplet_store: v4.triplet_store,
                symbol_idx: v4.symbol_idx,
                call_graph: v4.call_graph,
                code_files: v4.code_files,
                semantic_idx: v4.semantic_idx,
                coactivation_stats: HashMap::new(),
                ack_scores: HashMap::new(),
                correction_states: HashMap::new(),
                event_tape: EventTape::new(),
                decision_tape: DecisionTape::new(),
                turiya_monitor: TuriyaMonitor::new(),
                observer_state: ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V1 {
            // V1: pre-ANN SemanticIndex + 14-field MemoryState.
            eprintln!("[chitta-field] migrating v1 snapshot → v6 (ANN index + MemoryStatus + EpistemicStatus)");
            let r = &mut r;
            let v1: LegacyFullSnapshotV1 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut semantic_idx = SemanticIndex::new();
            for (mem_id, emb) in v1.semantic_idx.embeddings {
                semantic_idx.upsert(mem_id, emb, None);
            }
            let states = v1.states.into_iter().map(|(id, s)| (id, s.upgrade())).collect::<HashMap<_, _>>();
            return Ok(FullSnapshot {
                snapshot_seqno: v1.snapshot_seqno,
                payloads: upgrade_payloads(v1.payloads),
                states,
                assoc_edges: v1.assoc_edges,
                artifacts: v1.artifacts,
                artifact_paths: v1.artifact_paths,
                time_idx: v1.time_idx,
                keyword_idx: v1.keyword_idx,
                artifact_idx: v1.artifact_idx,
                triplet_store: v1.triplet_store,
                symbol_idx: v1.symbol_idx,
                call_graph: v1.call_graph,
                code_files: v1.code_files,
                semantic_idx,
                coactivation_stats: HashMap::new(),
                ack_scores: HashMap::new(),
                correction_states: HashMap::new(),
                event_tape: EventTape::new(),
                decision_tape: DecisionTape::new(),
                turiya_monitor: TuriyaMonitor::new(),
                observer_state: ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
            });
        }

        Err(FieldError::Manifest(format!("unknown snapshot magic: {:#x}", magic)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_v4_state() -> LegacyMemoryStateV4 {
        LegacyMemoryStateV4 {
            memory_id: 42,
            current_version: 1,
            current_chunk_hash: [0u8; 32],
            deleted: false,
            strength: 0.8,
            decay_rate: 0.001,
            confidence: 0.9,
            access_count: 3,
            last_accessed_ms: 1_000_000,
            last_strengthened_ms: 900_000,
            created_at_ms: 800_000,
            pinned: false,
            tier: 0,
            last_state_op_ts_ms: 1_000_001,
        }
    }

    fn legacy_v5_state() -> LegacyMemoryStateV5 {
        LegacyMemoryStateV5 {
            memory_id: 99,
            current_version: 2,
            current_chunk_hash: [0u8; 32],
            deleted: false,
            strength: 0.7,
            decay_rate: 0.002,
            confidence: 0.85,
            access_count: 5,
            last_accessed_ms: 2_000_000,
            last_strengthened_ms: 1_900_000,
            created_at_ms: 1_800_000,
            pinned: true,
            tier: 1,
            last_state_op_ts_ms: 2_000_001,
            retrieval_history: RetrievalHistory::default(),
        }
    }

    #[test]
    fn v4_upgrade_sets_surprise_zero() {
        assert_eq!(legacy_v4_state().upgrade().surprise, 0.0);
    }

    #[test]
    fn v5_upgrade_sets_surprise_zero() {
        assert_eq!(legacy_v5_state().upgrade().surprise, 0.0);
    }

    /// Bincode positional deserialization of a V4-layout byte stream into the
    /// current MemoryState via the upgrade path must yield surprise == 0.0.
    /// `#[serde(default)]` alone does not save bincode (positional); only the
    /// explicit upgrade path inserting `surprise: 0.0` does.
    #[test]
    fn v4_bincode_roundtrip_surprise_zero() {
        let legacy = legacy_v4_state();
        let bytes = bincode::serialize(&legacy).expect("serialize");
        let decoded: LegacyMemoryStateV4 = bincode::deserialize(&bytes).expect("deserialize");
        let upgraded = decoded.upgrade();
        assert_eq!(upgraded.surprise, 0.0);
        assert_eq!(upgraded.strength, 0.8);
    }

    #[test]
    fn v5_bincode_roundtrip_surprise_zero() {
        let legacy = legacy_v5_state();
        let bytes = bincode::serialize(&legacy).expect("serialize");
        let decoded: LegacyMemoryStateV5 = bincode::deserialize(&bytes).expect("deserialize");
        let upgraded = decoded.upgrade();
        assert_eq!(upgraded.surprise, 0.0);
        assert_eq!(upgraded.strength, 0.7);
        assert_eq!(upgraded.tier, 1);
    }
}

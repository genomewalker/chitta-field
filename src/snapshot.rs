use crate::error::{FieldError, Result};
use crate::field::{AssocEdge, CoActivationStats};
use crate::hnsw::SemanticIndex;
use crate::ids::{ArtifactId, MemoryId};
use crate::organ::artifact::ArtifactIndex;
use crate::organ::callgraph::CallGraph;
use crate::organ::codefile::CodeFileIndex;
use crate::organ::keyword::KeywordIndex;
use crate::organ::symbol::SymbolIndex;
use crate::organ::temporal::TemporalIndex;
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
/// Current magic (v1.0.11+): content moved to .pld sidecar; bincode field is empty Vec.
const FULL_SNAPSHOT_MAGIC: u64 = 0xF011_5741_7E00_000B;

/// Magic for the payload content sidecar (.pld).
const PLD_MAGIC: u64 = 0x504C_4400_0000_0001; // "PLD\0\0\0\0\x01"

// ── Compile-time guard: MemoryState size must match snapshot magic ────────────
// If you add/remove fields to MemoryState, this assert will fail.
// You MUST: (1) bump FULL_SNAPSHOT_MAGIC, (2) add a LegacyMemoryStateVN,
// (3) add migration in FullSnapshot::load(), (4) update this constant.
// bincode is positional — #[serde(default)] does NOT work with it.
// V10: added `staged: bool` + `invalidated_by: Option<String>`. Exact size depends on
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
    pub payloads: HashMap<MemoryId, MemoryPayload>,
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
    pub payloads: HashMap<MemoryId, MemoryPayload>,
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
    pub payloads: HashMap<MemoryId, MemoryPayload>,
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
    pub payloads: HashMap<MemoryId, MemoryPayload>,
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
    pub payloads: HashMap<MemoryId, MemoryPayload>,
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

#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV8 {
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
}

// ── Current snapshot struct (v9+) ────────────────────────────────────────────

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
        }
        std::fs::rename(&tmp, path)?;
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
            && magic != FULL_SNAPSHOT_MAGIC_V10
            && magic != FULL_SNAPSHOT_MAGIC_V9
            && magic != FULL_SNAPSHOT_MAGIC_V8
            && magic != FULL_SNAPSHOT_MAGIC_V7
            && magic != FULL_SNAPSHOT_MAGIC_V6
            && magic != FULL_SNAPSHOT_MAGIC_V5
            && magic != FULL_SNAPSHOT_MAGIC_V4
            && magic != FULL_SNAPSHOT_MAGIC_V1
        {
            return Err(FieldError::Manifest("invalid full snapshot magic".to_string()));
        }
        Ok(u64::from_le_bytes(buf[8..16].try_into().unwrap()))
    }

    /// Load a full snapshot from disk. Transparently migrates all legacy formats.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        if bytes.len() < 8 {
            return Err(FieldError::Manifest("snapshot too short".to_string()));
        }
        let magic = u64::from_le_bytes(bytes[0..8].try_into().unwrap());

        if magic == FULL_SNAPSHOT_MAGIC {
            // v11: embeddings in .emb sidecar, content in .pld sidecar; both Vec fields empty in bincode.
            // Caller (field.rs) populates both from sidecars after this returns.
            let r = BufReader::new(&bytes[8..]);
            let mut snap: Self = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            for state in snap.states.values_mut() { state.sanitize(); }
            return Ok(snap);
        }

        if magic == FULL_SNAPSHOT_MAGIC_V10 {
            // v10: embeddings in .emb sidecar, content still in bincode.
            // Caller (field.rs) populates embeddings from .emb; content already present.
            let r = BufReader::new(&bytes[8..]);
            let mut snap: Self = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            for state in snap.states.values_mut() { state.sanitize(); }
            return Ok(snap);
        }

        if magic == FULL_SNAPSHOT_MAGIC_V9 {
            // v9: embeddings included in bincode (pre-sidecar format).
            let r = BufReader::new(&bytes[8..]);
            let mut snap: Self = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            for state in snap.states.values_mut() { state.sanitize(); }
            return Ok(snap);
        }

        if magic == FULL_SNAPSHOT_MAGIC_V8 {
            // V8: no top-level sidecars — migrate to v9 with empty sidecars.
            eprintln!("[chitta-field] migrating v8 snapshot → v9 (adding ack_scores + correction_states)");
            let r = BufReader::new(&bytes[8..]);
            let v8: LegacyFullSnapshotV8 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut states = v8.states;
            for s in states.values_mut() { s.sanitize(); }
            return Ok(FullSnapshot {
                snapshot_seqno: v8.snapshot_seqno,
                payloads: v8.payloads,
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
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V7 {
            // V7: access_timestamps but no interference fields.
            eprintln!("[chitta-field] migrating v7 snapshot → v9");
            let r = BufReader::new(&bytes[8..]);
            let v7: LegacyFullSnapshotV7 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let states = v7.states.into_iter().map(|(id, s)| { let mut m = s.upgrade(); m.sanitize(); (id, m) }).collect();
            return Ok(FullSnapshot {
                snapshot_seqno: v7.snapshot_seqno,
                payloads: v7.payloads,
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
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V6 {
            // V6: MemoryState with affect/surprise but no access_timestamps.
            eprintln!("[chitta-field] migrating v6 snapshot → v9");
            let r = BufReader::new(&bytes[8..]);
            let v6: LegacyFullSnapshotV6 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let states = v6.states.into_iter().map(|(id, s)| { let mut m = s.upgrade(); m.sanitize(); (id, m) }).collect();
            return Ok(FullSnapshot {
                snapshot_seqno: v6.snapshot_seqno,
                payloads: v6.payloads,
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
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V5 {
            // V5: 14-field MemoryState + coactivation_stats.
            eprintln!("[chitta-field] migrating v5 snapshot → v9");
            let r = BufReader::new(&bytes[8..]);
            let v5: LegacyFullSnapshotV5 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let states = v5.states.into_iter().map(|(id, s)| { let mut m = s.upgrade(); m.sanitize(); (id, m) }).collect();
            return Ok(FullSnapshot {
                snapshot_seqno: v5.snapshot_seqno,
                payloads: v5.payloads,
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
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V4 {
            // V4: 14-field MemoryState, no coactivation_stats.
            eprintln!("[chitta-field] migrating v4 snapshot → v9");
            let r = BufReader::new(&bytes[8..]);
            let v4: LegacyFullSnapshotV4 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let states = v4.states.into_iter().map(|(id, s)| { let mut m = s.upgrade(); m.sanitize(); (id, m) }).collect();
            return Ok(FullSnapshot {
                snapshot_seqno: v4.snapshot_seqno,
                payloads: v4.payloads,
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
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V1 {
            // V1: pre-ANN SemanticIndex + 14-field MemoryState.
            eprintln!("[chitta-field] migrating v1 snapshot → v6 (ANN index + MemoryStatus + EpistemicStatus)");
            let r = BufReader::new(&bytes[8..]);
            let v1: LegacyFullSnapshotV1 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut semantic_idx = SemanticIndex::new();
            for (mem_id, emb) in v1.semantic_idx.embeddings {
                semantic_idx.upsert(mem_id, emb);
            }
            let states = v1.states.into_iter().map(|(id, s)| (id, s.upgrade())).collect::<HashMap<_, _>>();
            return Ok(FullSnapshot {
                snapshot_seqno: v1.snapshot_seqno,
                payloads: v1.payloads,
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
            });
        }

        Err(FieldError::Manifest(format!("unknown snapshot magic: {:#x}", magic)))
    }
}

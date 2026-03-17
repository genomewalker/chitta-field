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
use crate::organ::triplet::TripletStore;
use crate::payload::MemoryPayload;
use crate::state::MemoryState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Magic for snapshots written before the ANN SemanticIndex rewrite (v1.0.3 and earlier).
const FULL_SNAPSHOT_MAGIC_V1: u64 = 0xF011_5741_7E00_0003;
/// Magic for v1.0.4 snapshots (post-ANN, pre-CTM features).
const FULL_SNAPSHOT_MAGIC_V4: u64 = 0xF011_5741_7E00_0004;
/// Magic for current snapshots (v1.0.5+) with coactivation_stats + RetrievalHistory.
const FULL_SNAPSHOT_MAGIC: u64 = 0xF011_5741_7E00_0005;

/// SemanticIndex as it existed before the ANN rewrite (two fields only).
/// Used solely for migrating v1 snapshots on first load.
#[derive(Serialize, Deserialize)]
struct LegacySemanticIndex {
    embeddings: std::collections::HashMap<MemoryId, Vec<f32>>,
    deleted: std::collections::HashSet<MemoryId>,
}

/// FullSnapshot layout for v1 (pre-ANN) snapshots. Identical to FullSnapshot
/// except it uses LegacySemanticIndex so bincode can deserialize the old bytes.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshot {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, crate::payload::MemoryPayload>,
    pub states: HashMap<MemoryId, crate::state::MemoryState>,
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
    #[serde(default)]
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
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

    /// Read only the magic and snapshot_seqno without deserializing the full snapshot.
    /// Accepts both v1 and v2 magic values.
    pub fn peek_seqno(path: &Path) -> Result<u64> {
        let f = std::fs::File::open(path)?;
        let mut r = BufReader::new(f);
        let mut buf = [0u8; 16];
        r.read_exact(&mut buf)?;
        let magic = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        if magic != FULL_SNAPSHOT_MAGIC && magic != FULL_SNAPSHOT_MAGIC_V4 && magic != FULL_SNAPSHOT_MAGIC_V1 {
            return Err(FieldError::Manifest("invalid full snapshot magic".to_string()));
        }
        Ok(u64::from_le_bytes(buf[8..16].try_into().unwrap()))
    }

    /// Load a full snapshot from disk. Transparently migrates v1 (pre-ANN) snapshots.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        if bytes.len() < 8 {
            return Err(FieldError::Manifest("snapshot too short".to_string()));
        }
        let magic = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        if magic == FULL_SNAPSHOT_MAGIC {
            // Current format (v5): deserialize directly.
            let r = BufReader::new(&bytes[8..]);
            return bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()));
        }
        if magic == FULL_SNAPSHOT_MAGIC_V4 {
            // v4 snapshots have SemanticIndex (ANN) but no coactivation_stats.
            // Since coactivation_stats has #[serde(default)], deserialize directly.
            eprintln!("[chitta-field] migrating v4 snapshot to v5 format (coactivation_stats)");
            let r = BufReader::new(&bytes[8..]);
            return bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()));
        }
        if magic == FULL_SNAPSHOT_MAGIC_V1 {
            // Legacy format: deserialize with old SemanticIndex shape, then migrate.
            eprintln!("[chitta-field] migrating v1 snapshot to v2 format (ANN index)");
            let r = BufReader::new(&bytes[8..]);
            let legacy: LegacyFullSnapshot = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            // Rebuild SemanticIndex from legacy embeddings.
            let mut semantic_idx = SemanticIndex::new();
            for (mem_id, emb) in legacy.semantic_idx.embeddings {
                semantic_idx.upsert(mem_id, emb);
            }
            return Ok(FullSnapshot {
                snapshot_seqno: legacy.snapshot_seqno,
                payloads: legacy.payloads,
                states: legacy.states,
                assoc_edges: legacy.assoc_edges,
                artifacts: legacy.artifacts,
                artifact_paths: legacy.artifact_paths,
                time_idx: legacy.time_idx,
                keyword_idx: legacy.keyword_idx,
                artifact_idx: legacy.artifact_idx,
                triplet_store: legacy.triplet_store,
                symbol_idx: legacy.symbol_idx,
                call_graph: legacy.call_graph,
                code_files: legacy.code_files,
                semantic_idx,
                coactivation_stats: HashMap::new(),
            });
        }
        Err(FieldError::Manifest(format!("unknown snapshot magic: {:#x}", magic)))
    }
}

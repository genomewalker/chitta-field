use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use parking_lot::RwLock;
use crate::error::Result;
use crate::ids::{MemoryId, ArtifactId, MemoryIdAllocator, ArtifactIdAllocator, TripletIdAllocator};
use crate::log::OpLog;
use crate::manifest::Manifest;
use crate::ops::{Op, EdgeType};
use crate::payload::MemoryPayload;
use crate::state::MemoryState;
use crate::hnsw::SemanticIndex;
use crate::organ::temporal::{TemporalIndex, TemporalEntry};
use crate::organ::artifact::ArtifactIndex;
use crate::organ::keyword::KeywordIndex;
use crate::organ::triplet::TripletStore;

/// A single directed association edge stored in memory.
#[derive(Debug, Clone)]
pub struct AssocEdge {
    pub dst: MemoryId,
    pub edge_type: EdgeType,
    pub weight: f32,
}

pub struct ChittaField {
    pub(crate) data_dir: PathBuf,
    pub(crate) lock_dir: PathBuf,
    pub(crate) manifest: RwLock<Manifest>,
    pub(crate) log: RwLock<OpLog>,
    pub(crate) id_alloc: Arc<MemoryIdAllocator>,
    pub(crate) artifact_id_alloc: Arc<ArtifactIdAllocator>,
    pub(crate) payloads: RwLock<HashMap<MemoryId, MemoryPayload>>,
    pub(crate) states: RwLock<HashMap<MemoryId, MemoryState>>,
    pub(crate) assoc_edges: RwLock<HashMap<MemoryId, Vec<AssocEdge>>>,
    pub(crate) artifacts: RwLock<HashMap<String, ArtifactId>>,
    /// Reverse map: artifact_id -> normalized_path, for wiring artifact index during PutPayload.
    pub(crate) artifact_paths: RwLock<HashMap<ArtifactId, String>>,
    pub(crate) semantic_idx: RwLock<SemanticIndex>,
    pub(crate) time_idx: RwLock<TemporalIndex>,
    pub(crate) artifact_idx: RwLock<ArtifactIndex>,
    pub(crate) keyword_idx: RwLock<KeywordIndex>,
    pub(crate) triplet_store: RwLock<TripletStore>,
    pub(crate) triplet_id_alloc: Arc<TripletIdAllocator>,
}

impl ChittaField {
    pub fn open(data_dir: PathBuf, lock_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&data_dir)?;
        std::fs::create_dir_all(&lock_dir)?;
        std::fs::create_dir_all(data_dir.join("segments"))?;

        // Load or create manifest.
        let manifest = Manifest::load(&data_dir)?.unwrap_or_else(|| {
            Manifest::new_empty("bge-base-en-v1.5", 768)
        });

        // Determine replay start: one past the last checkpointed payload seqno.
        let replay_from = manifest
            .checkpoints
            .payload
            .as_ref()
            .map(|c| c.max_seqno + 1)
            .unwrap_or(1);

        // Open log positioned for new appends at last_seqno + 1.
        let log = OpLog::open(&data_dir, manifest.last_seqno + 1)?;

        let id_alloc = Arc::new(MemoryIdAllocator::new(manifest.next_memory_id.max(1)));
        let artifact_id_alloc = Arc::new(ArtifactIdAllocator::new(manifest.next_artifact_id.max(1)));

        let mut payloads: HashMap<MemoryId, MemoryPayload> = HashMap::new();
        let mut states: HashMap<MemoryId, MemoryState> = HashMap::new();
        let mut assoc_edges: HashMap<MemoryId, Vec<AssocEdge>> = HashMap::new();
        let mut artifacts: HashMap<String, ArtifactId> = HashMap::new();
        let mut artifact_paths: HashMap<ArtifactId, String> = HashMap::new();
        let mut semantic_idx = SemanticIndex::new();
        let mut time_idx = TemporalIndex::new();
        let mut artifact_idx = ArtifactIndex::new();
        let mut keyword_idx = KeywordIndex::new();
        let mut triplet_store = TripletStore::new();

        // Replay ops from the log into in-memory projections.
        log.replay(replay_from, |_seqno, op| {
            apply_op(
                op,
                &mut payloads,
                &mut states,
                &mut assoc_edges,
                &mut artifacts,
                &mut artifact_paths,
                &mut semantic_idx,
                &mut time_idx,
                &mut artifact_idx,
                &mut keyword_idx,
                &mut triplet_store,
            );
            Ok(())
        })?;

        // Restore the triplet ID allocator from the replayed store state.
        let triplet_id_alloc = Arc::new(TripletIdAllocator::new(triplet_store.next_id()));

        Ok(Self {
            data_dir,
            lock_dir,
            manifest: RwLock::new(manifest),
            log: RwLock::new(log),
            id_alloc,
            artifact_id_alloc,
            payloads: RwLock::new(payloads),
            states: RwLock::new(states),
            assoc_edges: RwLock::new(assoc_edges),
            artifacts: RwLock::new(artifacts),
            artifact_paths: RwLock::new(artifact_paths),
            semantic_idx: RwLock::new(semantic_idx),
            time_idx: RwLock::new(time_idx),
            artifact_idx: RwLock::new(artifact_idx),
            keyword_idx: RwLock::new(keyword_idx),
            triplet_store: RwLock::new(triplet_store),
            triplet_id_alloc,
        })
    }
}

/// Apply a single Op to the in-memory projections, including all indexes.
pub(crate) fn apply_op(
    op: Op,
    payloads: &mut HashMap<MemoryId, MemoryPayload>,
    states: &mut HashMap<MemoryId, MemoryState>,
    assoc_edges: &mut HashMap<MemoryId, Vec<AssocEdge>>,
    artifacts: &mut HashMap<String, ArtifactId>,
    artifact_paths: &mut HashMap<ArtifactId, String>,
    semantic_idx: &mut SemanticIndex,
    time_idx: &mut TemporalIndex,
    artifact_idx: &mut ArtifactIndex,
    keyword_idx: &mut KeywordIndex,
    triplet_store: &mut TripletStore,
) {
    match op {
        Op::PutPayload(put) => {
            let memory_id = put.memory_id;
            let chunk_hash = put.chunk_hash;
            let created_at_ms = put.created_at_ms;
            let authored_at_ms = put.authored_at_ms;
            let version = put.version;
            let kind = put.kind.clone();
            let realm = put.realm.clone();
            let embedding = put.embedding.clone();
            let artifact_refs = put.artifact_refs.clone();
            let content_str = String::from_utf8(put.content.clone()).unwrap_or_default();

            // Insert or replace state if this is a newer version.
            let state = states.entry(memory_id).or_insert_with(|| {
                MemoryState::new(memory_id, chunk_hash, created_at_ms)
            });
            state.current_version = version;
            state.current_chunk_hash = chunk_hash;
            let strength = state.strength;

            payloads.insert(memory_id, MemoryPayload::from(put));
            semantic_idx.upsert(memory_id, embedding);
            keyword_idx.index(memory_id, &content_str);

            // Update temporal index.
            time_idx.upsert(TemporalEntry {
                memory_id,
                ts_ms: authored_at_ms,
                kind,
                realm,
                strength,
            });

            // Update artifact index for each artifact ref, using the reverse path map.
            for art_ref in &artifact_refs {
                if let Some(path) = artifact_paths.get(&art_ref.artifact_id) {
                    artifact_idx.associate(memory_id, art_ref.artifact_id, path, strength);
                }
            }
        }
        Op::UpdateState(delta) => {
            let memory_id = delta.memory_id;
            // Use a wall-clock substitute of 0 during replay; callers with real
            // time-awareness should supply now_ms via a higher-level interface.
            if let Some(state) = states.get_mut(&memory_id) {
                state.apply_delta(&delta, 0);
            }
        }
        Op::DeleteMemory(del) => {
            let memory_id = del.memory_id;
            if let Some(state) = states.get_mut(&memory_id) {
                state.deleted = true;
            }
            semantic_idx.remove(memory_id);
            keyword_idx.remove(memory_id);
            // Remove from temporal index using the authored_at_ms stored in the payload.
            if let Some(payload) = payloads.get(&memory_id) {
                time_idx.remove(memory_id, payload.authored_at_ms);
            }
            artifact_idx.remove_memory(memory_id);
        }
        Op::AddAssocEdge(edge_op) => {
            let entry = assoc_edges.entry(edge_op.src).or_insert_with(Vec::new);
            entry.push(AssocEdge {
                dst: edge_op.dst,
                edge_type: edge_op.edge_type,
                weight: edge_op.weight,
            });
        }
        Op::UpsertArtifact(art_op) => {
            // Maintain both forward (path->id) and reverse (id->path) maps.
            artifacts.entry(art_op.normalized_path.clone()).or_insert(art_op.artifact_id);
            artifact_paths.entry(art_op.artifact_id).or_insert(art_op.normalized_path);
        }
        Op::AddTriplet(t) => {
            triplet_store.replay_add(
                t.triplet_id,
                t.subject,
                t.predicate,
                t.object,
                t.weight,
                t.valid_from_ms,
                t.source_memory_id,
                t.source_file,
            );
        }
        Op::InvalidateTriplet(inv) => {
            triplet_store.invalidate(inv.triplet_id, inv.invalidated_at_ms);
        }
    }
}

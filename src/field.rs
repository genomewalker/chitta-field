use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use parking_lot::RwLock;
use crate::error::Result;
use crate::ids::{MemoryId, ArtifactId, InstanceId, MemoryIdAllocator, ArtifactIdAllocator, TripletIdAllocator, new_instance_id};
use crate::log::OpLog;
use crate::ops::{Op, EdgeType};
use crate::payload::MemoryPayload;
use crate::state::MemoryState;
use crate::hnsw::SemanticIndex;
use crate::organ::temporal::{TemporalIndex, TemporalEntry};
use crate::organ::artifact::ArtifactIndex;
use crate::organ::keyword::KeywordIndex;
use crate::organ::triplet::TripletStore;
use crate::learner::LearnerSet;

/// A single directed association edge stored in memory.
#[derive(Debug, Clone)]
pub struct AssocEdge {
    pub dst: MemoryId,
    pub edge_type: EdgeType,
    pub weight: f32,
}

pub struct ChittaField {
    pub(crate) data_dir: PathBuf,
    pub(crate) instance_id: InstanceId,
    pub(crate) log: RwLock<OpLog>,
    pub(crate) id_alloc: Arc<MemoryIdAllocator>,
    pub(crate) artifact_id_alloc: Arc<ArtifactIdAllocator>,
    pub(crate) payloads: RwLock<HashMap<MemoryId, MemoryPayload>>,
    pub(crate) states: RwLock<HashMap<MemoryId, MemoryState>>,
    pub(crate) assoc_edges: RwLock<HashMap<MemoryId, Vec<AssocEdge>>>,
    pub(crate) artifacts: RwLock<HashMap<String, ArtifactId>>,
    pub(crate) artifact_paths: RwLock<HashMap<ArtifactId, String>>,
    pub(crate) semantic_idx: RwLock<SemanticIndex>,
    pub(crate) time_idx: RwLock<TemporalIndex>,
    pub(crate) artifact_idx: RwLock<ArtifactIndex>,
    pub(crate) keyword_idx: RwLock<KeywordIndex>,
    pub(crate) triplet_store: RwLock<TripletStore>,
    pub(crate) triplet_id_alloc: Arc<TripletIdAllocator>,
    pub(crate) learners: RwLock<LearnerSet>,
}

impl Drop for ChittaField {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

impl ChittaField {
    /// Flush the write buffer to the OS.
    pub fn flush(&self) -> Result<()> {
        self.log.write().flush_buf()
    }

    pub fn open(data_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&data_dir)?;
        std::fs::create_dir_all(data_dir.join("segments"))?;

        // Each open() generates a fresh InstanceId — no coordination needed.
        let instance_id = new_instance_id();

        // Open this instance's write log.
        let log = OpLog::open(&data_dir, instance_id, 1)?;

        // Allocators are partitioned by instance_id — no collision with other instances.
        let id_alloc = Arc::new(MemoryIdAllocator::with_instance(instance_id));
        let artifact_id_alloc = Arc::new(ArtifactIdAllocator::with_instance(instance_id));

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

        // Replay ALL segment files to rebuild in-memory state.
        log.replay(0, |_seqno, op| {
            apply_op(op, &mut payloads, &mut states, &mut assoc_edges,
                     &mut artifacts, &mut artifact_paths, &mut semantic_idx,
                     &mut time_idx, &mut artifact_idx, &mut keyword_idx, &mut triplet_store);
            Ok(())
        })?;

        let triplet_id_alloc = Arc::new(TripletIdAllocator::new(triplet_store.next_id()));

        Ok(Self {
            data_dir,
            instance_id,
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
            learners: RwLock::new(LearnerSet::new()),
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

            let state = states.entry(memory_id).or_insert_with(|| {
                MemoryState::new(memory_id, chunk_hash, created_at_ms)
            });
            state.current_version = version;
            state.current_chunk_hash = chunk_hash;
            let strength = state.strength;

            payloads.insert(memory_id, MemoryPayload::from(put));
            semantic_idx.upsert(memory_id, embedding);
            keyword_idx.index(memory_id, &content_str);

            time_idx.upsert(TemporalEntry {
                memory_id,
                ts_ms: authored_at_ms,
                kind,
                realm,
                strength,
            });

            for art_ref in &artifact_refs {
                if let Some(path) = artifact_paths.get(&art_ref.artifact_id) {
                    artifact_idx.associate(memory_id, art_ref.artifact_id, path, strength);
                }
            }
        }
        Op::UpdateState(delta) => {
            let memory_id = delta.memory_id;
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

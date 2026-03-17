use crate::error::Result;
use crate::hnsw::SemanticIndex;
use crate::ids::{
    new_instance_id, ArtifactId, ArtifactIdAllocator, InstanceId, MemoryId, MemoryIdAllocator,
    TripletIdAllocator,
};
use crate::learner::LearnerSet;
use crate::log::OpLog;
use crate::ops::{EdgeType, Op};
use crate::organ::analytics::AnalyticsRegistry;
use crate::organ::artifact::ArtifactIndex;
use crate::organ::callgraph::CallGraph;
use crate::organ::codefile::CodeFileIndex;
use crate::organ::cortex::{CorticalIndex, SparseCode, SparseEncoder};
use crate::organ::keyword::KeywordIndex;
use crate::organ::lite_encoder::LiteEncoder;
use crate::organ::pq::ProductQuantizer;
use crate::organ::session::SessionRegistry;
use crate::organ::symbol::{SymbolEntry, SymbolIndex};
use crate::organ::task::TaskRegistry;
use crate::organ::temporal::{TemporalEntry, TemporalIndex};
use crate::organ::theme_organ::ThemeOrgan;
use crate::organ::transcript::TranscriptRegistry;
use crate::organ::triplet::TripletStore;
use crate::organ::user_model::UserModelRegistry;
use crate::payload::MemoryPayload;
use crate::snapshot::FullSnapshot;
use crate::state::MemoryState;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// A single directed association edge stored in memory.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AssocEdge {
    pub dst: MemoryId,
    pub edge_type: EdgeType,
    pub weight: f32,
}

#[derive(Default)]
pub(crate) struct PendingRecallEffects {
    pub strengthen: HashSet<MemoryId>,
    pub co_retrieval_pairs: HashMap<(MemoryId, MemoryId), f32>,
    pub proto_windows: Vec<Vec<MemoryId>>,
}

pub struct ChittaField {
    #[allow(dead_code)]
    pub(crate) data_dir: PathBuf,
    #[allow(dead_code)]
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
    pub(crate) symbol_idx: RwLock<SymbolIndex>,
    pub(crate) call_graph: RwLock<CallGraph>,
    pub(crate) code_files: RwLock<CodeFileIndex>,
    pub(crate) symbol_id_alloc: Arc<TripletIdAllocator>,
    pub(crate) code_file_id_alloc: Arc<TripletIdAllocator>,
    pub(crate) learners: RwLock<LearnerSet>,
    pub(crate) sparse_encoder: RwLock<SparseEncoder>,
    pub(crate) cortical_idx: RwLock<CorticalIndex>,
    pub(crate) event_id_alloc: Arc<AtomicU64>,
    pub(crate) session_registry: RwLock<SessionRegistry>,
    pub(crate) transcript_registry: RwLock<TranscriptRegistry>,
    pub(crate) task_registry: RwLock<TaskRegistry>,
    pub(crate) user_model_registry: RwLock<UserModelRegistry>,
    pub(crate) theme_organ: RwLock<ThemeOrgan>,
    pub(crate) analytics_registry: RwLock<AnalyticsRegistry>,
    pub(crate) lite_encoder: RwLock<Option<LiteEncoder>>,
    /// Byte offsets for each foreign segment file, used by sync_foreign().
    pub(crate) seen_offsets: RwLock<HashMap<PathBuf, u64>>,
    pub(crate) chunk_hash_idx: RwLock<HashMap<crate::ids::ChunkHash, MemoryId>>,
    pub(crate) realm_members: RwLock<HashMap<String, HashSet<MemoryId>>>,
    pub(crate) pending_recall: Mutex<PendingRecallEffects>,
}

impl Drop for ChittaField {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

impl ChittaField {
    /// Flush the write buffer to the OS.
    pub fn flush(&self) -> Result<()> {
        self.drain_pending_recall_effects()?;
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
        let mut symbol_idx = SymbolIndex::new();
        let mut call_graph = CallGraph::new();
        let mut code_files = CodeFileIndex::new();
        let mut cortical_idx = CorticalIndex::new();
        let mut session_registry = SessionRegistry::new();
        let mut transcript_registry = TranscriptRegistry::new();
        let mut task_registry = TaskRegistry::new();
        let mut user_model_registry = UserModelRegistry::new();
        let mut theme_organ = ThemeOrgan::new();
        let mut analytics_registry = AnalyticsRegistry::new();
        let mut chunk_hash_idx: HashMap<crate::ids::ChunkHash, MemoryId> = HashMap::new();

        // Find best cortical snapshot by peeking seqno (16-byte read per file), then load only that one.
        let mut snapshot_seqno: u64 = 0;
        let mut best_cortex_path: Option<std::path::PathBuf> = None;
        let mut stale_cortex_paths: Vec<std::path::PathBuf> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&data_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("cortex.") && name_str.ends_with(".snapshot") {
                    match CorticalIndex::peek_snapshot_seqno(&entry.path()) {
                        Ok(seqno) if seqno > snapshot_seqno => {
                            if let Some(prev) = best_cortex_path.replace(entry.path()) {
                                stale_cortex_paths.push(prev);
                            }
                            snapshot_seqno = seqno;
                        }
                        Ok(_) => stale_cortex_paths.push(entry.path()),
                        Err(e) => {
                            eprintln!(
                                "[chitta-field] skipping corrupt cortical snapshot {:?}: {}",
                                entry.path(),
                                e
                            );
                            stale_cortex_paths.push(entry.path());
                        }
                    }
                }
            }
        }
        if let Some(ref path) = best_cortex_path {
            match CorticalIndex::load_snapshot(path) {
                Ok((loaded, seqno)) => {
                    cortical_idx = loaded;
                    snapshot_seqno = seqno;
                    eprintln!(
                        "[chitta-field] loaded cortical snapshot seqno={} from {:?}",
                        seqno, path
                    );
                }
                Err(e) => eprintln!(
                    "[chitta-field] failed to load cortical snapshot {:?}: {}",
                    path, e
                ),
            }
        }
        for path in &stale_cortex_paths {
            let _ = std::fs::remove_file(path);
        }

        // Find best full snapshot by peeking seqno (16-byte read per file), then load only that one.
        let mut full_snapshot_seqno: u64 = 0;
        let mut best_full_path: Option<std::path::PathBuf> = None;
        let mut stale_full_paths: Vec<std::path::PathBuf> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&data_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("chitta.") && name_str.ends_with(".snapshot") {
                    match FullSnapshot::peek_seqno(&entry.path()) {
                        Ok(seqno) if seqno > full_snapshot_seqno => {
                            if let Some(prev) = best_full_path.replace(entry.path()) {
                                stale_full_paths.push(prev);
                            }
                            full_snapshot_seqno = seqno;
                        }
                        Ok(_) => stale_full_paths.push(entry.path()),
                        Err(e) => {
                            eprintln!(
                                "[chitta-field] skipping corrupt full snapshot {:?}: {}",
                                entry.path(),
                                e
                            );
                            stale_full_paths.push(entry.path());
                        }
                    }
                }
            }
        }
        if let Some(ref path) = best_full_path {
            match FullSnapshot::load(path) {
                Ok(snap) => {
                    full_snapshot_seqno = snap.snapshot_seqno;
                    payloads = snap.payloads;
                    states = snap.states;
                    assoc_edges = snap.assoc_edges;
                    artifacts = snap.artifacts;
                    artifact_paths = snap.artifact_paths;
                    time_idx = snap.time_idx;
                    keyword_idx = snap.keyword_idx;
                    artifact_idx = snap.artifact_idx;
                    triplet_store = snap.triplet_store;
                    symbol_idx = snap.symbol_idx;
                    call_graph = snap.call_graph;
                    code_files = snap.code_files;
                    semantic_idx = snap.semantic_idx;
                    eprintln!(
                        "[chitta-field] loaded full snapshot seqno={} from {:?}",
                        full_snapshot_seqno, path
                    );
                }
                Err(e) => eprintln!(
                    "[chitta-field] failed to load full snapshot {:?}: {}",
                    path, e
                ),
            }
        }
        for path in &stale_full_paths {
            let _ = std::fs::remove_file(path);
        }

        // Replay ALL segment files to rebuild in-memory state.
        // Skip ops already covered by the full snapshot or cortical snapshot.
        let mut replay_realm_members: HashMap<String, HashSet<MemoryId>> = HashMap::new();
        log.replay(0, |seqno, op| {
            if seqno <= full_snapshot_seqno {
                // This op is covered by the full snapshot.
                // Still apply cortical ops not covered by the cortical snapshot.
                match &op {
                    Op::UpdateSparseCode(_) | Op::TrainPQ(_) | Op::UpdateResidualPQ(_)
                        if seqno <= snapshot_seqno =>
                    {
                        return Ok(())
                    }
                    Op::UpdateSparseCode(_) | Op::TrainPQ(_) | Op::UpdateResidualPQ(_) => {
                        // fall through: apply to cortical index only
                    }
                    _ => return Ok(()), // covered by full snapshot
                }
            } else if seqno <= snapshot_seqno {
                // Covered by cortical snapshot but not by full snapshot (shouldn't normally happen,
                // but handle gracefully by skipping cortical ops).
                if matches!(
                    op,
                    Op::UpdateSparseCode(_) | Op::TrainPQ(_) | Op::UpdateResidualPQ(_)
                ) {
                    return Ok(());
                }
            }
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
                &mut symbol_idx,
                &mut call_graph,
                &mut code_files,
                &mut cortical_idx,
                &mut session_registry,
                &mut transcript_registry,
                &mut task_registry,
                &mut user_model_registry,
                &mut theme_organ,
                &mut analytics_registry,
                &mut chunk_hash_idx,
                &mut replay_realm_members,
            );
            Ok(())
        })?;

        semantic_idx.normalize_all();
        keyword_idx.rebuild_reverse_index();
        let realm_members = build_realm_members(&payloads, &states);

        // Fix temporal entries that have ts_ms=0 (stored before authored_at_ms default fix)
        {
            let zero_entries = time_idx.entries_with_ts(0);
            let fixed_count = zero_entries.len();
            for entry in zero_entries {
                if let Some(payload) = payloads.get(&entry.memory_id) {
                    let correct_ts = if payload.authored_at_ms != 0 {
                        payload.authored_at_ms
                    } else {
                        payload.created_at_ms
                    };
                    if correct_ts != 0 {
                        time_idx.remove(entry.memory_id, 0);
                        time_idx.upsert(TemporalEntry {
                            memory_id: entry.memory_id,
                            ts_ms: correct_ts,
                            kind: entry.kind.clone(),
                            realm: entry.realm.clone(),
                            strength: entry.strength,
                        });
                    }
                }
            }
            if fixed_count > 0 {
                eprintln!(
                    "[chitta-field] fixed {} temporal entries with ts_ms=0",
                    fixed_count
                );
            }
        }

        let triplet_id_alloc = Arc::new(TripletIdAllocator::new(triplet_store.next_id()));

        // Sync symbol and code-file id allocators from the loaded indexes.
        let max_symbol_id = symbol_idx.max_id().unwrap_or(0);
        let symbol_id_alloc = Arc::new(TripletIdAllocator::new(max_symbol_id + 1));
        let max_code_file_id = code_files.max_id().unwrap_or(0);
        let code_file_id_alloc = Arc::new(TripletIdAllocator::new(max_code_file_id + 1));

        let loaded_lite_encoder = Self::load_lite_encoder(&data_dir);
        let loaded_seen_offsets = Self::load_seen_offsets(&data_dir, instance_id);

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
            symbol_idx: RwLock::new(symbol_idx),
            call_graph: RwLock::new(call_graph),
            code_files: RwLock::new(code_files),
            symbol_id_alloc,
            code_file_id_alloc,
            learners: RwLock::new(LearnerSet::new()),
            sparse_encoder: RwLock::new(SparseEncoder::new()),
            cortical_idx: RwLock::new(cortical_idx),
            event_id_alloc: Arc::new(AtomicU64::new(1)),
            session_registry: RwLock::new(session_registry),
            transcript_registry: RwLock::new(transcript_registry),
            task_registry: RwLock::new(task_registry),
            user_model_registry: RwLock::new(user_model_registry),
            theme_organ: RwLock::new(theme_organ),
            analytics_registry: RwLock::new(analytics_registry),
            lite_encoder: RwLock::new(loaded_lite_encoder),
            seen_offsets: RwLock::new(loaded_seen_offsets),
            chunk_hash_idx: RwLock::new(chunk_hash_idx),
            realm_members: RwLock::new(realm_members),
            pending_recall: Mutex::new(PendingRecallEffects::default()),
        })
    }
}

impl ChittaField {
    /// Ingest new ops from all foreign-instance segment files.
    ///
    /// Scans `data_dir/segments/` for files not belonging to this instance, reads
    /// any bytes appended since the last call, and applies those ops to the in-memory
    /// state without writing to this instance's own WAL.  CRC mismatches or truncated
    /// reads at the tail of a file are treated as in-progress writes from a concurrent
    /// peer and are silently skipped.
    ///
    /// Returns the count of ops applied.
    pub fn sync_foreign(&self) -> crate::error::Result<usize> {
        use crate::log::{collect_foreign_segments, replay_from_offset};

        let foreign_segs = collect_foreign_segments(&self.data_dir, self.instance_id)?;
        let mut ops = Vec::new();

        {
            let mut seen = self.seen_offsets.write();
            for seg_path in &foreign_segs {
                let offset = *seen.get(seg_path).unwrap_or(&0);
                let new_offset = replay_from_offset(seg_path, offset, |_seqno, op| {
                    ops.push(op);
                    Ok(())
                })?;
                seen.insert(seg_path.clone(), new_offset);
            }
        }

        if ops.is_empty() {
            return Ok(0);
        }

        let count = ops.len();

        // Acquire all write locks and apply every foreign op in one pass,
        // reusing the same apply_op path used during startup replay.
        let mut payloads = self.payloads.write();
        let mut states = self.states.write();
        let mut assoc_edges = self.assoc_edges.write();
        let mut artifacts = self.artifacts.write();
        let mut artifact_paths = self.artifact_paths.write();
        let mut semantic_idx = self.semantic_idx.write();
        let mut time_idx = self.time_idx.write();
        let mut artifact_idx = self.artifact_idx.write();
        let mut keyword_idx = self.keyword_idx.write();
        let mut triplet_store = self.triplet_store.write();
        let mut symbol_idx = self.symbol_idx.write();
        let mut call_graph = self.call_graph.write();
        let mut code_files = self.code_files.write();
        let mut cortical_idx = self.cortical_idx.write();
        let mut session_reg = self.session_registry.write();
        let mut transcript_reg = self.transcript_registry.write();
        let mut task_reg = self.task_registry.write();
        let mut user_model_reg = self.user_model_registry.write();
        let mut theme_organ = self.theme_organ.write();
        let mut analytics_reg = self.analytics_registry.write();
        let mut chunk_hash_idx = self.chunk_hash_idx.write();
        let mut realm_members = self.realm_members.write();

        for op in ops {
            apply_op(
                op,
                &mut *payloads,
                &mut *states,
                &mut *assoc_edges,
                &mut *artifacts,
                &mut *artifact_paths,
                &mut *semantic_idx,
                &mut *time_idx,
                &mut *artifact_idx,
                &mut *keyword_idx,
                &mut *triplet_store,
                &mut *symbol_idx,
                &mut *call_graph,
                &mut *code_files,
                &mut *cortical_idx,
                &mut *session_reg,
                &mut *transcript_reg,
                &mut *task_reg,
                &mut *user_model_reg,
                &mut *theme_organ,
                &mut *analytics_reg,
                &mut *chunk_hash_idx,
                &mut *realm_members,
            );
        }

        if count > 0 {
            self.persist_seen_offsets();
        }

        Ok(count)
    }
}

impl ChittaField {
    fn seen_offsets_path(
        data_dir: &std::path::Path,
        instance_id: crate::ids::InstanceId,
    ) -> std::path::PathBuf {
        data_dir.join(format!("seen_offsets.{:08x}.json", instance_id))
    }

    fn load_seen_offsets(
        data_dir: &std::path::Path,
        instance_id: crate::ids::InstanceId,
    ) -> HashMap<PathBuf, u64> {
        let path = Self::seen_offsets_path(data_dir, instance_id);
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    fn persist_seen_offsets(&self) {
        let path = Self::seen_offsets_path(&self.data_dir, self.instance_id);
        let offsets = self.seen_offsets.read();
        if let Ok(json) = serde_json::to_string(&*offsets) {
            let tmp = path.with_extension("tmp");
            if std::fs::write(&tmp, &json).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
        }
    }

    fn load_lite_encoder(data_dir: &std::path::Path) -> Option<LiteEncoder> {
        let path = data_dir.join("lite_encoder.bin");
        if !path.exists() {
            return None;
        }
        match std::fs::read(&path) {
            Ok(bytes) => match LiteEncoder::from_bytes(&bytes) {
                Ok(enc) => {
                    eprintln!(
                        "[chitta-field] loaded lite encoder ({} vocab, {} examples)",
                        enc.vocab.len(),
                        enc.training_examples
                    );
                    Some(enc)
                }
                Err(e) => {
                    eprintln!("[chitta-field] lite encoder load failed: {}", e);
                    None
                }
            },
            Err(_) => None,
        }
    }

    /// Train the lite encoder from all memories with sparse codes.
    /// Returns the number of training examples used.
    pub fn train_lite_encoder(&self) -> Result<usize> {
        let payloads = self.payloads.read();
        let cortical_idx = self.cortical_idx.read();

        let mut examples: Vec<(String, SparseCode)> = Vec::new();
        for (mem_id, payload) in payloads.iter() {
            if let Some(code) = cortical_idx.mem_codes.get(mem_id) {
                if !code.is_empty() {
                    let content = String::from_utf8_lossy(&payload.content).into_owned();
                    if !content.trim().is_empty() {
                        examples.push((content, code.clone()));
                    }
                }
            }
        }

        let count = examples.len();
        if count == 0 {
            return Ok(0);
        }

        let encoder = LiteEncoder::train(&examples);
        *self.lite_encoder.write() = Some(encoder);
        Ok(count)
    }

    /// Encode text via lite encoder. Returns None if not trained or no words match vocab.
    pub fn encode_lite(&self, text: &str) -> Option<SparseCode> {
        self.lite_encoder.read().as_ref()?.encode(text)
    }

    /// Save lite encoder to <data_dir>/lite_encoder.bin
    pub fn save_lite_encoder(&self) -> Result<()> {
        let guard = self.lite_encoder.read();
        let enc = guard.as_ref().ok_or_else(|| {
            crate::error::FieldError::Manifest("lite encoder not trained".to_string())
        })?;
        let bytes = enc.to_bytes();
        let path = self.data_dir.join("lite_encoder.bin");
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    /// Check if the lite encoder is trained and ready.
    pub fn lite_encoder_ready(&self) -> bool {
        self.lite_encoder.read().is_some()
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
    symbol_idx: &mut SymbolIndex,
    call_graph: &mut CallGraph,
    code_files: &mut CodeFileIndex,
    cortical_idx: &mut CorticalIndex,
    session_registry: &mut SessionRegistry,
    transcript_registry: &mut TranscriptRegistry,
    task_registry: &mut TaskRegistry,
    user_model_registry: &mut UserModelRegistry,
    theme_organ: &mut ThemeOrgan,
    analytics_registry: &mut AnalyticsRegistry,
    chunk_hash_idx: &mut HashMap<crate::ids::ChunkHash, MemoryId>,
    realm_members: &mut HashMap<String, HashSet<MemoryId>>,
) {
    match op {
        Op::PutPayload(put) => {
            let memory_id = put.memory_id;
            let chunk_hash = put.chunk_hash;
            let created_at_ms = put.created_at_ms;
            let authored_at_ms = if put.authored_at_ms == 0 {
                created_at_ms
            } else {
                put.authored_at_ms
            };
            let version = put.version;
            let kind = put.kind.clone();
            let realm = put.realm.clone();
            let embedding = put.embedding.clone();
            let artifact_refs = put.artifact_refs.clone();
            let content_str = String::from_utf8(put.content.clone()).unwrap_or_default();

            let state = states
                .entry(memory_id)
                .or_insert_with(|| MemoryState::new(memory_id, chunk_hash, created_at_ms));
            state.current_version = version;
            state.current_chunk_hash = chunk_hash;
            let strength = state.strength;

            payloads.insert(memory_id, MemoryPayload::from(put));
            chunk_hash_idx.entry(chunk_hash).or_insert(memory_id);
            semantic_idx.upsert(memory_id, embedding);
            keyword_idx.index(memory_id, &content_str);
            realm_members
                .entry(realm.clone())
                .or_default()
                .insert(memory_id);

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
                let remove_realm = if let Some(ids) = realm_members.get_mut(&payload.realm) {
                    ids.remove(&memory_id);
                    ids.is_empty()
                } else {
                    false
                };
                if remove_realm {
                    realm_members.remove(&payload.realm);
                }
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
            artifacts
                .entry(art_op.normalized_path.clone())
                .or_insert(art_op.artifact_id);
            artifact_paths
                .entry(art_op.artifact_id)
                .or_insert(art_op.normalized_path);
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
        Op::UpsertSymbol(s) => {
            let entry = SymbolEntry {
                id: s.symbol_id,
                kind: s.kind,
                name: s.name,
                signature: s.signature,
                file_path: s.file_path,
                line_start: s.line_start,
                line_end: s.line_end,
                repo_id: s.repo_id,
                embedding: s.embedding,
                description: s.description,
                memory_id: s.memory_id,
            };
            symbol_idx.upsert(entry);
        }
        Op::RemoveSymbol(r) => {
            symbol_idx.remove(r.symbol_id);
            call_graph.remove_symbol(r.symbol_id);
        }
        Op::AddSymCallEdge(e) => {
            call_graph.add_edge(e.caller_id, e.callee_id);
        }
        Op::RemoveSymCallEdge(e) => {
            // Edges are stored in the bidirectional maps; remove individually.
            let callees = call_graph.get_callees(e.caller_id);
            if callees.contains(&e.callee_id) {
                // Reconstruct by removing and re-adding remaining edges for caller.
                let remaining: Vec<u64> =
                    callees.into_iter().filter(|&c| c != e.callee_id).collect();
                call_graph.remove_symbol(e.caller_id);
                for callee in remaining {
                    call_graph.add_edge(e.caller_id, callee);
                }
            }
        }
        Op::UpsertCodeFile(f) => {
            code_files.upsert(&f.path, &f.project, f.mtime, || f.file_id);
        }
        Op::UpdateSparseCode(op) => {
            let code = SparseCode {
                feature_ids: op.feature_ids.clone(),
                activations: op.activations.clone(),
            };
            let strength = states.get(&op.memory_id).map(|s| s.strength).unwrap_or(0.5);
            let (kind, ts_ms) = payloads
                .get(&op.memory_id)
                .map(|p| (p.kind.as_str(), p.authored_at_ms))
                .unwrap_or(("unknown", op.ts_ms));
            cortical_idx.index(op.memory_id, &code, strength, ts_ms, kind);
        }
        Op::DemoteMemory(d) => {
            if let Some(state) = states.get_mut(&d.memory_id) {
                state.tier = d.new_tier;
            }
        }
        Op::TrainPQ(t) => {
            if let Ok(pq) = bincode::deserialize::<ProductQuantizer>(&t.codebook_bytes) {
                cortical_idx.set_pq(pq);
            }
        }
        Op::UpdateResidualPQ(u) => {
            if u.pq_bytes.len() == 32 {
                let mut codes = [0u8; 32];
                codes.copy_from_slice(&u.pq_bytes);
                cortical_idx.index_pq(u.memory_id, codes);
            }
        }
        Op::SessionEvent(ev) => match ev.kind.as_str() {
            "register" => {
                let kind = serde_json::from_slice::<serde_json::Value>(&ev.payload_json)
                    .ok()
                    .and_then(|v| {
                        v.get("kind")
                            .and_then(|k| k.as_str())
                            .map(|s| s.to_string())
                    })
                    .unwrap_or_default();
                session_registry.register(ev.session_id, kind, ev.realm, ev.ts_ms);
            }
            "heartbeat" => session_registry.heartbeat(&ev.session_id, ev.ts_ms),
            "deregister" => session_registry.deregister(&ev.session_id),
            _ => {}
        },
        Op::TranscriptEvent(ev) => {
            // Always store the raw payload for cf_get_latest_event("transcript", kind, session_id)
            let payload_str = String::from_utf8(ev.payload_json.clone()).unwrap_or_default();
            transcript_registry.set_session_event(&ev.session_id, &ev.kind, payload_str);

            match ev.kind.as_str() {
                "register" => {
                    let payload =
                        serde_json::from_slice::<serde_json::Value>(&ev.payload_json).ok();
                    let transcript_id = payload
                        .as_ref()
                        .and_then(|v| {
                            v.get("transcript_id")
                                .and_then(|k| k.as_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_default();
                    if !transcript_id.is_empty() {
                        transcript_registry.register(transcript_id, ev.session_id);
                    }
                }
                "update_progress" => {
                    let payload =
                        serde_json::from_slice::<serde_json::Value>(&ev.payload_json).ok();
                    let transcript_id = payload
                        .as_ref()
                        .and_then(|v| {
                            v.get("transcript_id")
                                .and_then(|k| k.as_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_default();
                    let pct = payload
                        .as_ref()
                        .and_then(|v| v.get("progress_pct").and_then(|k| k.as_f64()))
                        .unwrap_or(0.0) as f32;
                    if !transcript_id.is_empty() {
                        transcript_registry.update_progress(&transcript_id, pct);
                    }
                }
                "add_turn" => {
                    let payload =
                        serde_json::from_slice::<serde_json::Value>(&ev.payload_json).ok();
                    let transcript_id = payload
                        .as_ref()
                        .and_then(|v| {
                            v.get("transcript_id")
                                .and_then(|k| k.as_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_default();
                    let role = payload
                        .as_ref()
                        .and_then(|v| {
                            v.get("role")
                                .and_then(|k| k.as_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_default();
                    let content = payload
                        .as_ref()
                        .and_then(|v| {
                            v.get("content")
                                .and_then(|k| k.as_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_default();
                    if !transcript_id.is_empty() {
                        transcript_registry.add_turn(&transcript_id, role, content, ev.ts_ms);
                    }
                }
                _ => {}
            }
        }
        Op::TaskEvent(ev) => {
            let payload_str = String::from_utf8(ev.payload_json).unwrap_or_default();
            match ev.kind.as_str() {
                "create" => {
                    task_registry.create(
                        ev.task_id,
                        ev.task_type,
                        payload_str,
                        ev.ts_ms,
                        ev.fencing_token,
                    );
                }
                "start" | "pause" | "resume" | "complete" | "fail" => {
                    task_registry.transition(&ev.task_id, &ev.kind, ev.ts_ms, ev.fencing_token);
                }
                _ => {}
            }
        }
        Op::UserModelEvent(ev) => {
            let payload_str = String::from_utf8(ev.payload_json).unwrap_or_default();
            match ev.kind.as_str() {
                "upsert" => {
                    user_model_registry.upsert(ev.entity_id, ev.entity_type, payload_str, ev.ts_ms);
                }
                "observe" | "progress" | "complete" => {
                    user_model_registry.observe(&ev.entity_id, ev.ts_ms);
                }
                _ => {}
            }
        }
        Op::ThemeEvent(ev) => {
            let payload_str = String::from_utf8(ev.payload_json).unwrap_or_default();
            match ev.kind.as_str() {
                "create" => {
                    let name = serde_json::from_str::<serde_json::Value>(&payload_str)
                        .ok()
                        .and_then(|v| {
                            v.get("name")
                                .and_then(|n| n.as_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_default();
                    theme_organ.create(ev.theme_id, name);
                }
                "update_centroid" => {
                    let centroid = serde_json::from_str::<serde_json::Value>(&payload_str)
                        .ok()
                        .and_then(|v| {
                            v.get("centroid_json")
                                .and_then(|c| c.as_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or(payload_str);
                    theme_organ.update_centroid(ev.theme_id, centroid);
                }
                "assign_member" => {
                    let memory_id = serde_json::from_str::<serde_json::Value>(&payload_str)
                        .ok()
                        .and_then(|v| v.get("memory_id").and_then(|m| m.as_u64()))
                        .unwrap_or(0);
                    if memory_id > 0 {
                        theme_organ.assign_member(ev.theme_id, memory_id);
                    }
                }
                "remove_member" => {
                    let memory_id = serde_json::from_str::<serde_json::Value>(&payload_str)
                        .ok()
                        .and_then(|v| v.get("memory_id").and_then(|m| m.as_u64()))
                        .unwrap_or(0);
                    if memory_id > 0 {
                        theme_organ.remove_member(ev.theme_id, memory_id);
                    }
                }
                _ => {}
            }
        }
        Op::AnalyticsEvent(ev) => {
            let payload_str = String::from_utf8(ev.payload_json).unwrap_or_default();
            analytics_registry.append(ev.kind, ev.session_id, payload_str, ev.ts_ms);
        }
        Op::ClearProject(cp) => {
            let removed_paths = code_files.remove_by_project(&cp.project);
            let removed_ids = symbol_idx.remove_by_file_paths(&removed_paths);
            for id in removed_ids {
                call_graph.remove_symbol(id);
            }
        }
        Op::UpdateSymbolDescription(usd) => {
            if let Some(sym) = symbol_idx.get_mut(usd.symbol_id) {
                sym.description = Some(usd.description);
            }
        }
        Op::UpdateMemoryContent(umc) => {
            let content_str = String::from_utf8(umc.content.clone()).unwrap_or_default();
            if let Some(payload) = payloads.get_mut(&umc.memory_id) {
                payload.content = umc.content;
                if !umc.embedding.is_empty() {
                    payload.embedding = umc.embedding.clone();
                }
            }
            if !umc.embedding.is_empty() {
                semantic_idx.upsert(umc.memory_id, umc.embedding);
            }
            keyword_idx.index(umc.memory_id, &content_str);
        }
    }
}

fn build_realm_members(
    payloads: &HashMap<MemoryId, MemoryPayload>,
    states: &HashMap<MemoryId, MemoryState>,
) -> HashMap<String, HashSet<MemoryId>> {
    let mut realm_members: HashMap<String, HashSet<MemoryId>> = HashMap::new();
    for (&memory_id, payload) in payloads {
        let not_deleted = states.get(&memory_id).map(|s| !s.deleted).unwrap_or(false);
        if !not_deleted {
            continue;
        }
        realm_members
            .entry(payload.realm.clone())
            .or_default()
            .insert(memory_id);
    }
    realm_members
}

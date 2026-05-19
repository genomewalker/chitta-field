use crate::error::{FieldError, Result};
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
use crate::organ::hopfield::HopfieldNetwork;
use crate::organ::keyword::KeywordIndex;
use crate::organ::lite_encoder::LiteEncoder;
use crate::organ::agent::AgentRegistry;
use crate::organ::constraint::ConstraintStore;
use crate::organ::msg::MsgRegistry;
use crate::organ::predictor::AccessPredictor;
use crate::organ::skill::SkillRegistry;
use crate::organ::trigger::TriggerStore;
use crate::organ::surprise::SurpriseStore;
use crate::organ::agent_protocol::AgentProtocolStore;
use crate::organ::wisdom_lineage::WisdomLineageStore;
use crate::organ::epistemic_debt::EpistemicDebtStore;
use crate::organ::integration::IntegrationKernel;
use crate::organ::surprise_learning::SurpriseLearningStore;
use crate::organ::wisdom_promotion::WisdomPromotionStore;
use crate::organ::intervention::InterventionStore;
use crate::organ::symbol_events::{SymbolEvent, SymbolEventKind, SymbolEventLog};
use crate::scoring::learned::LearnedScoringModel;
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
use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::sync::Arc;

/// A single directed association edge stored in memory.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AssocEdge {
    pub dst: MemoryId,
    pub edge_type: EdgeType,
    pub weight: f32,
}

/// Tracks how often two memories were co-retrieved and in how many
/// distinct query contexts. Used to weight Hebbian edge strengthening.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct CoActivationStats {
    /// Total number of co-retrieval events.
    pub sim_count: u32,
    /// Distinct context_hash values seen (capped at 64).
    pub recent_context_hashes: Vec<u64>,
    /// Number of distinct contexts (len of deduplicated recent_context_hashes).
    pub diversity_count: u16,
    pub last_seen_ms: i64,
}

impl CoActivationStats {
    const MAX_HASHES: usize = 64;

    pub fn record(&mut self, context_hash: u64, ts_ms: i64) {
        self.sim_count += 1;
        self.last_seen_ms = ts_ms;
        if !self.recent_context_hashes.contains(&context_hash) {
            if self.recent_context_hashes.len() >= Self::MAX_HASHES {
                self.recent_context_hashes.remove(0);
            }
            self.recent_context_hashes.push(context_hash);
        }
        self.diversity_count = self.recent_context_hashes.len() as u16;
    }

    /// Hebbian multiplier: sim_count * diversity_count, capped at 16.0
    pub fn hebbian_multiplier(&self) -> f32 {
        let raw = self.sim_count as f32 * self.diversity_count as f32;
        raw.min(16.0)
    }
}

/// Cap coactivation_stats to `max_per_memory` strongest pairs per memory.
/// Returns the number of entries removed.
pub(crate) fn prune_coactivation_stats(
    stats: &mut HashMap<(MemoryId, MemoryId), CoActivationStats>,
    max_per_memory: usize,
) -> usize {
    use std::collections::HashSet;
    let mut per_memory: HashMap<MemoryId, Vec<((MemoryId, MemoryId), u32)>> =
        HashMap::with_capacity(stats.len().min(65536));
    for (key, stat) in stats.iter() {
        per_memory.entry(key.0).or_default().push((*key, stat.sim_count));
        per_memory.entry(key.1).or_default().push((*key, stat.sim_count));
    }
    let mut to_remove: HashSet<(MemoryId, MemoryId)> = HashSet::new();
    for (_mem, mut pairs) in per_memory {
        if pairs.len() > max_per_memory {
            pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            for (key, _) in pairs.into_iter().skip(max_per_memory) {
                to_remove.insert(key);
            }
        }
    }
    let removed = to_remove.len();
    for key in &to_remove { stats.remove(key); }
    removed
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
    pub(crate) msg_registry: RwLock<MsgRegistry>,
    pub(crate) skill_registry: RwLock<SkillRegistry>,
    pub(crate) agent_registry: RwLock<AgentRegistry>,
    pub(crate) constraint_store: RwLock<ConstraintStore>,
    pub(crate) trigger_store: RwLock<TriggerStore>,
    pub(crate) predictor: RwLock<AccessPredictor>,
    pub(crate) surprise_store: RwLock<SurpriseStore>,
    pub(crate) epistemic_debt_store: RwLock<EpistemicDebtStore>,
    pub(crate) integration_kernel: RwLock<IntegrationKernel>,
    pub(crate) surprise_learning: RwLock<SurpriseLearningStore>,
    pub(crate) wisdom_promotion: RwLock<WisdomPromotionStore>,
    pub(crate) learned_scorer: RwLock<LearnedScoringModel>,
    pub(crate) intervention_store: RwLock<InterventionStore>,
    pub(crate) agent_protocol_store: RwLock<AgentProtocolStore>,
    pub(crate) wisdom_lineage_store: RwLock<WisdomLineageStore>,
    pub(crate) symbol_event_log: RwLock<SymbolEventLog>,
    pub(crate) lite_encoder: RwLock<Option<LiteEncoder>>,
    /// Byte offsets for each foreign segment file, used by sync_foreign().
    pub(crate) seen_offsets: RwLock<HashMap<PathBuf, u64>>,
    pub(crate) chunk_hash_idx: RwLock<HashMap<crate::ids::ChunkHash, MemoryId>>,
    pub(crate) realm_members: RwLock<HashMap<String, HashSet<MemoryId>>>,
    pub(crate) kind_members:  RwLock<HashMap<String, HashSet<MemoryId>>>,
    pub(crate) pending_embed_count: Arc<AtomicUsize>,
    pub(crate) last_compact_ms: Arc<std::sync::atomic::AtomicI64>,
    pub(crate) pending_recall: Mutex<PendingRecallEffects>,
    pub(crate) coactivation_stats: RwLock<HashMap<(MemoryId, MemoryId), CoActivationStats>>,
    /// Asymmetric Hopfield network for energy-based attractor recall. FEP §3.2.
    pub(crate) hopfield: RwLock<HopfieldNetwork>,
    pub(crate) filter_level: std::sync::Arc<std::sync::atomic::AtomicU8>,
    pub(crate) scoring_pipeline: RwLock<crate::scoring::ScoringPipeline>,
    pub(crate) realm_stats: RwLock<HashMap<String, crate::store::GroupStats>>,
    pub(crate) kind_stats:  RwLock<HashMap<String, crate::store::GroupStats>>,
    /// Ack/nack usage scores — persisted in FullSnapshot.ack_scores (v9+).
    pub(crate) ack_scores: RwLock<HashMap<MemoryId, i32>>,
    /// Soul REPL session namespaces — persisted to repl_sessions.json (not in snapshot).
    pub(crate) repl_sessions: RwLock<crate::repl_sessions::ReplSessionStore>,
    /// Hyperdimensional Computing index — O(n) Hamming recall, no floats.
    pub(crate) hdc_idx:    RwLock<crate::hdc::HdcStore>,
    pub(crate) event_tape:   RwLock<crate::organ::event_tape::EventTape>,
    pub(crate) cdawg:        RwLock<crate::organ::cdawg::CdawgOrgan>,
    pub(crate) episode_hdc:        RwLock<crate::hdc::EpisodeHdcStore>,
    pub(crate) refutation_ledger:  RwLock<crate::organ::refutation_ledger::RefutationLedger>,
    pub(crate) cec_policy_store:   RwLock<crate::organ::intervention_store::InterventionStore>,
    pub(crate) decision_tape:      RwLock<crate::organ::decision_tape::DecisionTape>,
    /// Ephemeral — rebuilt from refutation_ledger after each consolidation_pass.
    pub(crate) hypothesis_market:  RwLock<crate::organ::hypothesis_market::HypothesisMarket>,
    /// CEC Phase 11 — Turīya Monitor: read-only organ that watches CEC organ health.
    /// Serialized in snapshot (rolling 100-sample window persists across sessions).
    pub(crate) turiya_monitor:     RwLock<crate::organ::turiya_monitor::TuriyaMonitor>,
    /// Ephemeral — rebuilt from EventTape alongside CDAWG at load.
    pub(crate) fep_prior:          RwLock<crate::organ::fep_prior::FepPriorOrgan>,
    /// CEC Phase 12 — cumulative count of events tombstoned by temporal compression.
    /// Ephemeral (not in snapshot) — lifetime of this daemon process.
    pub(crate) tape_tombstoned:    std::sync::atomic::AtomicU64,
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

    /// Return the current chain tip hash (SHA256). Zero if only V1 data.
    pub fn chain_head(&self) -> crate::log::ChainHash {
        self.log.read().chain_head()
    }

    pub fn open(data_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&data_dir)?;
        std::fs::create_dir_all(data_dir.join("segments"))?;

        // Each open() generates a fresh InstanceId — no coordination needed.
        let instance_id = new_instance_id();

        // Open this instance's write log.
        let mut log = OpLog::open(&data_dir, instance_id, 1)?;

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
        let mut msg_registry = MsgRegistry::new();
        let mut skill_registry = SkillRegistry::new();
        let mut agent_registry = AgentRegistry::new();
        let mut constraint_store = ConstraintStore::new();
        let mut trigger_store = TriggerStore::new();
        let predictor = AccessPredictor::new();
        let mut surprise_store = SurpriseStore::new();
        let mut epistemic_debt_store = EpistemicDebtStore::new();
        let mut integration_kernel = IntegrationKernel::new();
        let mut surprise_learning = SurpriseLearningStore::new();
        let mut wisdom_promotion = WisdomPromotionStore::new();
        let mut learned_scorer = LearnedScoringModel::new("v5.14".to_string());
        let mut intervention_store = InterventionStore::new();
        let mut agent_protocol_store = AgentProtocolStore::new();
        let mut wisdom_lineage_store = WisdomLineageStore::new();
        let mut symbol_event_log = SymbolEventLog::new();
        let mut chunk_hash_idx: HashMap<crate::ids::ChunkHash, MemoryId> = HashMap::new();
        let mut snapshot_coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats> = HashMap::new();
        let mut snap_ack_scores: HashMap<MemoryId, i32> = HashMap::new();
        let mut snap_correction_states: HashMap<u64, crate::organ::triplet::CorrectionState> = HashMap::new();
        let mut snap_event_tape    = crate::organ::event_tape::EventTape::new();
        let mut snap_decision_tape = crate::organ::decision_tape::DecisionTape::new();

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
        // Keep the 1 most recent stale cortex snapshot as a safety net.
        // Cortex is a cache (rebuilt from segments), so 1 backup is enough.
        let mut stale_cortex_by_seqno: Vec<(u64, &std::path::PathBuf)> = stale_cortex_paths
            .iter()
            .filter_map(|p| CorticalIndex::peek_snapshot_seqno(p).ok().map(|s| (s, p)))
            .collect();
        stale_cortex_by_seqno.sort_by(|a, b| b.0.cmp(&a.0));
        for (_, path) in stale_cortex_by_seqno.iter().skip(1) {
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
        let had_full_snapshots = best_full_path.is_some() || !stale_full_paths.is_empty();
        let mut full_snapshot_loaded = false;
        // Try best snapshot first, then fall back to stale ones (sorted by seqno descending).
        let mut candidates: Vec<std::path::PathBuf> = Vec::new();
        if let Some(ref path) = best_full_path {
            candidates.push(path.clone());
        }
        // Sort stale paths by seqno descending so we try the most recent first.
        let mut stale_with_seqno: Vec<(u64, std::path::PathBuf)> = stale_full_paths
            .iter()
            .filter_map(|p| FullSnapshot::peek_seqno(p).ok().map(|s| (s, p.clone())))
            .collect();
        stale_with_seqno.sort_by(|a, b| b.0.cmp(&a.0));
        for (_, p) in stale_with_seqno {
            candidates.push(p);
        }
        for candidate in &candidates {
            match FullSnapshot::load(candidate) {
                Ok(mut snap) => {
                    full_snapshot_seqno = snap.snapshot_seqno;
                    // v11+: content in .pld sidecar; v10: content already in bincode (no-op).
                    FullSnapshot::load_payload_sidecar(&candidate.with_extension("pld"), &mut snap.payloads);
                    snap.triplet_store.load_supersession_sidecar(&candidate.with_extension("sup.json"));
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
                    snapshot_coactivation_stats = snap.coactivation_stats;
                    snap_ack_scores = snap.ack_scores;
                    snap_correction_states = snap.correction_states;
                    snap_event_tape    = snap.event_tape;
                    snap_decision_tape = snap.decision_tape;
                    eprintln!(
                        "[chitta-field] loaded full snapshot seqno={} ({} memories) from {:?}",
                        full_snapshot_seqno, payloads.len(), candidate
                    );
                    full_snapshot_loaded = true;
                    break;
                }
                Err(e) => eprintln!(
                    "[chitta-field] failed to load full snapshot {:?}: {}",
                    candidate, e
                ),
            }
        }
        if had_full_snapshots && !full_snapshot_loaded {
            return Err(FieldError::Manifest(
                "all full snapshots failed to load — refusing to start with empty store \
                 (this prevents data loss; fix the snapshot format or restore from backup)"
                    .to_string(),
            ));
        }
        // Only clean up stale snapshots if we successfully loaded one.
        // Keep the 1 most recent stale snapshot as a safety net against format bugs.
        if full_snapshot_loaded {
            let mut stale_by_seqno: Vec<(u64, &std::path::PathBuf)> = stale_full_paths
                .iter()
                .filter(|p| best_full_path.as_ref() != Some(p))
                .filter_map(|p| FullSnapshot::peek_seqno(p).ok().map(|s| (s, p)))
                .collect();
            stale_by_seqno.sort_by(|a, b| b.0.cmp(&a.0));
            // Skip the most recent stale (keep as backup), delete the rest + their sidecars.
            for (_, path) in stale_by_seqno.iter().skip(1) {
                let _ = std::fs::remove_file(path);
                for ext in &["emb", "bin", "hnsw", "pld", "snapshot.tmp"] {
                    let _ = std::fs::remove_file(path.with_extension(ext));
                }
                // delta.hnsw: with_extension replaces only last component, handle separately
                let delta = path.with_extension("delta.hnsw");
                let _ = std::fs::remove_file(&delta);
            }

            // Prune orphaned sidecars: files whose snapshot hash is not current best or backup.
            let mut live_hashes = std::collections::HashSet::new();
            let extract_hash = |p: &std::path::Path| -> Option<String> {
                p.file_name()?.to_str()?.splitn(3, '.').nth(1).map(String::from)
            };
            if let Some(ref p) = best_full_path {
                if let Some(h) = extract_hash(p) { live_hashes.insert(h); }
            }
            if let Some((_, p)) = stale_by_seqno.first() {
                if let Some(h) = extract_hash(p) { live_hashes.insert(h); }
            }
            if let Ok(entries) = std::fs::read_dir(&data_dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if !name.starts_with("chitta.") { continue; }
                    let parts: Vec<&str> = name.splitn(3, '.').collect();
                    if parts.len() < 3 { continue; }
                    let hash = parts[1];
                    if live_hashes.contains(hash) { continue; }
                    let ext = parts[2];
                    if matches!(ext, "emb" | "bin" | "hnsw" | "pld" | "delta.hnsw" | "snapshot.tmp") {
                        let removed = std::fs::remove_file(entry.path()).is_ok();
                        if removed {
                            eprintln!("[chitta-field] pruned orphaned sidecar: {}", name);
                        }
                    }
                }
            }
        }

        // Replay ALL segment files to rebuild in-memory state.
        // Skip ops already covered by the full snapshot or cortical snapshot.
        let mut replay_realm_members: HashMap<String, HashSet<MemoryId>> = HashMap::new();
        let mut replay_kind_members:  HashMap<String, HashSet<MemoryId>> = HashMap::new();
        let mut replay_coactivation_stats = snapshot_coactivation_stats;
        triplet_store.correction_states = snap_correction_states;
        // Load embedding + binary-code sidecars, then HNSW — all before WAL replay.
        if let Some(ref snap_path) = best_full_path {
            // .emb: flat binary embeddings (v10+). For v9 snapshots the sidecar won't exist
            // yet, so this is a no-op and embeddings remain populated from bincode.
            let emb_loaded = semantic_idx.load_embeddings_sidecar(&snap_path.with_extension("emb"));
            if !emb_loaded && semantic_idx.embeddings_count() == 0 {
                eprintln!("[chitta-field] WARNING: v10 snapshot but .emb sidecar missing — embeddings will be empty until backfill");
            }
            // .bin: binary codes sidecar — skip O(N×256) reconstruction in normalize_all.
            let _ = semantic_idx.load_binary_sidecar(&snap_path.with_extension("bin"));
            // .emb mmap: serve embeddings from mmap whenever sidecar exists — avoids 1GB+ heap
            // copy regardless of collection size.
            if emb_loaded {
                let _ = semantic_idx.activate_mmap_embeddings(&snap_path.with_extension("emb"));
            }
            // .hnsw + .delta.hnsw: load both tiers; backfill handles WAL-replay additions.
            let _ = semantic_idx.load_hnsw(&snap_path.with_extension("hnsw"));
            let _ = semantic_idx.load_delta_hnsw(&snap_path.with_extension("delta.hnsw"));
        }
        // Inhibit HNSW inserts during replay — binary Hamming takes over after normalize_all(),
        // so building the O(N log N) HNSW graph incrementally would waste time and RAM.
        semantic_idx.set_inhibit_hnsw(true);
        let mut max_replayed_seqno = full_snapshot_seqno;
        log.replay(0, |seqno, op| {
            if seqno > max_replayed_seqno { max_replayed_seqno = seqno; }
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
                &mut msg_registry,
                &mut skill_registry,
                &mut agent_registry,
                &mut constraint_store,
                &mut trigger_store,
                &mut surprise_store,
                &mut epistemic_debt_store,
                &mut integration_kernel,
                &mut surprise_learning,
                &mut wisdom_promotion,
                &mut learned_scorer,
                &mut intervention_store,
                &mut agent_protocol_store,
                &mut wisdom_lineage_store,
                &mut symbol_event_log,
                &mut chunk_hash_idx,
                &mut replay_realm_members,
                &mut replay_kind_members,
                &mut replay_coactivation_stats,
            );
            Ok(())
        })?;
        log.set_next_seqno(max_replayed_seqno + 1);
        semantic_idx.set_inhibit_hnsw(false);
        let purged_ids = semantic_idx.purge_wrong_dim();
        for id in &purged_ids {
            if let Some(state) = states.get_mut(id) {
                if !state.deleted {
                    state.embed_pending = true;
                }
            }
        }
        semantic_idx.normalize_all();
        keyword_idx.rebuild_reverse_index();

        // One-time migration: mark SSL memories (content with →) for gloss-baked re-embed.
        // Guard file prevents re-running after backfill completes.
        {
            let flag = data_dir.join("ssl_gloss_v1.migrated");
            if !flag.exists() {
                let arrow: &[u8] = b"\xe2\x86\x92"; // UTF-8 →
                let mut ssl_count = 0usize;
                for (id, payload) in &payloads {
                    if payload.content.windows(3).any(|w| w == arrow) {
                        if let Some(state) = states.get_mut(id) {
                            if !state.embed_pending && !state.deleted {
                                state.embed_pending = true;
                                ssl_count += 1;
                            }
                        }
                    }
                }
                if ssl_count > 0 {
                    eprintln!("[chitta-field] SSL gloss migration: marked {ssl_count} memories for re-embed with gloss");
                }
                let _ = std::fs::write(&flag, "done");
            }
        }

        let realm_members = build_realm_members(&payloads, &states);
        let kind_members  = build_kind_members(&payloads, &states);
        let init_pending = states.values().filter(|s| s.embed_pending && !s.deleted).count();

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
        let scoring_config = crate::scoring::config::ScoringConfig::load(&data_dir);
        let loaded_repl_sessions = crate::repl_sessions::ReplSessionStore::load(&data_dir);

        // Build HDC index from persisted payloads (skips deleted memories via states map).
        let mut hdc_store = crate::hdc::HdcStore::new();
        {
            let entries = payloads.iter()
                .filter(|(id, _)| states.get(id).map(|s| !s.deleted).unwrap_or(false))
                .map(|(id, p)| (*id, std::str::from_utf8(&p.content).unwrap_or(""), p.realm.as_str()));
            hdc_store.rebuild(entries);
        }

        // Build EventTape from snapshot, seed entity interner from triplets, synthesize
        // legacy events for existing memories, then rebuild CDAWG from the tape.
        // Use persisted EventTape from snapshot if available; otherwise synthesize from memories.
        let mut event_tape = if !snap_event_tape.events.is_empty() {
            snap_event_tape
        } else {
            let mut tape = crate::organ::event_tape::EventTape::new();
            let subjects: Vec<String> = triplet_store.all_subjects();
            tape.seed_from_triplets(subjects.iter().map(|s| s.as_str()));
            let mut sorted_payloads: Vec<_> = payloads.iter()
                .filter(|(id, _)| states.get(id).map(|s| !s.deleted).unwrap_or(false))
                .collect();
            sorted_payloads.sort_by_key(|(_, p)| p.authored_at_ms);
            for (_, p) in sorted_payloads {
                tape.synthesize_legacy(&p.realm, p.authored_at_ms);
            }
            tape
        };
        let mut cdawg = crate::organ::cdawg::CdawgOrgan::new();
        cdawg.rebuild_from_tape(&event_tape);
        let mut episode_hdc = crate::hdc::EpisodeHdcStore::new();
        episode_hdc.rebuild(&event_tape);
        let refutation_ledger = crate::organ::refutation_ledger::RefutationLedger::new();
        let cec_policy_store  = crate::organ::intervention_store::InterventionStore::new();
        let decision_tape     = snap_decision_tape;
        let hypothesis_market = crate::organ::hypothesis_market::HypothesisMarket::new();
        let turiya_monitor    = crate::organ::turiya_monitor::TuriyaMonitor::new();
        let fep_prior         = crate::organ::fep_prior::FepPriorOrgan::new();

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
            triplet_store: RwLock::new({
                let before = triplet_store.triplet_count();
                let purged = triplet_store.purge_invalidated();
                let deduped = triplet_store.dedup_entries();
                if purged > 0 || deduped > 0 {
                    eprintln!("[chitta-field] triplet migration on load: purged {} invalidated, deduped {} duplicates ({} → {})",
                        purged, deduped, before, triplet_store.triplet_count());
                }
                triplet_store
            }),
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
            msg_registry: RwLock::new(msg_registry),
            skill_registry: RwLock::new(skill_registry),
            agent_registry: RwLock::new(agent_registry),
            constraint_store: RwLock::new(constraint_store),
            trigger_store: RwLock::new(trigger_store),
            predictor: RwLock::new(predictor),
            surprise_store: RwLock::new(surprise_store),
            epistemic_debt_store: RwLock::new(epistemic_debt_store),
            integration_kernel: RwLock::new(integration_kernel),
            surprise_learning: RwLock::new(surprise_learning),
            wisdom_promotion: RwLock::new(wisdom_promotion),
            learned_scorer: RwLock::new(learned_scorer),
            intervention_store: RwLock::new(intervention_store),
            agent_protocol_store: RwLock::new(agent_protocol_store),
            wisdom_lineage_store: RwLock::new(wisdom_lineage_store),
            symbol_event_log: RwLock::new(symbol_event_log),
            lite_encoder: RwLock::new(loaded_lite_encoder),
            seen_offsets: RwLock::new(loaded_seen_offsets),
            chunk_hash_idx: RwLock::new(chunk_hash_idx),
            realm_members: RwLock::new(realm_members),
            kind_members:  RwLock::new(kind_members),
            pending_embed_count: Arc::new(AtomicUsize::new(init_pending)),
            last_compact_ms: Arc::new(std::sync::atomic::AtomicI64::new(0)),
            realm_stats: RwLock::new(HashMap::new()),
            kind_stats:  RwLock::new(HashMap::new()),
            ack_scores:  RwLock::new(snap_ack_scores),
            repl_sessions: RwLock::new(loaded_repl_sessions),
            pending_recall: Mutex::new(PendingRecallEffects::default()),
            coactivation_stats: RwLock::new({
                let mut cs = replay_coactivation_stats;
                let n = cs.len();
                // Prune to 100 pairs/memory to bound startup RAM on old snapshots.
                crate::field::prune_coactivation_stats(&mut cs, 20);
                if cs.len() < n {
                    eprintln!("[chitta-field] pruned {} coactivation pairs on load", n - cs.len());
                }
                cs
            }),
            hopfield: RwLock::new(HopfieldNetwork::new()),
            filter_level: std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0)),
            scoring_pipeline: RwLock::new(crate::scoring::ScoringPipeline::new(scoring_config)),
            hdc_idx:    RwLock::new(hdc_store),
            event_tape:   RwLock::new(event_tape),
            cdawg:        RwLock::new(cdawg),
            episode_hdc:        RwLock::new(episode_hdc),
            refutation_ledger:  RwLock::new(refutation_ledger),
            cec_policy_store:   RwLock::new(cec_policy_store),
            decision_tape:      RwLock::new(decision_tape),
            hypothesis_market:  RwLock::new(hypothesis_market),
            turiya_monitor:     RwLock::new(turiya_monitor),
            fep_prior:          RwLock::new(fep_prior),
            tape_tombstoned:    std::sync::atomic::AtomicU64::new(0),
        })
    }
}

impl ChittaField {
    pub fn set_filter_level(&self, level: crate::store::FilterLevel) {
        self.filter_level.store(level as u8, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn filter_level(&self) -> crate::store::FilterLevel {
        match self.filter_level.load(std::sync::atomic::Ordering::Relaxed) {
            1 => crate::store::FilterLevel::Signatures,
            2 => crate::store::FilterLevel::MinimalContext,
            _ => crate::store::FilterLevel::None,
        }
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
        let mut msg_reg = self.msg_registry.write();
        let mut skill_reg = self.skill_registry.write();
        let mut agent_reg = self.agent_registry.write();
        let mut constraint_reg = self.constraint_store.write();
        let mut trigger_reg = self.trigger_store.write();
        let mut surprise_reg = self.surprise_store.write();
        let mut epistemic_debt_reg = self.epistemic_debt_store.write();
        let mut integration_reg = self.integration_kernel.write();
        let mut surprise_learning_reg = self.surprise_learning.write();
        let mut wisdom_promotion_reg = self.wisdom_promotion.write();
        let mut learned_scorer_reg = self.learned_scorer.write();
        let mut intervention_store_reg = self.intervention_store.write();
        let mut agent_protocol_store_reg = self.agent_protocol_store.write();
        let mut wisdom_lineage_store_reg = self.wisdom_lineage_store.write();
        let mut symbol_event_log_reg = self.symbol_event_log.write();
        let mut chunk_hash_idx = self.chunk_hash_idx.write();
        let mut realm_members = self.realm_members.write();
        let mut kind_members  = self.kind_members.write();
        let mut coactivation_stats = self.coactivation_stats.write();

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
                &mut *msg_reg,
                &mut *skill_reg,
                &mut *agent_reg,
                &mut *constraint_reg,
                &mut *trigger_reg,
                &mut *surprise_reg,
                &mut *epistemic_debt_reg,
                &mut *integration_reg,
                &mut *surprise_learning_reg,
                &mut *wisdom_promotion_reg,
                &mut *learned_scorer_reg,
                &mut *intervention_store_reg,
                &mut *agent_protocol_store_reg,
                &mut *wisdom_lineage_store_reg,
                &mut *symbol_event_log_reg,
                &mut *chunk_hash_idx,
                &mut *realm_members,
                &mut *kind_members,
                &mut *coactivation_stats,
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
    msg_registry: &mut MsgRegistry,
    skill_registry: &mut SkillRegistry,
    agent_registry: &mut AgentRegistry,
    constraint_store: &mut ConstraintStore,
    trigger_store: &mut TriggerStore,
    surprise_store: &mut SurpriseStore,
    epistemic_debt_store: &mut EpistemicDebtStore,
    integration_kernel: &mut IntegrationKernel,
    surprise_learning: &mut SurpriseLearningStore,
    wisdom_promotion: &mut WisdomPromotionStore,
    learned_scorer: &mut LearnedScoringModel,
    intervention_store: &mut InterventionStore,
    agent_protocol_store: &mut AgentProtocolStore,
    wisdom_lineage_store: &mut WisdomLineageStore,
    symbol_event_log: &mut SymbolEventLog,
    chunk_hash_idx: &mut HashMap<crate::ids::ChunkHash, MemoryId>,
    realm_members: &mut HashMap<String, HashSet<MemoryId>>,
    kind_members:  &mut HashMap<String, HashSet<MemoryId>>,
    coactivation_stats: &mut HashMap<(MemoryId, MemoryId), CoActivationStats>,
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
            kind_members
                .entry(kind.clone())
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
                // Use op_ts_ms as the reference time during replay so that
                // last_accessed_ms / last_strengthened_ms are set to the
                // wall-clock time of the original operation, not epoch 0.
                let replay_now = if delta.op_ts_ms > 0 { delta.op_ts_ms } else { state.created_at_ms };
                state.apply_delta(&delta, replay_now);
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
                let remove_kind = if let Some(ids) = kind_members.get_mut(&payload.kind) {
                    ids.remove(&memory_id);
                    ids.is_empty()
                } else {
                    false
                };
                if remove_kind {
                    kind_members.remove(&payload.kind);
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
            code_files.upsert(
                &f.path, &f.project, f.mtime,
                f.content_hash.clone(), f.git_commit.clone(),
                f.git_author.clone(), f.git_timestamp_ms,
                || f.file_id,
            );
        }
        Op::InvalidateTripletsBySourceFile(op) => {
            triplet_store.invalidate_by_source_file(&op.source_file, op.invalidated_at_ms);
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
        Op::SessionEvent(ev) => {
            let payload_str = String::from_utf8(ev.payload_json.clone()).unwrap_or_default();
            match ev.kind.as_str() {
                "register" => {
                    let kind = serde_json::from_slice::<serde_json::Value>(&ev.payload_json)
                        .ok()
                        .and_then(|v| {
                            v.get("kind")
                                .and_then(|k| k.as_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_default();
                    session_registry.register(ev.session_id.clone(), kind, ev.realm.clone(), ev.ts_ms);
                }
                "heartbeat" => session_registry.heartbeat(&ev.session_id, ev.ts_ms),
                "deregister" => session_registry.deregister(&ev.session_id),
                _ => {}
            }
            // Mirror into msg_registry for get_events_by_domain_kind queries.
            use crate::organ::msg::MsgEvent;
            msg_registry.insert(MsgEvent {
                event_id: ev.event_id,
                domain: "session".to_string(),
                kind: ev.kind,
                target: ev.session_id,
                payload_json: payload_str,
                realm: ev.realm,
                ts_ms: ev.ts_ms,
            });
        }
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
                // An UpdateMemoryContent with a real embedding means the backfill
                // completed. Clear embed_pending so replayed state matches live state.
                if let Some(st) = states.get_mut(&umc.memory_id) {
                    st.embed_pending = false;
                }
            }
            keyword_idx.index(umc.memory_id, &content_str);
        }
        Op::UpdateMemoryKind(umk) => {
            if let Some(payload) = payloads.get_mut(&umk.memory_id) {
                payload.kind = umk.new_kind;
            }
        }
        Op::RecordRecallBatch(op) => {
            let ctx = crate::state::RetrievalContext {
                centroid_q: op.centroid_q.clone(),
                scale: op.centroid_scale,
                context_hash: op.context_hash,
                ts_ms: op.ts_ms,
            };
            // Touch each memory and append retrieval context
            for &mid in &op.memory_ids {
                if let Some(state) = states.get_mut(&mid) {
                    state.access_count += 1;
                    state.last_accessed_ms = op.ts_ms;
                    state.retrieval_history.push(ctx.clone());
                }
            }
            // Update pairwise co-activation stats and strengthen edges
            let ids = &op.memory_ids;
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    let key = (ids[i].min(ids[j]), ids[i].max(ids[j]));
                    let stats = coactivation_stats.entry(key).or_default();
                    stats.record(op.context_hash, op.ts_ms);
                    let multiplier = stats.hebbian_multiplier();
                    let delta = op.base_assoc_delta * multiplier;
                    strengthen_assoc_edge_map(assoc_edges, ids[i], ids[j], crate::ops::EdgeType::CoRetrieved, delta);
                }
            }
        }
        Op::StrengthenAssocEdge(op) => {
            strengthen_assoc_edge_map(assoc_edges, op.src, op.dst, op.edge_type, op.delta);
        }
        Op::MsgEvent(ev) => {
            use crate::organ::msg::MsgEvent;
            let payload_str = String::from_utf8(ev.payload_json).unwrap_or_default();
            msg_registry.insert(MsgEvent {
                event_id: ev.event_id,
                domain: ev.domain,
                kind: ev.kind,
                target: ev.target,
                payload_json: payload_str,
                realm: ev.realm,
                ts_ms: ev.ts_ms,
            });
        }
        Op::SkillUpload(s) => {
            skill_registry.upload(&s.skill_id, &s.content, &s.uploaded_by, &s.tags, s.ts_ms);
        }
        Op::SkillDeprecate(s) => {
            skill_registry.deprecate(&s.skill_id);
        }
        Op::AgentUpsert(a) => {
            agent_registry.upsert(&a.agent_id, &a.display_name, &a.description, a.ts_ms);
        }
        Op::AgentDisable(a) => {
            agent_registry.disable(&a.agent_id);
        }
        Op::AssertConstraint(c) => {
            use crate::organ::constraint::Provenance;
            constraint_store.replay_assert(
                c.fact_id, c.subject, c.predicate, c.object, c.confidence,
                c.scope, c.branch_id,
                Provenance {
                    source: c.provenance_source,
                    session_id: c.provenance_session,
                    confidence_basis: c.provenance_basis,
                },
                c.valid_from_ms, c.source_memory_id,
            );
        }
        Op::RetractConstraint(r) => {
            constraint_store.replay_retract(r.fact_id, r.retracted_at_ms);
        }
        Op::CreateBranch(b) => {
            constraint_store.replay_create_branch(b.branch_id, b.parent_id, b.scope, b.created_ms);
        }
        Op::ResolveBranch(r) => {
            constraint_store.replay_resolve_branch(r.winner_id, r.loser_id, r.resolved_at_ms);
        }
        Op::AddTrigger(t) => {
            if let Ok(trigger) = serde_json::from_slice(&t.trigger_json) {
                trigger_store.replay_add(trigger);
            }
        }
        Op::UpdateTrigger(u) => {
            let status = match u.status {
                0 => crate::organ::trigger::TriggerStatus::Armed,
                1 => crate::organ::trigger::TriggerStatus::Fired,
                2 => crate::organ::trigger::TriggerStatus::Expired,
                _ => crate::organ::trigger::TriggerStatus::Inhibited,
            };
            trigger_store.replay_update_status(u.trigger_id, status, u.fired_ms);
        }
        Op::FireTrigger(f) => {
            trigger_store.replay_update_status(
                f.trigger_id,
                crate::organ::trigger::TriggerStatus::Fired,
                f.fired_ms,
            );
        }
        Op::RecordSurprise(s) => {
            surprise_store.replay_record(crate::organ::surprise::SurpriseEvent {
                id: s.event_id,
                context_sketch: s.context_sketch,
                action: s.action,
                expected: s.expected,
                actual: s.actual,
                surprise_magnitude: s.surprise_magnitude,
                domain: s.domain,
                timestamp_ms: s.timestamp_ms,
                realm: s.realm,
                session_id: s.session_id,
                source_memory_id: s.source_memory_id,
            });
        }
        Op::RegisterDebt(d) => {
            epistemic_debt_store.replay_register(crate::organ::epistemic_debt::EpistemicDebt {
                id: d.debt_id,
                pattern: d.pattern,
                competing_hypotheses: d.competing_hypotheses,
                discriminating_test: d.discriminating_test,
                fragility_score: d.fragility_score,
                domain: d.domain,
                status: crate::organ::epistemic_debt::DebtStatus::Open,
                created_ms: d.created_ms,
                resolved_ms: 0,
                resolution: None,
                realm: d.realm,
                source_session: d.source_session,
                evidence: Vec::new(),
                auto_resolved: false,
            });
        }
        Op::UpdateDebt(u) => {
            let status = crate::organ::epistemic_debt::DebtStatus::from_u8(u.status);
            epistemic_debt_store.replay_update(u.debt_id, status, u.resolved_ms, u.resolution);
        }
        Op::UpdateSourceWeight(w) => {
            integration_kernel.replay_update_weight(w.source, w.query_domain, w.weight);
        }
        Op::RecordFeedback(f) => {
            integration_kernel.replay_feedback(
                f.source,
                f.query_domain,
                f.new_weight,
                f.success_count,
                f.total_count,
            );
        }
        Op::UpdateSurpriseCredit(c) => {
            surprise_learning.replay_credit(
                crate::organ::surprise_learning::SurpriseLearningState {
                    memory_id: c.memory_id,
                    credit: c.credit,
                    last_dir: c.last_dir,
                    same_dir_streak: c.same_dir_streak,
                    last_surprise_id: c.last_surprise_id,
                    updated_ms: c.updated_ms,
                },
            );
        }
        Op::UpsertWisdomCandidate(w) => {
            wisdom_promotion.replay_upsert(
                crate::organ::wisdom_promotion::WisdomCandidate {
                    id: w.candidate_id,
                    cluster_key: w.cluster_key,
                    domain: w.domain,
                    action: w.action,
                    summary: w.summary,
                    episode_ids: w.episode_ids,
                    debt_ids: w.debt_ids,
                    support_count: w.support_count,
                    cross_session_count: w.cross_session_count,
                    mean_surprise: w.mean_surprise,
                    promotion_score: w.promotion_score,
                    contradiction_count: 0,
                    lifecycle: crate::organ::wisdom_promotion::WisdomLifecycle::Candidate,
                    memory_id: None,
                    created_ms: w.created_ms,
                    updated_ms: w.created_ms,
                },
            );
        }
        Op::UpdateWisdomLifecycle(l) => {
            wisdom_promotion.replay_lifecycle(
                l.candidate_id,
                l.new_state,
                l.memory_id,
                l.contradiction_count,
                l.updated_ms,
            );
        }
        Op::UpdateScorerModel(m) => {
            learned_scorer.apply_update(
                &m.weights_json,
                m.model_version,
                m.mean_loss,
                m.outcome_count,
                m.applied_at_ms,
            );
        }
        Op::AttachDebtEvidence(e) => {
            epistemic_debt_store.replay_attach_evidence(
                e.debt_id,
                e.evidence_memory_ids,
                e.confidence,
                e.note,
                e.attached_ms,
            );
        }
        Op::StartIntervention(s) => {
            use crate::organ::intervention::{ActionType, InterventionRecord, InterventionStatus, ReversalCost};
            intervention_store.replay_start(InterventionRecord {
                id: s.id,
                realm: s.realm,
                session_id: s.session_id,
                task_id: s.task_id,
                agent_id: s.agent_id,
                domain: s.domain,
                intent: s.intent,
                action_type: ActionType::from_u8(s.action_type),
                action_ref: s.action_ref,
                preconditions: s.preconditions,
                expected_observables: s.expected_observables,
                reversal_cost: ReversalCost::from_u8(s.reversal_cost),
                started_ms: s.started_ms,
                closed_ms: None,
                status: InterventionStatus::Open,
            });
        }
        Op::AddObservation(o) => {
            use crate::organ::intervention::{ObservationKind, ObservationRecord};
            intervention_store.replay_observation(ObservationRecord {
                id: o.id,
                intervention_id: o.intervention_id,
                kind: ObservationKind::from_u8(o.kind),
                evidence_refs: o.evidence_refs,
                summary: o.summary,
                confidence: o.confidence,
                timestamp_ms: o.timestamp_ms,
            });
        }
        Op::CloseIntervention(c) => {
            use crate::organ::intervention::InterventionStatus;
            intervention_store.replay_close(
                c.intervention_id,
                InterventionStatus::from_u8(c.status),
                c.closed_ms,
            );
        }
        Op::RecordAttribution(a) => {
            use crate::organ::intervention::{AttributionClass, AttributionRecord};
            intervention_store.replay_attribution(AttributionRecord {
                intervention_id: a.intervention_id,
                primary_class: AttributionClass::from_u8(a.primary_class),
                secondary_class: a.secondary_class.map(AttributionClass::from_u8),
                confidence_delta: a.confidence_delta,
                surprise_id: a.surprise_id,
                debt_ids: a.debt_ids,
                source_memory_ids: a.source_memory_ids,
                skill_memory_ids: a.skill_memory_ids,
                note: a.note,
                timestamp_ms: a.timestamp_ms,
            });
        }
        // ── Layer 8: Agent Protocol Memory ──────────────────────────────────
        Op::RegisterTask(t) => {
            use crate::organ::agent_protocol::TaskContract;
            use crate::organ::agent_protocol::TaskStatus;
            agent_protocol_store.replay_register_task(TaskContract {
                id: t.id,
                session_id: t.session_id,
                realm: t.realm,
                goal: t.goal,
                constraints: t.constraints,
                acceptance_criteria: t.acceptance_criteria,
                priority: t.priority,
                status: TaskStatus::Active,
                parent_task_id: t.parent_task_id,
                intervention_ids: Vec::new(),
                tags: t.tags,
                created_ms: t.created_ms,
                updated_ms: t.created_ms,
                deadline_ms: t.deadline_ms,
            });
        }
        Op::UpdateTask(u) => {
            agent_protocol_store.replay_update_task(
                u.task_id,
                u.status,
                u.add_intervention_id,
                u.add_tag,
                u.updated_ms,
            );
        }
        Op::AddDelegation(d) => {
            use crate::organ::agent_protocol::{DelegationEdge, DelegationStatus};
            agent_protocol_store.replay_delegation(DelegationEdge {
                id: d.id,
                task_id: d.task_id,
                from_agent: d.from_agent,
                to_agent: d.to_agent,
                delegated_at: d.delegated_at,
                handoff_note: d.handoff_note,
                status: DelegationStatus::Active,
            });
        }
        Op::LinkEvidence(e) => {
            use crate::organ::agent_protocol::{EvidenceKind, EvidenceLink};
            agent_protocol_store.replay_evidence(EvidenceLink {
                id: e.id,
                task_id: e.task_id,
                memory_id: e.memory_id,
                produced_by: e.produced_by,
                evidence_kind: EvidenceKind::from_u8(e.evidence_kind),
                relevance: e.relevance,
                created_ms: e.created_ms,
            });
        }
        Op::AddProbe(p) => {
            use crate::organ::agent_protocol::{PendingProbe, ProbeStatus};
            agent_protocol_store.replay_probe(PendingProbe {
                id: p.id,
                task_id: p.task_id,
                question: p.question,
                expected_answerer: p.expected_answerer,
                priority: p.priority,
                status: ProbeStatus::Open,
                created_ms: p.created_ms,
                resolved_ms: None,
                answer: None,
            });
        }
        Op::ResolveProbe(r) => {
            agent_protocol_store.replay_resolve_probe(
                r.probe_id,
                r.status,
                r.answer,
                r.resolved_ms,
            );
        }
        Op::SetCriterion(c) => {
            use crate::organ::agent_protocol::CompletionCriterion;
            agent_protocol_store.replay_criterion(CompletionCriterion {
                id: c.id,
                task_id: c.task_id,
                criterion: c.criterion,
                is_met: c.is_met,
                checked_ms: Some(c.checked_ms),
                evidence_note: c.evidence_note,
            });
        }
        // Layer 9: Wisdom Homeostasis
        Op::UpsertWisdomLineage(l) => {
            use crate::organ::wisdom_lineage::{ApplicabilityEnvelope, WisdomLineage, LineageState};
            let envelope: ApplicabilityEnvelope =
                serde_json::from_str(&l.envelope_json).unwrap_or_default();
            wisdom_lineage_store.replay_upsert(WisdomLineage {
                id: l.lineage_id,
                wisdom_candidate_id: l.wisdom_candidate_id,
                claim: l.claim,
                envelope,
                seed_episode_ids: l.seed_episode_ids,
                seed_surprise_ids: l.seed_surprise_ids,
                seed_intervention_ids: l.seed_intervention_ids,
                seed_debt_ids: l.seed_debt_ids,
                ancestor_lineage_id: l.ancestor_lineage_id,
                derivation_version: l.derivation_version,
                derivation_relation: l.derivation_relation,
                support_mass: 0.0,
                contradiction_mass: 0.0,
                staleness_mass: 0.0,
                last_supported_ms: l.created_ms,
                last_challenged_ms: 0,
                state: LineageState::Trusted,
                challengers: Vec::new(),
                rederive_task_id: None,
                rederive_opened_ms: None,
                rederive_ttl_ms: l.rederive_ttl_ms,
                created_ms: l.created_ms,
                updated_ms: l.updated_ms,
            });
        }
        Op::AdjudicateLineage(a) => {
            wisdom_lineage_store.replay_adjudicate(
                a.lineage_id,
                a.support_mass,
                a.contradiction_mass,
                a.staleness_mass,
                a.last_supported_ms,
                a.last_challenged_ms,
                a.adjudicated_ms,
            );
        }
        Op::TransitionLineage(t) => {
            use crate::organ::wisdom_lineage::LineageState;
            wisdom_lineage_store.transition_state(
                t.lineage_id,
                LineageState::from_u8(t.new_state),
                "",
                t.rederive_task_id,
                t.transitioned_ms,
            );
        }
        Op::RecordChallenger(c) => {
            use crate::organ::wisdom_lineage::ChallengerEvidence;
            wisdom_lineage_store.record_challenger(
                c.lineage_id,
                ChallengerEvidence {
                    intervention_id: c.intervention_id,
                    surprise_id: c.surprise_id,
                    outcome_summary: c.outcome_summary,
                    attached_ms: c.attached_ms,
                },
                c.attached_ms,
            );
        }
        Op::CloseRederive(r) => {
            use crate::organ::wisdom_lineage::{RederiveAction, ApplicabilityEnvelope};
            let new_envelope = r.new_envelope_json.as_deref()
                .and_then(|j| serde_json::from_str::<ApplicabilityEnvelope>(j).ok());
            wisdom_lineage_store.close_rederive(
                r.lineage_id,
                RederiveAction::from_u8(r.action),
                new_envelope,
                r.fork_claim,
                r.fork_lineage_id,
                r.closed_ms,
            );
        }

        Op::SymbolEvent(e) => {
            symbol_event_log.replay(SymbolEvent {
                id: e.id,
                symbol_name: e.symbol_name,
                file_path: e.file_path,
                symbol_id: e.symbol_id,
                kind: SymbolEventKind::from_u8(e.kind),
                session_id: e.session_id,
                harness: e.harness,
                memory_id: e.memory_id,
                timestamp_ms: e.timestamp_ms,
                notes: e.notes,
            });
        }
    }
}

/// Upsert an assoc edge: if one already exists between (src, dst, edge_type),
/// add `delta` to its weight. Otherwise insert a new edge with weight = `delta`.
/// Also maintains the reverse direction edge.
/// Upsert an assoc edge on a raw HashMap. Public so FFI can apply in-memory
/// without going through the full apply_op path.
pub fn strengthen_assoc_edge_map(
    assoc_edges: &mut HashMap<MemoryId, Vec<AssocEdge>>,
    src: MemoryId,
    dst: MemoryId,
    edge_type: EdgeType,
    delta: f32,
) {
    // Forward direction
    let edges = assoc_edges.entry(src).or_default();
    let mut found = false;
    for edge in edges.iter_mut() {
        if edge.dst == dst && std::mem::discriminant(&edge.edge_type) == std::mem::discriminant(&edge_type) {
            edge.weight = (edge.weight + delta).min(1.0);
            found = true;
            break;
        }
    }
    if !found {
        edges.push(AssocEdge {
            dst,
            edge_type: edge_type.clone(),
            weight: delta.clamp(0.0, 1.0),
        });
    }
    // Reverse direction
    let rev_edges = assoc_edges.entry(dst).or_default();
    let mut rev_found = false;
    for edge in rev_edges.iter_mut() {
        if edge.dst == src && std::mem::discriminant(&edge.edge_type) == std::mem::discriminant(&edge_type) {
            edge.weight = (edge.weight + delta).min(1.0);
            rev_found = true;
            break;
        }
    }
    if !rev_found {
        rev_edges.push(AssocEdge {
            dst: src,
            edge_type,
            weight: delta.clamp(0.0, 1.0),
        });
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

fn build_kind_members(
    payloads: &HashMap<MemoryId, MemoryPayload>,
    states: &HashMap<MemoryId, MemoryState>,
) -> HashMap<String, HashSet<MemoryId>> {
    let mut kind_members: HashMap<String, HashSet<MemoryId>> = HashMap::new();
    for (&memory_id, payload) in payloads {
        let not_deleted = states.get(&memory_id).map(|s| !s.deleted).unwrap_or(false);
        if !not_deleted {
            continue;
        }
        kind_members
            .entry(payload.kind.clone())
            .or_default()
            .insert(memory_id);
    }
    kind_members
}

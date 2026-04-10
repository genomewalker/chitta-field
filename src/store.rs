use crate::error::{FieldError, Result};
use crate::field::{AssocEdge, ChittaField};
use crate::ids::{compute_chunk_hash, ArtifactId, ChunkHash, MemoryId};
use crate::learner::route::{Route, RouteLearner};
use crate::ops::EMBED_DIM;
use crate::ops::{
    AddAssocEdgeOp, AddSymCallEdgeOp, AddTripletOp, ArtifactRef, DeleteMemoryOp, DemoteMemoryOp,
    EdgeType, InvalidateTripletOp, Op, PutPayloadOp, RemoveSymbolOp, StateDeltaOp, TrainPQOp,
    UpdateResidualPQOp, UpdateSparseCodeOp, UpsertArtifactOp, UpsertCodeFileOp, UpsertSymbolOp,
};
use crate::organ::pq::ProductQuantizer;
use crate::organ::symbol::SymbolEntry;
use crate::organ::triplet::TripletEntry;
use crate::payload::MemoryPayload;
use crate::recall::RecallHit;
use crate::state::{EpistemicStatus, MemoryState, MemoryStatus};
use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterLevel {
    #[default]
    None,
    Signatures,
    MinimalContext,
}

pub fn extract_bm25_text(content: &str, level: FilterLevel) -> String {
    match level {
        FilterLevel::None => content.to_string(),
        FilterLevel::Signatures => extract_signatures(content),
        FilterLevel::MinimalContext => extract_signatures_with_docs(content),
    }
}

fn is_signature_line(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("pub fn ") || t.starts_with("fn ") ||
    t.starts_with("pub struct ") || t.starts_with("struct ") ||
    t.starts_with("pub enum ") || t.starts_with("enum ") ||
    t.starts_with("pub trait ") || t.starts_with("trait ") ||
    t.starts_with("impl ") || t.starts_with("pub impl ") ||
    t.starts_with("pub type ") || t.starts_with("type ") ||
    t.starts_with("pub const ") || t.starts_with("const ") ||
    t.starts_with("def ") || t.starts_with("class ") ||
    t.starts_with("function ") || t.starts_with("async fn ") ||
    t.starts_with("pub async fn ")
}

fn extract_signatures(content: &str) -> String {
    content.lines().filter(|l| is_signature_line(l)).collect::<Vec<_>>().join("\n")
}

fn extract_signatures_with_docs(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let mut result = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if is_signature_line(line) {
            if i > 0 {
                let prev = lines[i - 1].trim();
                if prev.starts_with("///") || prev.starts_with("//") || prev.starts_with('#') {
                    result.push(lines[i - 1]);
                }
            }
            result.push(line);
        }
    }
    result.join("\n")
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// Score multiplier derived from memory status.
/// Superseded/Contradicted/Archived return None → excluded from recall.
/// Active=1.0, Verified=1.15, Observed=0.85, Proposed=0.65.
fn status_score_multiplier(status: &MemoryStatus) -> Option<f32> {
    match status {
        MemoryStatus::Active       => Some(1.00),
        MemoryStatus::Verified     => Some(1.15),
        MemoryStatus::Observed     => Some(0.85),
        MemoryStatus::Proposed     => Some(0.65),
        MemoryStatus::Superseded   => None,
        MemoryStatus::Contradicted => None,
        MemoryStatus::Archived     => None,
    }
}

/// Score multiplier based on memory kind for tiered recall prioritization.
/// Corrections and preferences are high-signal; episodes are low-signal noise.
fn realm_kind_multiplier(kind: &str) -> f32 {
    match kind {
        "correction" | "preference" => 1.3,
        "wisdom"                    => 1.1,
        "insight"                   => 1.05,
        "episode"                   => 0.7,
        _                           => 1.0,
    }
}

/// Score multiplier derived from epistemic status.
/// UserStated=1.0, ToolDerived=0.95, ModelInferred=0.85, AutonomousSynthesis=0.75.
fn epistemic_score_multiplier(es: &EpistemicStatus) -> f32 {
    match es {
        EpistemicStatus::UserStated          => 1.00,
        EpistemicStatus::ToolDerived         => 0.95,
        EpistemicStatus::ModelInferred       => 0.85,
        EpistemicStatus::AutonomousSynthesis => 0.75,
    }
}

/// Compute embedding geometry stats for a group of embeddings.
/// Returns JSON value with group name, count, effective_dim, isotropy, mean_cosine_sim.
fn compute_geometry(embeddings: &[&[f32]], group_name: &str) -> Option<serde_json::Value> {
    let n = embeddings.len();
    if n < 2 {
        return None;
    }
    let dim = EMBED_DIM;
    let n_f = n as f64;

    // Per-dimension mean
    let mut mean = vec![0.0f64; dim];
    for emb in embeddings {
        for (i, &v) in emb.iter().enumerate() {
            mean[i] += v as f64;
        }
    }
    for m in &mut mean {
        *m /= n_f;
    }

    // Per-dimension variance
    let mut variance = vec![0.0f64; dim];
    for emb in embeddings {
        for (i, &v) in emb.iter().enumerate() {
            let d = v as f64 - mean[i];
            variance[i] += d * d;
        }
    }
    for v in &mut variance {
        *v /= n_f;
    }

    // Participation ratio: effective dimensionality
    let sum_var: f64 = variance.iter().sum();
    let sum_var_sq: f64 = variance.iter().map(|v| v * v).sum();
    let effective_dim = if sum_var_sq > 1e-30 {
        (sum_var * sum_var) / sum_var_sq
    } else {
        0.0
    };
    let isotropy = effective_dim / dim as f64;

    // Mean pairwise cosine similarity (sample if large)
    let max_pairs = 500usize;
    let mut cos_sum = 0.0f64;
    let mut pair_count = 0u64;
    if n <= 32 {
        for i in 0..n {
            for j in (i + 1)..n {
                let dot: f64 = embeddings[i]
                    .iter()
                    .zip(embeddings[j].iter())
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum();
                cos_sum += dot;
                pair_count += 1;
            }
        }
    } else {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        group_name.hash(&mut h);
        let mut seed = h.finish();
        for _ in 0..max_pairs {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let i = (seed >> 32) as usize % n;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (seed >> 32) as usize % n;
            if i == j {
                continue;
            }
            let dot: f64 = embeddings[i]
                .iter()
                .zip(embeddings[j].iter())
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum();
            cos_sum += dot;
            pair_count += 1;
        }
    }
    let mean_cosine = if pair_count > 0 {
        cos_sum / pair_count as f64
    } else {
        0.0
    };

    Some(serde_json::json!({
        "group": group_name,
        "count": n,
        "effective_dim": (effective_dim * 10.0).round() / 10.0,
        "isotropy": (isotropy * 1000.0).round() / 1000.0,
        "mean_cosine_sim": (mean_cosine * 1000.0).round() / 1000.0,
    }))
}

impl ChittaField {
    /// Store a new memory. Returns `(MemoryId, ChunkHash)`.
    pub fn put_memory(
        &self,
        kind: &str,
        realm: &str,
        content: &[u8],
        embedding: &[f32],
        confidence: f32,
        decay_rate: f32,
        authored_at_ms: i64,
        artifact_refs: Vec<ArtifactRef>,
        source_session: Option<String>,
        source_tool: Option<String>,
    ) -> Result<(MemoryId, ChunkHash)> {
        let embed_pending = embedding.is_empty();
        if !embed_pending && embedding.len() != EMBED_DIM {
            return Err(FieldError::InvalidEmbedDim {
                expected: EMBED_DIM,
                actual: embedding.len(),
            });
        }

        let chunk_hash = compute_chunk_hash(kind, realm, content, embedding);

        {
            let idx = self.chunk_hash_idx.read();
            if let Some(&existing_id) = idx.get(&chunk_hash) {
                drop(idx);
                // Recurrence: same observation seen again → boost confidence (+0.05)
                // After 6+ recurrences, provisional (0.50) reaches durable tier (0.80)
                let _ = self.update_state(existing_id, Some(0.0), Some(0.05), None, true, None);
                return Ok((existing_id, chunk_hash));
            }
        }

        // Semantic novelty gate (Omni-SimpleMem selective ingestion):
        // If a near-duplicate already exists (cosine_sim ≥ 0.88), skip storage and
        // lightly reinforce the existing memory instead of creating a new node.
        // Only deduplicates within the same realm — cross-realm near-matches must
        // produce independent nodes to prevent silent cross-realm reinforcement.
        if !embed_pending {
            let neighbors = self.semantic_idx.read().search(embedding, 1, None);
            if let Some(top) = neighbors.first() {
                if top.cosine_similarity >= 0.88 && top.cosine_similarity < 0.9999 {
                    let candidate_realm = self.payloads.read()
                        .get(&top.memory_id)
                        .map(|p| p.realm.clone())
                        .unwrap_or_default();
                    if candidate_realm == realm {
                        let _ = self.update_state(top.memory_id, Some(0.0), Some(0.02), None, true, None);
                        return Ok((top.memory_id, chunk_hash));
                    }
                }
            }
        }

        let memory_id = self.id_alloc.next_id();
        let ts = now_ms();

        let authored_at_ms = if authored_at_ms == 0 {
            ts
        } else {
            authored_at_ms
        };

        let op = PutPayloadOp {
            memory_id,
            version: 0,
            chunk_hash,
            created_at_ms: ts,
            authored_at_ms,
            kind: kind.to_string(),
            realm: realm.to_string(),
            content: content.to_vec(),
            embedding_model: if embed_pending { "none".to_string() } else { "bge-base-en-v1.5".to_string() },
            embedding: embedding.to_vec(),
            artifact_refs: artifact_refs.clone(),
            source_session,
            source_tool,
        };

        let op_enum = Op::PutPayload(op.clone());
        let _seqno = self.log.write().append(&op_enum)?;
        // Sync after durable write
        let _ = self.log.write().sync();

        let payload = MemoryPayload::from(op);
        let mut state = MemoryState::new(memory_id, chunk_hash, ts);
        state.confidence = confidence;
        state.decay_rate = decay_rate;

        state.embed_pending = embed_pending;
        self.payloads.write().insert(memory_id, payload);
        self.states.write().insert(memory_id, state);
        self.chunk_hash_idx
            .write()
            .entry(chunk_hash)
            .or_insert(memory_id);
        self.realm_members
            .write()
            .entry(realm.to_string())
            .or_default()
            .insert(memory_id);
        if !embed_pending {
            self.semantic_idx
                .write()
                .upsert(memory_id, embedding.to_vec());
        }
        let content_str = std::str::from_utf8(content).unwrap_or("").to_string();
        let index_text = extract_bm25_text(&content_str, self.filter_level());
        self.keyword_idx.write().index(memory_id, &index_text);

        // Update temporal index.
        {
            use crate::organ::temporal::TemporalEntry;
            self.time_idx.write().upsert(TemporalEntry {
                memory_id,
                ts_ms: authored_at_ms,
                kind: kind.to_string(),
                realm: realm.to_string(),
                strength: 1.0,
            });
        }

        // Update artifact index for each artifact ref.
        {
            let artifact_paths = self.artifact_paths.read();
            let mut artifact_idx = self.artifact_idx.write();
            for art_ref in &artifact_refs {
                if let Some(path) = artifact_paths.get(&art_ref.artifact_id) {
                    artifact_idx.associate(memory_id, art_ref.artifact_id, path, 1.0);
                }
            }
        }

        // Auto-encode into cortical sparse index (non-fatal if fails)
        let _ = self.encode_memory(memory_id);

        // PoE: corrections penalise the realm they target.
        // A correction stored in realm X signals that X produced an error.
        if kind == "correction" {
            self.learners
                .write()
                .domain_reliability
                .record_correction(realm);
        }

        Ok((memory_id, chunk_hash))
    }

    /// Retrieve the payload for a memory. Also records a touch access.
    pub fn get_memory(&self, memory_id: MemoryId) -> Result<MemoryPayload> {
        {
            let states = self.states.read();
            let state = states
                .get(&memory_id)
                .ok_or(FieldError::NotFound(memory_id))?;
            if state.deleted {
                return Err(FieldError::Deleted(memory_id));
            }
        }

        let ts = now_ms();

        // Record access in plasticity learner and get recommended decay rate.
        let recommended_decay = self
            .learners
            .write()
            .plasticity
            .record_access(memory_id, ts);

        // Check if current decay rate differs significantly; if so, update it.
        let new_decay_rate = {
            let states = self.states.read();
            states.get(&memory_id).and_then(|state| {
                let diff = (state.decay_rate - recommended_decay).abs();
                if diff > 0.0001 {
                    Some(recommended_decay)
                } else {
                    None
                }
            })
        };

        // Touch: append UpdateState op then apply to in-memory state.
        let delta = StateDeltaOp {
            memory_id,
            strength_delta: None,
            confidence_delta: None,
            decay_rate: new_decay_rate,
            touch: true,
            pin: None,
            op_ts_ms: ts,
            status: None,
            epistemic_status: None,
        };
        let _seqno = self.log.write().append(&Op::UpdateState(delta.clone()))?;

        {
            let mut states = self.states.write();
            if let Some(state) = states.get_mut(&memory_id) {
                state.apply_delta(&delta, ts);
            }
        }

        let payloads = self.payloads.read();
        payloads
            .get(&memory_id)
            .cloned()
            .ok_or(FieldError::NotFound(memory_id))
    }

    /// Return current mutable state for a memory.
    pub fn get_state(&self, memory_id: MemoryId) -> Result<MemoryState> {
        self.states
            .read()
            .get(&memory_id)
            .cloned()
            .ok_or(FieldError::NotFound(memory_id))
    }

    /// Apply a delta to a memory's mutable state.
    pub fn update_state(
        &self,
        memory_id: MemoryId,
        strength_delta: Option<f32>,
        confidence_delta: Option<f32>,
        decay_rate: Option<f32>,
        touch: bool,
        pin: Option<bool>,
    ) -> Result<()> {
        {
            let states = self.states.read();
            let state = states
                .get(&memory_id)
                .ok_or(FieldError::NotFound(memory_id))?;
            if state.deleted {
                return Err(FieldError::Deleted(memory_id));
            }
        }

        let ts = now_ms();
        let delta = StateDeltaOp {
            memory_id,
            strength_delta,
            confidence_delta,
            decay_rate,
            touch,
            pin,
            op_ts_ms: ts,
            status: None,
            epistemic_status: None,
        };
        let _seqno = self.log.write().append(&Op::UpdateState(delta.clone()))?;

        {
            let mut states = self.states.write();
            if let Some(state) = states.get_mut(&memory_id) {
                state.apply_delta(&delta, ts);
            }
        }

        Ok(())
    }

    /// Soft-delete a memory.
    pub fn forget(&self, memory_id: MemoryId) -> Result<()> {
        {
            let states = self.states.read();
            states
                .get(&memory_id)
                .ok_or(FieldError::NotFound(memory_id))?;
        }

        let ts = now_ms();
        let op = Op::DeleteMemory(DeleteMemoryOp {
            memory_id,
            deleted_at_ms: ts,
        });
        let _seqno = self.log.write().append(&op)?;
        let _ = self.log.write().sync(); // forget is irreversible — sync immediately

        {
            let mut states = self.states.write();
            if let Some(state) = states.get_mut(&memory_id) {
                state.deleted = true;
            }
        }
        self.semantic_idx.write().remove(memory_id);
        self.keyword_idx.write().remove(memory_id);
        self.cortical_idx.write().remove(memory_id);

        // Remove from temporal index (need authored_at_ms from payload).
        {
            let payloads = self.payloads.read();
            if let Some(payload) = payloads.get(&memory_id) {
                self.time_idx
                    .write()
                    .remove(memory_id, payload.authored_at_ms);
                let mut realm_members = self.realm_members.write();
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
        }
        self.artifact_idx.write().remove_memory(memory_id);

        Ok(())
    }

    /// Add a directed association edge between two memories.
    pub fn add_assoc_edge(
        &self,
        src: MemoryId,
        dst: MemoryId,
        edge_type: EdgeType,
        weight: f32,
    ) -> Result<()> {
        {
            let states = self.states.read();
            let src_state = states.get(&src).ok_or(FieldError::NotFound(src))?;
            if src_state.deleted {
                return Err(FieldError::Deleted(src));
            }
            let dst_state = states.get(&dst).ok_or(FieldError::NotFound(dst))?;
            if dst_state.deleted {
                return Err(FieldError::Deleted(dst));
            }
        }

        let op = Op::AddAssocEdge(AddAssocEdgeOp {
            src,
            dst,
            edge_type: edge_type.clone(),
            weight,
        });
        let _seqno = self.log.write().append(&op)?;

        self.assoc_edges
            .write()
            .entry(src)
            .or_insert_with(Vec::new)
            .push(AssocEdge {
                dst,
                edge_type,
                weight,
            });

        Ok(())
    }

    /// Return outbound association edges for a memory.
    pub fn list_neighbors(&self, memory_id: MemoryId) -> Result<Vec<AssocEdge>> {
        Ok(self
            .assoc_edges
            .read()
            .get(&memory_id)
            .cloned()
            .unwrap_or_default())
    }

    /// Total count of non-deleted memories.
    pub fn memory_count(&self) -> usize {
        self.states.read().values().filter(|s| !s.deleted).count()
    }

    fn enqueue_recall_effects(&self, hit_ids: &[MemoryId]) {
        const MAX_PENDING_STRENGTHEN: usize = 100_000;
        const MAX_PENDING_PAIRS: usize = 200_000;
        const MAX_PENDING_WINDOWS: usize = 20_000;

        if hit_ids.is_empty() {
            return;
        }

        let strengthen_ids = &hit_ids[..hit_ids.len().min(16)];
        let window_ids = hit_ids[..hit_ids.len().min(5)].to_vec();

        let mut pending = self.pending_recall.lock();
        for &id in strengthen_ids {
            if pending.strengthen.len() >= MAX_PENDING_STRENGTHEN {
                break;
            }
            pending.strengthen.insert(id);
        }

        for i in 0..window_ids.len() {
            for j in (i + 1)..window_ids.len() {
                if pending.co_retrieval_pairs.len() >= MAX_PENDING_PAIRS
                    && !pending
                        .co_retrieval_pairs
                        .contains_key(&(window_ids[i], window_ids[j]))
                {
                    continue;
                }
                *pending
                    .co_retrieval_pairs
                    .entry((window_ids[i], window_ids[j]))
                    .or_insert(0.0) += 0.05;
            }
        }

        if !window_ids.is_empty() && pending.proto_windows.len() < MAX_PENDING_WINDOWS {
            pending.proto_windows.push(window_ids);
        }
    }

    pub(crate) fn drain_pending_recall_effects(&self) -> Result<()> {
        let pending = {
            let mut guard = self.pending_recall.lock();
            if guard.strengthen.is_empty()
                && guard.co_retrieval_pairs.is_empty()
                && guard.proto_windows.is_empty()
            {
                return Ok(());
            }
            std::mem::take(&mut *guard)
        };

        for memory_id in pending.strengthen {
            let _ = self.update_state(memory_id, Some(0.01), None, None, true, None);
            let new_strength = self.states.read().get(&memory_id).map(|s| s.strength);
            if let Some(strength) = new_strength {
                self.cortical_idx
                    .write()
                    .update_strength(memory_id, strength);
            }
        }

        for ((src, dst), weight) in pending.co_retrieval_pairs {
            let _ = self.add_assoc_edge(src, dst, EdgeType::CoRetrieved, weight);
        }

        if !pending.proto_windows.is_empty() {
            let mut cortical_idx = self.cortical_idx.write();
            for window in pending.proto_windows {
                cortical_idx.strengthen_proto_transitions(&window);
            }
        }

        Ok(())
    }

    /// Semantic recall: find k most similar memories to a query embedding.
    ///
    /// Applies realm filter and strength-weighted final scoring:
    ///   `score = semantic_score × (0.5 + 0.5 × effective_strength) × confidence`
    ///
    /// Uses the ANN semantic index directly, with optional realm filtering.
    /// Recall-side maintenance effects are deferred until flush/snapshot.
    pub fn recall_semantic(
        &self,
        query_embedding: &[f32],
        k: usize,
        realm: Option<&str>,
    ) -> Result<Vec<RecallHit>> {
        if query_embedding.len() != EMBED_DIM {
            return Err(FieldError::InvalidEmbedDim {
                expected: EMBED_DIM,
                actual: query_embedding.len(),
            });
        }

        let now = now_ms();
        let realm_members = self.realm_members.read();
        let allowed = realm.and_then(|r| realm_members.get(r));

        let result_limit = k.saturating_mul(3).max(k);
        let semantic_hits = self
            .semantic_idx
            .read()
            .search(query_embedding, result_limit, allowed);

        let states = self.states.read();
        let payloads = self.payloads.read();

        let mut hits: Vec<RecallHit> = semantic_hits
            .into_iter()
            .filter_map(|hit| {
                let memory_id = hit.memory_id;
                let state = states.get(&memory_id)?;
                if state.deleted {
                    return None;
                }
                // soul:* realms are internal synthesis artifacts — exclude from unscoped
                // queries so they never compete with domain memories in brahman recall.
                // Only surfaced when the caller explicitly targets a soul:* realm.
                let payload_realm = payloads.get(&memory_id).map(|p| p.realm.as_str()).unwrap_or("");
                if payload_realm.starts_with("soul:") && realm.map(|r| !r.starts_with("soul:")).unwrap_or(true) {
                    return None;
                }
                // Exclude semantically invalidated statuses; apply score modifiers for the rest.
                let status_mul = status_score_multiplier(&state.status)?;
                let epistemic_mul = epistemic_score_multiplier(&state.epistemic_status);
                let payload = payloads.get(&memory_id)?;
                let eff_strength = state.effective_strength(now);
                let semantic_score = hit.cosine_similarity;
                let semantic_weight = ((semantic_score + 1.0) / 2.0).max(0.0);
                let strength_factor = 0.5 + 0.5 * eff_strength;
                // ACT-R: power-law base-level activation (Anderson & Schooler 1991)
                let actr_activation = state.actr_base_level_activation(now);
                let actr_factor = 0.3 + 0.7 * actr_activation; // [0.3, 1.0]
                // FEP: prediction error weighting — high-surprise memories are novel
                let surprise_boost = 1.0 + 0.25 * state.surprise; // [1.0, 1.25]
                // Flashbulb: high-arousal memories resist forgetting
                let arousal_boost = 1.0 + 0.15 * state.affect_arousal; // [1.0, 1.15]
                let base = semantic_weight * actr_factor * state.confidence
                    * surprise_boost * arousal_boost;
                // PoE: apply per-realm reliability multiplier
                let poe_mul = self.learners.read().domain_reliability.reliability(&payload.realm);
                let kind_mul = realm_kind_multiplier(&payload.kind);
                Some(RecallHit {
                    memory_id,
                    score: base * status_mul * epistemic_mul * poe_mul * kind_mul,
                    semantic_score,
                    ts_ms: payload.authored_at_ms,
                    kind: payload.kind.clone(),
                    realm: payload.realm.clone(),
                    strength: eff_strength,
                    confidence: state.confidence,
                    access_count: state.access_count,
                    content: String::from_utf8(payload.content.clone()).unwrap_or_default(),
                    semantic_weight,
                    status_mul,
                    epistemic_mul,
                    strength_factor,
                    affect_valence: state.affect_valence,
                    affect_arousal: state.affect_arousal,
                    actr_activation,
                    surprise_boost,
                    arousal_boost,
                })
            })
            .collect();

        hits.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);

        let hit_ids: Vec<MemoryId> = hits.iter().map(|h| h.memory_id).collect();
        drop(states);
        drop(payloads);
        self.enqueue_recall_effects(&hit_ids);

        Ok(hits)
    }

    /// Expand from seed memory IDs via typed association edges (max 2 hops).
    ///
    /// Returns memories discovered via the association graph, scored by
    /// spreading activation with hop decay (×0.55 per hop).
    ///
    /// Edge type priors:
    ///   DerivedFrom=1.0, SameArtifact=0.8, SameSession=0.6, CoRetrieved=0.5,
    ///   Supports=0.4, Contradicts=0.3
    pub fn expand_associations(
        &self,
        seed_ids: &[MemoryId],
        max_hops: usize,
        limit: usize,
    ) -> Result<Vec<RecallHit>> {
        const HOP_DECAY: f32 = 0.55;
        const FANOUT_CAP: usize = 16;
        let max_hops = max_hops.min(2);

        let edge_prior = |et: &EdgeType| -> f32 {
            match et {
                EdgeType::DerivedFrom => 1.0,
                EdgeType::SameArtifact => 0.8,
                EdgeType::SameSession => 0.6,
                EdgeType::CoRetrieved => 0.5,
                EdgeType::Supports => 0.4,
                EdgeType::Contradicts => 0.3,
            }
        };

        let seed_set: HashSet<MemoryId> = seed_ids.iter().copied().collect();
        // activation accumulator: memory_id -> max activation score seen
        let mut activation: std::collections::HashMap<MemoryId, f32> =
            std::collections::HashMap::new();

        // frontier: (memory_id, activation_score, hops_remaining)
        let mut frontier: Vec<(MemoryId, f32, usize)> =
            seed_ids.iter().map(|&id| (id, 1.0, max_hops)).collect();

        let assoc_edges = self.assoc_edges.read();
        let states = self.states.read();

        while let Some((node, act, hops_left)) = frontier.pop() {
            if hops_left == 0 {
                continue;
            }
            let neighbors = match assoc_edges.get(&node) {
                Some(v) => v,
                None => continue,
            };
            for edge in neighbors.iter().take(FANOUT_CAP) {
                let dst = edge.dst;
                // Skip deleted memories.
                if states.get(&dst).map(|s| s.deleted).unwrap_or(true) {
                    continue;
                }
                let edge_act = act * HOP_DECAY * edge_prior(&edge.edge_type) * edge.weight;
                let entry = activation.entry(dst).or_insert(0.0);
                if edge_act > *entry {
                    *entry = edge_act;
                    // Only continue expanding if this is a new or improved path.
                    if hops_left > 1 {
                        frontier.push((dst, edge_act, hops_left - 1));
                    }
                }
            }
        }

        drop(assoc_edges);

        let payloads = self.payloads.read();
        let now = now_ms();

        let mut hits: Vec<RecallHit> = activation
            .into_iter()
            .filter(|(id, _)| !seed_set.contains(id))
            .filter_map(|(id, act_score)| {
                let state = states.get(&id)?;
                let payload = payloads.get(&id)?;
                let eff_strength = state.effective_strength(now);
                let score = act_score * eff_strength;
                Some(RecallHit {
                    memory_id: id,
                    score,
                    semantic_score: 0.0,
                    ts_ms: payload.created_at_ms,
                    kind: payload.kind.clone(),
                    realm: payload.realm.clone(),
                    strength: eff_strength,
                    confidence: state.confidence,
                    access_count: state.access_count,
                    content: String::from_utf8(payload.content.clone()).unwrap_or_default(),
                    semantic_weight: 0.0,
                    status_mul: 0.0,
                    epistemic_mul: 0.0,
                    strength_factor: 0.0,
                    affect_valence: 0.0,
                    affect_arousal: 0.0,
                    actr_activation: 0.0,
                    surprise_boost: 1.0,
                    arousal_boost: 1.0,
                })
            })
            .collect();

        hits.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(limit);

        Ok(hits)
    }

    /// Register a file artifact, returning its ArtifactId (idempotent by path).
    pub fn upsert_artifact(
        &self,
        normalized_path: &str,
        repo_root: Option<String>,
    ) -> Result<ArtifactId> {
        // Fast path: already exists.
        if let Some(&id) = self.artifacts.read().get(normalized_path) {
            return Ok(id);
        }

        let artifact_id = self.artifact_id_alloc.next_id();
        let op = Op::UpsertArtifact(UpsertArtifactOp {
            artifact_id,
            normalized_path: normalized_path.to_string(),
            repo_root,
        });
        let _seqno = self.log.write().append(&op)?;

        // Double-checked insert: another thread may have raced past the read guard.
        let id = {
            let mut artifacts = self.artifacts.write();
            *artifacts
                .entry(normalized_path.to_string())
                .or_insert(artifact_id)
        };
        // Keep reverse map in sync.
        self.artifact_paths
            .write()
            .entry(id)
            .or_insert_with(|| normalized_path.to_string());

        Ok(id)
    }

    /// Recall memories within a time range [start_ms, end_ms].
    pub fn recall_temporal(
        &self,
        start_ms: i64,
        end_ms: i64,
        realm: Option<&str>,
        limit: usize,
    ) -> Result<Vec<RecallHit>> {
        let entries = self
            .time_idx
            .read()
            .range_query(start_ms, end_ms, realm, limit);
        let now = now_ms();
        let states = self.states.read();
        let payloads = self.payloads.read();

        let hits = entries
            .into_iter()
            .filter_map(|entry| {
                let state = states.get(&entry.memory_id)?;
                if state.deleted {
                    return None;
                }
                let payload = payloads.get(&entry.memory_id)?;
                let eff_strength = state.effective_strength(now);
                Some(RecallHit {
                    memory_id: entry.memory_id,
                    score: eff_strength * state.confidence,
                    semantic_score: 0.0,
                    ts_ms: payload.authored_at_ms,
                    kind: payload.kind.clone(),
                    realm: payload.realm.clone(),
                    strength: eff_strength,
                    confidence: state.confidence,
                    access_count: state.access_count,
                    content: String::from_utf8(payload.content.clone()).unwrap_or_default(),
                    semantic_weight: 0.0,
                    status_mul: 0.0,
                    epistemic_mul: 0.0,
                    strength_factor: 0.0,
                    affect_valence: 0.0,
                    affect_arousal: 0.0,
                    actr_activation: 0.0,
                    surprise_boost: 1.0,
                    arousal_boost: 1.0,
                })
            })
            .collect();

        Ok(hits)
    }

    /// Keyword (BM25) recall.
    pub fn recall_keyword(&self, query: &str, k: usize) -> Result<Vec<RecallHit>> {
        let keyword_hits = self.keyword_idx.read().search(query, k * 3);

        let now = now_ms();
        let states = self.states.read();
        let payloads = self.payloads.read();

        let mut hits: Vec<RecallHit> = keyword_hits
            .into_iter()
            .filter_map(|hit| {
                let state = states.get(&hit.memory_id)?;
                if state.deleted {
                    return None;
                }
                // soul:* realms are internal synthesis artifacts — exclude from unscoped recall.
                let payload = payloads.get(&hit.memory_id)?;
                if payload.realm.starts_with("soul:") {
                    return None;
                }
                let status_mul = status_score_multiplier(&state.status)?;
                let epistemic_mul = epistemic_score_multiplier(&state.epistemic_status);
                let kind_mul = realm_kind_multiplier(&payload.kind);
                let eff_strength = state.effective_strength(now);
                let strength_factor = 0.5 + 0.5 * eff_strength;
                // ACT-R + FEP + flashbulb (same as semantic path)
                let actr_activation = state.actr_base_level_activation(now);
                let actr_factor = 0.3 + 0.7 * actr_activation;
                let surprise_boost = 1.0 + 0.25 * state.surprise;
                let arousal_boost = 1.0 + 0.15 * state.affect_arousal;
                let base = hit.bm25_score * actr_factor * state.confidence
                    * surprise_boost * arousal_boost;
                Some(RecallHit {
                    memory_id: hit.memory_id,
                    score: base * status_mul * epistemic_mul * kind_mul,
                    semantic_score: 0.0,
                    ts_ms: payload.authored_at_ms,
                    kind: payload.kind.clone(),
                    realm: payload.realm.clone(),
                    strength: eff_strength,
                    confidence: state.confidence,
                    access_count: state.access_count,
                    content: String::from_utf8(payload.content.clone()).unwrap_or_default(),
                    semantic_weight: hit.bm25_score,
                    status_mul,
                    epistemic_mul,
                    strength_factor,
                    affect_valence: state.affect_valence,
                    affect_arousal: state.affect_arousal,
                    actr_activation,
                    surprise_boost,
                    arousal_boost,
                })
            })
            .collect();

        hits.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);

        let hit_ids: Vec<MemoryId> = hits.iter().map(|h| h.memory_id).collect();
        drop(states);
        drop(payloads);
        self.enqueue_recall_effects(&hit_ids);

        Ok(hits)
    }

    pub fn recall_with_fallback(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        k: usize,
        realm: Option<&str>,
    ) -> Result<Vec<RecallHit>> {
        let hits = self.recall_semantic(query_embedding, k, realm)?;
        if !hits.is_empty() {
            return Ok(hits);
        }

        let store_size = self.memory_count();
        if store_size < 10 {
            return Ok(vec![]);
        }

        log::warn!(
            "recall_semantic returned empty with {} memories, falling back to BM25",
            store_size
        );
        let bm25_hits = self.recall_keyword(query_text, k)?;
        if !bm25_hits.is_empty() {
            return Ok(bm25_hits);
        }

        log::warn!("BM25 fallback also empty, falling back to recency");
        let now = now_ms();
        self.recall_temporal(0, now, realm, k)
    }

    /// Recall memories associated with a file path (exact match).
    pub fn recall_artifact(&self, path: &str, limit: usize) -> Result<Vec<RecallHit>> {
        let entries = self.artifact_idx.read().query_path(path, limit);
        let now = now_ms();
        let states = self.states.read();
        let payloads = self.payloads.read();

        let hits = entries
            .into_iter()
            .filter_map(|entry| {
                let state = states.get(&entry.memory_id)?;
                if state.deleted {
                    return None;
                }
                let payload = payloads.get(&entry.memory_id)?;
                let eff_strength = state.effective_strength(now);
                Some(RecallHit {
                    memory_id: entry.memory_id,
                    score: entry.strength * eff_strength * state.confidence,
                    semantic_score: 0.0,
                    ts_ms: payload.authored_at_ms,
                    kind: payload.kind.clone(),
                    realm: payload.realm.clone(),
                    strength: eff_strength,
                    confidence: state.confidence,
                    access_count: state.access_count,
                    content: String::from_utf8(payload.content.clone()).unwrap_or_default(),
                    semantic_weight: 0.0,
                    status_mul: 0.0,
                    epistemic_mul: 0.0,
                    strength_factor: 0.0,
                    affect_valence: 0.0,
                    affect_arousal: 0.0,
                    actr_activation: 0.0,
                    surprise_boost: 1.0,
                    arousal_boost: 1.0,
                })
            })
            .collect();

        Ok(hits)
    }

    /// Add a triplet fact. Returns the triplet ID.
    pub fn add_triplet(
        &self,
        subject: String,
        predicate: String,
        object: String,
        weight: f32,
        source_memory_id: Option<MemoryId>,
        source_file: Option<String>,
    ) -> Result<u64> {
        let triplet_id = self.triplet_id_alloc.next_id();
        let valid_from_ms = now_ms();

        let op = Op::AddTriplet(AddTripletOp {
            triplet_id,
            subject: subject.clone(),
            predicate: predicate.clone(),
            object: object.clone(),
            weight,
            valid_from_ms,
            source_memory_id,
            source_file: source_file.clone(),
        });
        let _seqno = self.log.write().append(&op)?;

        self.triplet_store.write().replay_add(
            triplet_id,
            subject,
            predicate,
            object,
            weight,
            valid_from_ms,
            source_memory_id,
            source_file,
        );

        Ok(triplet_id)
    }

    /// Set the lifecycle status of a memory (Active/Superseded/Contradicted/Archived).
    /// Durable: writes UpdateState op to WAL.
    pub fn set_memory_status(&self, memory_id: MemoryId, status: crate::state::MemoryStatus) -> Result<()> {
        use crate::state::MemoryStatus;
        let status_u8: u8 = match status {
            MemoryStatus::Active       => 0,
            MemoryStatus::Superseded   => 1,
            MemoryStatus::Contradicted => 2,
            MemoryStatus::Archived     => 3,
            MemoryStatus::Proposed     => 4,
            MemoryStatus::Observed     => 5,
            MemoryStatus::Verified     => 6,
        };
        // Check existence before writing to WAL
        {
            let states = self.states.read();
            if !states.contains_key(&memory_id) {
                return Err(FieldError::NotFound(memory_id));
            }
        }
        let delta = crate::ops::StateDeltaOp {
            memory_id,
            strength_delta: None,
            confidence_delta: None,
            decay_rate: None,
            touch: false,
            pin: None,
            op_ts_ms: now_ms(),
            status: Some(status_u8),
            epistemic_status: None,
        };
        self.log.write().append(&Op::UpdateState(delta.clone()))?;
        let _ = self.log.write().sync(); // status transitions are critical lifecycle events
        let mut states = self.states.write();
        if let Some(st) = states.get_mut(&memory_id) {
            st.apply_delta(&delta, now_ms());
            Ok(())
        } else {
            Err(FieldError::NotFound(memory_id))
        }
    }

    /// Set the epistemic status of a memory (UserStated/ToolDerived/ModelInferred/AutonomousSynthesis).
    /// Durable: writes UpdateState op to WAL.
    pub fn set_epistemic_status(&self, memory_id: MemoryId, es: crate::state::EpistemicStatus) -> Result<()> {
        use crate::state::EpistemicStatus;
        let es_u8: u8 = match es {
            EpistemicStatus::UserStated          => 0,
            EpistemicStatus::ToolDerived         => 1,
            EpistemicStatus::ModelInferred       => 2,
            EpistemicStatus::AutonomousSynthesis => 3,
        };
        // Check existence before writing to WAL
        {
            let states = self.states.read();
            if !states.contains_key(&memory_id) {
                return Err(FieldError::NotFound(memory_id));
            }
        }
        let delta = crate::ops::StateDeltaOp {
            memory_id,
            strength_delta: None,
            confidence_delta: None,
            decay_rate: None,
            touch: false,
            pin: None,
            op_ts_ms: now_ms(),
            status: None,
            epistemic_status: Some(es_u8),
        };
        self.log.write().append(&Op::UpdateState(delta.clone()))?;
        let _ = self.log.write().sync();
        let mut states = self.states.write();
        if let Some(st) = states.get_mut(&memory_id) {
            st.apply_delta(&delta, now_ms());
            Ok(())
        } else {
            Err(FieldError::NotFound(memory_id))
        }
    }

    /// Set affect dimensions on a memory (valence: -1..+1, arousal: 0..1).
    /// In-memory only (not WAL-persisted) — affect is re-derived from content on reload.
    pub fn set_affect(&self, memory_id: MemoryId, valence: f32, arousal: f32) -> Result<()> {
        let mut states = self.states.write();
        if let Some(st) = states.get_mut(&memory_id) {
            st.affect_valence = valence.clamp(-1.0, 1.0);
            st.affect_arousal = arousal.clamp(0.0, 1.0);
            Ok(())
        } else {
            Err(FieldError::NotFound(memory_id))
        }
    }

    /// Invalidate a triplet (marks it as expired at the current time).
    /// Backfill embedding for a memory stored with embed_pending=true.
    /// Durable: writes UpdateMemoryContent op to WAL (content unchanged, new embedding).
    pub fn backfill_embedding(&self, memory_id: MemoryId, embedding: &[f32]) -> Result<()> {
        if embedding.len() != EMBED_DIM {
            return Err(FieldError::InvalidEmbedDim { expected: EMBED_DIM, actual: embedding.len() });
        }
        let existing_content = {
            let states = self.states.read();
            match states.get(&memory_id) {
                None => return Err(FieldError::NotFound(memory_id)),
                Some(st) if !st.embed_pending => return Ok(()),
                Some(_) => {
                    self.payloads.read().get(&memory_id)
                        .map(|p| p.content.clone())
                        .unwrap_or_default()
                }
            }
        };

        // Persist to WAL via UpdateMemoryContent (empty content = reuse existing)
        let op = Op::UpdateMemoryContent(crate::ops::UpdateMemoryContentOp {
            memory_id,
            content: existing_content.clone(),
            embedding: embedding.to_vec(),
        });
        self.log.write().append(&op)?;

        // Update payload embedding in memory
        {
            let mut payloads = self.payloads.write();
            if let Some(p) = payloads.get_mut(&memory_id) {
                p.embedding = embedding.to_vec();
                p.embedding_model = "bge-base-en-v1.5".to_string();
            }
        }

        // Update semantic index
        self.semantic_idx.write().upsert(memory_id, embedding.to_vec());

        // Re-encode cortical sparse index (non-fatal)
        let _ = self.encode_memory(memory_id);

        // Clear embed_pending in state
        {
            let mut states = self.states.write();
            if let Some(st) = states.get_mut(&memory_id) {
                st.embed_pending = false;
            }
        }
        Ok(())
    }

    /// Return memory IDs with embed_pending=true, sorted oldest first, up to limit.
    pub fn pending_embeddings(&self, limit: usize) -> Vec<MemoryId> {
        let s = self.states.read();
        let mut pending: Vec<(i64, MemoryId)> = s
            .iter()
            .filter(|(_, st)| st.embed_pending && !st.deleted)
            .map(|(id, st)| (st.created_at_ms, *id))
            .collect();
        pending.sort_by_key(|(ts, _)| *ts);
        pending.into_iter().take(limit).map(|(_, id)| id).collect()
    }

    /// Remove triplet by subject+predicate+object (invalidates first matching entry).
    pub fn forget_triplet(&self, subject: &str, predicate: &str, object: &str) -> Result<bool> {
        let at_ms = now_ms();
        let store = self.triplet_store.read();
        let matches: Vec<u64> = store
            .query_subject(subject, at_ms)
            .into_iter()
            .filter(|t| t.predicate == predicate && t.object == object)
            .map(|t| t.id)
            .collect();
        drop(store);
        for id in &matches {
            self.invalidate_triplet(*id)?;
        }
        Ok(!matches.is_empty())
    }

    pub fn invalidate_triplet(&self, triplet_id: u64) -> Result<()> {
        let now = now_ms();
        let op = Op::InvalidateTriplet(InvalidateTripletOp {
            triplet_id,
            invalidated_at_ms: now,
        });
        let _seqno = self.log.write().append(&op)?;

        self.triplet_store.write().invalidate(triplet_id, now);

        Ok(())
    }

    /// Query all currently-valid triplets with the given subject.
    pub fn query_subject(&self, subject: &str) -> Result<Vec<TripletEntry>> {
        let at_ms = now_ms();
        let store = self.triplet_store.read();
        Ok(store
            .query_subject(subject, at_ms)
            .into_iter()
            .cloned()
            .collect())
    }

    /// Query all currently-valid triplets with the given object.
    pub fn query_object(&self, object: &str) -> Result<Vec<TripletEntry>> {
        let at_ms = now_ms();
        let store = self.triplet_store.read();
        Ok(store
            .query_object(object, at_ms)
            .into_iter()
            .cloned()
            .collect())
    }

    /// Query all currently-valid triplets where subject OR object matches.
    pub fn query_entity(&self, entity: &str) -> Result<Vec<TripletEntry>> {
        let at_ms = now_ms();
        let store = self.triplet_store.read();
        Ok(store
            .query_entity(entity, at_ms)
            .into_iter()
            .cloned()
            .collect())
    }

    /// Get memory IDs that contradict the given memory (bidirectional).
    pub fn get_conflicts(&self, memory_id: MemoryId) -> Result<Vec<MemoryId>> {
        let id_str = memory_id.to_string();
        let at_ms = now_ms();
        let store = self.triplet_store.read();
        let mut result = Vec::new();
        for entry in store.query_subject(&id_str, at_ms) {
            if entry.predicate == "contradicts" {
                if let Ok(id) = entry.object.parse::<u64>() {
                    result.push(id);
                }
            }
        }
        for entry in store.query_object(&id_str, at_ms) {
            if entry.predicate == "contradicts" {
                if let Ok(id) = entry.subject.parse::<u64>() {
                    result.push(id);
                }
            }
        }
        Ok(result)
    }

    /// Follow "supersedes" edges to build the full supersession chain.
    /// Chain starts with memory_id itself. Max depth 20, cycle-safe.
    pub fn get_supersession_chain(&self, memory_id: MemoryId) -> Result<Vec<MemoryId>> {
        let at_ms = now_ms();
        let store = self.triplet_store.read();
        let mut chain = vec![memory_id];
        let mut visited = std::collections::HashSet::new();
        visited.insert(memory_id);
        let mut current = memory_id;
        for _ in 0..20 {
            let id_str = current.to_string();
            let next = store
                .query_object(&id_str, at_ms)
                .into_iter()
                .find(|e| e.predicate == "supersedes")
                .and_then(|e| e.subject.parse::<u64>().ok());
            match next {
                Some(n) if !visited.contains(&n) => {
                    visited.insert(n);
                    chain.push(n);
                    current = n;
                }
                _ => break,
            }
        }
        Ok(chain)
    }

    /// Get memory IDs that confirm the given memory.
    pub fn get_confirmations(&self, memory_id: MemoryId) -> Result<Vec<MemoryId>> {
        let id_str = memory_id.to_string();
        let at_ms = now_ms();
        let store = self.triplet_store.read();
        let mut result = Vec::new();
        for entry in store.query_object(&id_str, at_ms) {
            if entry.predicate == "confirms" {
                if let Ok(id) = entry.subject.parse::<u64>() {
                    result.push(id);
                }
            }
        }
        Ok(result)
    }

    /// Apply feedback for a recall episode (route learning).
    pub fn feedback(&self, episode_id: u64, reward: f32) -> Result<()> {
        self.learners.write().route.feedback(episode_id, reward);
        Ok(())
    }

    /// Get recommended window size for a session type.
    pub fn recommended_window(&self, session_type: &str) -> usize {
        self.learners
            .read()
            .context
            .recommended_window(session_type)
    }

    /// Record context outcome for a session type and window size.
    pub fn record_context_outcome(&self, session_type: &str, size: usize, outcome: f32) {
        self.learners
            .write()
            .context
            .record_outcome(session_type, size, outcome);
    }

    /// Select a retrieval route using Thompson sampling. Returns (episode_id, route).
    pub fn select_route(&self, query: &str) -> (u64, Route) {
        let intent = RouteLearner::detect_intent(query);
        let now_ms = now_ms() as u64;
        self.learners.write().route.select_route(intent, now_ms)
    }

    // ── Code Intelligence ────────────────────────────────────────────────────

    /// Upsert a symbol. Deduplicates by (kind, name, file_path, line_start).
    /// Returns the SymbolId.
    pub fn upsert_symbol(
        &self,
        kind: &str,
        name: &str,
        signature: &str,
        file_path: &str,
        line_start: u32,
        line_end: u32,
        repo_id: u64,
        embedding: &[f32],
        description: Option<String>,
        memory_id: Option<MemoryId>,
    ) -> Result<u64> {
        let symbol_id = self.symbol_id_alloc.next_id();
        let op = Op::UpsertSymbol(UpsertSymbolOp {
            symbol_id,
            kind: kind.to_string(),
            name: name.to_string(),
            signature: signature.to_string(),
            file_path: file_path.to_string(),
            line_start,
            line_end,
            repo_id,
            embedding: embedding.to_vec(),
            description: description.clone(),
            memory_id,
        });
        let _seqno = self.log.write().append(&op)?;

        let entry = SymbolEntry {
            id: symbol_id,
            kind: kind.to_string(),
            name: name.to_string(),
            signature: signature.to_string(),
            file_path: file_path.to_string(),
            line_start,
            line_end,
            repo_id,
            embedding: embedding.to_vec(),
            description,
            memory_id,
        };
        let actual_id = self.symbol_idx.write().upsert(entry);
        Ok(actual_id)
    }

    /// Remove a symbol and all its call edges.
    pub fn remove_symbol(&self, symbol_id: u64) -> Result<()> {
        let op = Op::RemoveSymbol(RemoveSymbolOp { symbol_id });
        let _seqno = self.log.write().append(&op)?;

        self.symbol_idx.write().remove(symbol_id);
        self.call_graph.write().remove_symbol(symbol_id);
        Ok(())
    }

    /// Get a symbol by ID.
    pub fn get_symbol(&self, symbol_id: u64) -> Result<Option<SymbolEntry>> {
        Ok(self.symbol_idx.read().get(symbol_id).cloned())
    }

    /// Search symbols by name (exact or prefix match).
    pub fn search_symbols_by_name(&self, query: &str, limit: usize) -> Vec<SymbolEntry> {
        self.symbol_idx
            .read()
            .search_by_name(query, limit)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Semantic symbol search: find k nearest by cosine similarity.
    pub fn search_symbols_semantic(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        self.symbol_idx.read().search_semantic(query, k)
    }

    /// Get all symbols in a file.
    pub fn symbols_in_file(&self, file_path: &str) -> Vec<SymbolEntry> {
        self.symbol_idx
            .read()
            .by_file(file_path)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Add a call edge between two symbols (idempotent).
    pub fn add_call_edge(&self, caller_id: u64, callee_id: u64) -> Result<()> {
        let op = Op::AddSymCallEdge(AddSymCallEdgeOp {
            caller_id,
            callee_id,
        });
        let _seqno = self.log.write().append(&op)?;
        self.call_graph.write().add_edge(caller_id, callee_id);
        Ok(())
    }

    /// Get symbols called by the given symbol.
    pub fn get_callees(&self, symbol_id: u64) -> Vec<u64> {
        self.call_graph.read().get_callees(symbol_id)
    }

    /// Get symbols that call the given symbol.
    pub fn get_callers(&self, symbol_id: u64) -> Vec<u64> {
        self.call_graph.read().get_callers(symbol_id)
    }

    /// Upsert a code file record. Returns its CodeFileId.
    pub fn upsert_code_file(&self, path: &str, project: &str, mtime: i64) -> Result<u64> {
        let next_id_fn = || self.code_file_id_alloc.next_id();
        let (file_id, _was_updated) = self
            .code_files
            .write()
            .upsert(path, project, mtime, next_id_fn);
        let op = Op::UpsertCodeFile(UpsertCodeFileOp {
            file_id,
            path: path.to_string(),
            project: project.to_string(),
            mtime,
        });
        let _seqno = self.log.write().append(&op)?;
        Ok(file_id)
    }

    pub fn symbol_count(&self) -> usize {
        self.symbol_idx.read().count()
    }

    pub fn code_file_count(&self) -> usize {
        self.code_files.read().count()
    }

    pub fn cortical_count(&self) -> usize {
        self.cortical_idx.read().len()
    }

    pub fn prototype_count(&self) -> usize {
        self.cortical_idx.read().prototype_count()
    }

    /// Per-realm embedding geometry stats (inspired by "Geometry of Forgetting").
    /// Returns JSON: `{"by_realm": [...], "by_kind": [...], "anomalies": [...]}`
    pub fn spectral_stats_by_realm(&self) -> String {
        let realm_members = self.realm_members.read();
        let semantic_idx = self.semantic_idx.read();
        let payloads = self.payloads.read();
        let states = self.states.read();

        // By realm
        let mut realm_results: Vec<serde_json::Value> = Vec::new();
        let mut realms: Vec<&String> = realm_members.keys().collect();
        realms.sort();
        for realm in realms {
            let ids = match realm_members.get(realm.as_str()) {
                Some(ids) if ids.len() >= 2 => ids,
                _ => continue,
            };
            let embeddings: Vec<&[f32]> = ids
                .iter()
                .filter_map(|id| semantic_idx.get_embedding(*id))
                .collect();
            if let Some(stats) = compute_geometry(&embeddings, realm) {
                realm_results.push(stats);
            }
        }

        // By kind
        let mut kind_ids: std::collections::HashMap<&str, Vec<MemoryId>> =
            std::collections::HashMap::new();
        for (mid, payload) in payloads.iter() {
            if states.get(mid).map(|s| s.deleted).unwrap_or(true) {
                continue;
            }
            kind_ids.entry(&payload.kind).or_default().push(*mid);
        }
        let mut kind_results: Vec<serde_json::Value> = Vec::new();
        let mut kinds: Vec<&&str> = kind_ids.keys().collect();
        kinds.sort();
        for kind in kinds {
            let ids = &kind_ids[*kind];
            if ids.len() < 2 {
                continue;
            }
            let embeddings: Vec<&[f32]> = ids
                .iter()
                .filter_map(|id| semantic_idx.get_embedding(*id))
                .collect();
            if let Some(stats) = compute_geometry(&embeddings, *kind) {
                kind_results.push(stats);
            }
        }

        // Anomalies
        let mut anomalies: Vec<serde_json::Value> = Vec::new();
        for entry in realm_results.iter().chain(kind_results.iter()) {
            let label = entry["group"].as_str().unwrap_or("?");
            let cos = entry["mean_cosine_sim"].as_f64().unwrap_or(0.0);
            let iso = entry["isotropy"].as_f64().unwrap_or(1.0);
            let count = entry["count"].as_u64().unwrap_or(0);
            let has_newline = label.contains('\n') || label.contains('\r');
            if cos > 0.95 && count >= 5 {
                anomalies.push(serde_json::json!({
                    "group": label, "issue": "high_similarity",
                    "detail": format!("cos={:.3} across {} memories — likely duplicates", cos, count)
                }));
            }
            if iso < 0.3 && count >= 5 {
                anomalies.push(serde_json::json!({
                    "group": label, "issue": "collapsed_embeddings",
                    "detail": format!("isotropy={:.3} — embeddings occupy narrow subspace", iso)
                }));
            }
            if has_newline {
                anomalies.push(serde_json::json!({
                    "group": label.trim(), "issue": "dirty_realm_name",
                    "detail": "realm contains trailing whitespace/newline"
                }));
            }
        }

        serde_json::to_string(&serde_json::json!({
            "by_realm": realm_results,
            "by_kind": kind_results,
            "anomalies": anomalies,
        }))
        .unwrap_or_else(|_| "{}".to_string())
    }

    /// Fix realm names that contain trailing whitespace/newlines.
    /// Returns the number of memories whose realm was trimmed.
    pub fn trim_realm_names(&self) -> usize {
        // Collect dirty memories: (memory_id, old_realm, trimmed_realm)
        let dirty: Vec<(MemoryId, String, String)> = {
            let payloads = self.payloads.read();
            let states = self.states.read();
            payloads
                .iter()
                .filter_map(|(mid, p)| {
                    if states.get(mid).map(|s| s.deleted).unwrap_or(true) {
                        return None;
                    }
                    let trimmed = p.realm.trim().to_string();
                    if trimmed != p.realm {
                        Some((*mid, p.realm.clone(), trimmed))
                    } else {
                        None
                    }
                })
                .collect()
        };

        let count = dirty.len();
        for (mid, old_realm, new_realm) in dirty {
            // Update payload realm
            if let Some(p) = self.payloads.write().get_mut(&mid) {
                p.realm = new_realm.clone();
            }
            // Update realm_members index
            let mut rm = self.realm_members.write();
            if let Some(set) = rm.get_mut(&old_realm) {
                set.remove(&mid);
                if set.is_empty() {
                    rm.remove(&old_realm);
                }
            }
            rm.entry(new_realm).or_default().insert(mid);
        }
        count
    }

    /// Save a spectral stats snapshot for temporal drift tracking.
    /// Writes `spectral_snapshot_{timestamp}.json` to the data dir.
    pub fn save_spectral_snapshot(&self) -> Result<String> {
        let stats_json = self.spectral_stats_by_realm();
        let ts = now_ms();
        let filename = format!("spectral_snapshot_{}.json", ts);
        let path = self.data_dir.join(&filename);
        let wrapped = serde_json::json!({
            "ts_ms": ts,
            "stats": serde_json::from_str::<serde_json::Value>(&stats_json).unwrap_or_default(),
        });
        let content = serde_json::to_string_pretty(&wrapped)
            .map_err(|e| FieldError::Serialization(e.to_string()))?;
        std::fs::write(&path, content)
            .map_err(|e| FieldError::Io(e))?;
        Ok(filename)
    }

    /// Load spectral drift: compare current stats with most recent snapshot.
    /// Returns JSON with per-realm/kind delta for isotropy and mean_cosine_sim.
    pub fn spectral_drift(&self) -> String {
        // Find most recent snapshot
        let entries = match std::fs::read_dir(&self.data_dir) {
            Ok(e) => e,
            Err(_) => return "{}".to_string(),
        };
        let mut snapshots: Vec<(i64, std::path::PathBuf)> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                if name.starts_with("spectral_snapshot_") && name.ends_with(".json") {
                    let ts_str = name
                        .strip_prefix("spectral_snapshot_")?
                        .strip_suffix(".json")?;
                    let ts: i64 = ts_str.parse().ok()?;
                    Some((ts, e.path()))
                } else {
                    None
                }
            })
            .collect();
        snapshots.sort_by_key(|(ts, _)| -*ts);

        let prev_snap = match snapshots.first() {
            Some((_, path)) => {
                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => return "{}".to_string(),
                };
                match serde_json::from_str::<serde_json::Value>(&content) {
                    Ok(v) => v,
                    Err(_) => return "{}".to_string(),
                }
            }
            None => return serde_json::json!({"error": "no previous snapshot"}).to_string(),
        };

        let prev_ts = prev_snap["ts_ms"].as_i64().unwrap_or(0);
        let prev_stats = &prev_snap["stats"];

        // Current stats
        let current_json = self.spectral_stats_by_realm();
        let current: serde_json::Value =
            serde_json::from_str(&current_json).unwrap_or_default();

        let mut drifts: Vec<serde_json::Value> = Vec::new();

        for section in &["by_realm", "by_kind"] {
            let prev_arr = prev_stats[section].as_array();
            let curr_arr = current[section].as_array();
            if let (Some(prev_items), Some(curr_items)) = (prev_arr, curr_arr) {
                let prev_map: std::collections::HashMap<&str, &serde_json::Value> = prev_items
                    .iter()
                    .filter_map(|v| v["group"].as_str().map(|g| (g, v)))
                    .collect();
                for curr in curr_items {
                    let group = match curr["group"].as_str() {
                        Some(g) => g,
                        None => continue,
                    };
                    if let Some(prev) = prev_map.get(group) {
                        let iso_prev = prev["isotropy"].as_f64().unwrap_or(0.0);
                        let iso_curr = curr["isotropy"].as_f64().unwrap_or(0.0);
                        let cos_prev = prev["mean_cosine_sim"].as_f64().unwrap_or(0.0);
                        let cos_curr = curr["mean_cosine_sim"].as_f64().unwrap_or(0.0);
                        let iso_delta = iso_curr - iso_prev;
                        let cos_delta = cos_curr - cos_prev;
                        if iso_delta.abs() > 0.005 || cos_delta.abs() > 0.005 {
                            drifts.push(serde_json::json!({
                                "section": section,
                                "group": group,
                                "isotropy_delta": (iso_delta * 1000.0).round() / 1000.0,
                                "cosine_delta": (cos_delta * 1000.0).round() / 1000.0,
                                "isotropy_now": iso_curr,
                                "cosine_now": cos_curr,
                            }));
                        }
                    }
                }
            }
        }

        let hours_since = (now_ms() - prev_ts) as f64 / 3_600_000.0;
        serde_json::to_string(&serde_json::json!({
            "snapshot_age_hours": (hours_since * 10.0).round() / 10.0,
            "drifts": drifts,
            "total_drifted": drifts.len(),
        }))
        .unwrap_or_else(|_| "{}".to_string())
    }

    /// Encode a memory's embedding into sparse codes and index into the cortical index.
    /// Persists via UpdateSparseCode op.
    pub fn encode_memory(&self, memory_id: MemoryId) -> Result<()> {
        let embedding = {
            let payloads = self.payloads.read();
            payloads.get(&memory_id).map(|p| p.embedding.clone())
        };
        let Some(embedding) = embedding else {
            return Ok(());
        };
        if embedding.len() != EMBED_DIM {
            return Ok(());
        }

        let encoder = self.sparse_encoder.read();
        let code = encoder.encode(&embedding);
        if code.is_empty() {
            return Ok(());
        }

        // Compute surprise (reconstruction error) before updating encoder
        let surprise = encoder.reconstruction_error(&embedding, &code);
        drop(encoder);

        // Update surprise in memory state and plasticity learner
        {
            let mut states = self.states.write();
            if let Some(state) = states.get_mut(&memory_id) {
                state.surprise = surprise;
            }
            let mut learners = self.learners.write();
            learners.plasticity.update_surprise(memory_id, surprise);
        }

        // FEP-derived update (accuracy + complexity + orthogonalization)
        self.sparse_encoder.write().update(&embedding, &code);

        let ts_ms = now_ms();
        let op = Op::UpdateSparseCode(UpdateSparseCodeOp {
            memory_id,
            feature_ids: code.feature_ids.clone(),
            activations: code.activations.clone(),
            ts_ms,
        });
        self.log.write().append(&op)?;

        // Index in cortical index
        let strength = self
            .states
            .read()
            .get(&memory_id)
            .map(|s| s.strength)
            .unwrap_or(0.5);
        let kind = self
            .payloads
            .read()
            .get(&memory_id)
            .map(|p| p.kind.clone())
            .unwrap_or_default();
        let authored_at = self
            .payloads
            .read()
            .get(&memory_id)
            .map(|p| p.authored_at_ms)
            .unwrap_or(ts_ms);
        self.cortical_idx
            .write()
            .index(memory_id, &code, strength, authored_at, &kind);

        Ok(())
    }

    /// Save the cortical index + encoder + prototype state to a binary snapshot.
    /// After this, on next open the snapshot covers all UpdateSparseCode ops
    /// up to the current log position, so those ops can be skipped in replay.
    pub fn save_snapshot(&self) -> Result<()> {
        self.drain_pending_recall_effects()?;
        let seqno = self.log.read().last_seqno();
        let path = self
            .data_dir
            .join(format!("cortex.{:08x}.snapshot", self.instance_id));
        self.cortical_idx.read().save_snapshot(&path, seqno)
    }

    /// Save full in-memory state to a binary snapshot (chitta.snapshot).
    /// After this, on next open only ops after snapshot_seqno need to be replayed.
    pub fn save_full_snapshot(&self) -> Result<()> {
        use crate::snapshot::FullSnapshot;
        self.drain_pending_recall_effects()?;
        let seqno = self.log.read().last_seqno();
        let snap = FullSnapshot {
            snapshot_seqno: seqno,
            payloads: self.payloads.read().clone(),
            states: self.states.read().clone(),
            assoc_edges: self.assoc_edges.read().clone(),
            artifacts: self.artifacts.read().clone(),
            artifact_paths: self.artifact_paths.read().clone(),
            time_idx: self.time_idx.read().clone(),
            keyword_idx: self.keyword_idx.read().clone(),
            artifact_idx: self.artifact_idx.read().clone(),
            triplet_store: self.triplet_store.read().clone(),
            symbol_idx: self.symbol_idx.read().clone(),
            call_graph: self.call_graph.read().clone(),
            code_files: self.code_files.read().clone(),
            semantic_idx: self.semantic_idx.read().clone(),
            coactivation_stats: self.coactivation_stats.read().clone(),
        };
        let path = self
            .data_dir
            .join(format!("chitta.{:08x}.snapshot", self.instance_id));
        snap.save(&path)
    }


    /// Compact WAL: save full snapshot then delete WAL segments covered by it.
    /// After compaction, only segments with seqno >= snapshot_seqno are kept.
    /// This bounds WAL growth and speeds up startup replay.
    pub fn compact_wal(&self) -> Result<usize> {
        let count = {
            let states = self.states.read();
            states.values().filter(|s| !s.deleted).count()
        };
        if count < 100 {
            return Err(FieldError::Other(format!(
                "refusing compact_wal on near-empty store ({} live memories, minimum 100)", count
            )));
        }
        self.save_full_snapshot()?;
        let snapshot_seqno = self.log.read().last_seqno();
        
        let seg_dir = self.data_dir.join("segments");
        let mut deleted = 0usize;
        
        if let Ok(entries) = std::fs::read_dir(&seg_dir) {
            let mut paths: Vec<std::path::PathBuf> = entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().map(|e| e == "seg").unwrap_or(false))
                .collect();
            paths.sort();
            
            // Keep the last segment (may be partially covered) — delete only fully-covered ones
            // A segment is covered if its max seqno < snapshot_seqno
            // Since seqno is in the filename ({instance_id}_{first_seqno}.seg), we can use it
            for path in &paths[..paths.len().saturating_sub(1)] {
                // Extract first_seqno from filename
                let fname = path.file_stem().and_then(|f| f.to_str()).unwrap_or("");
                let first_seqno: u64 = fname.split('_').nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(u64::MAX);
                if first_seqno < snapshot_seqno {
                    if std::fs::remove_file(path).is_ok() {
                        deleted += 1;
                    }
                }
            }
        }
        Ok(deleted)
    }

    /// Run a single tier demotion pass over all memories.
    /// Returns `(demoted_count, deleted_count)`.
    ///
    /// Tiers: 0=L1 (hippocampus), 1=L2 (cortex), 2=L3 (archive), then delete.
    /// Uses `access_count` as rehearsal proxy and `strength` as utility proxy.
    pub fn run_demotion_pass(&self, now_ms: i64) -> Result<(usize, usize)> {
        const L1_TO_L2_AGE_MS: i64 = 7 * 24 * 3600 * 1000;
        const L1_TO_L2_LAST_ACCESS_MS: i64 = 2 * 24 * 3600 * 1000;
        const L1_TO_L2_MAX_STRENGTH: f32 = 0.80;

        const L2_TO_L3_AGE_BASE_MS: i64 = 45 * 24 * 3600 * 1000;
        const L2_TO_L3_REHEARSAL_BONUS_MS: i64 = 7 * 24 * 3600 * 1000;
        const L2_TO_L3_LAST_ACCESS_MS: i64 = 14 * 24 * 3600 * 1000;
        const L2_TO_L3_MAX_STRENGTH: f32 = 0.50;

        const L3_DELETE_AGE_BASE_MS: i64 = 365 * 24 * 3600 * 1000;
        const L3_DELETE_REHEARSAL_BONUS_MS: i64 = 30 * 24 * 3600 * 1000;
        const L3_DELETE_LAST_ACCESS_MS: i64 = 120 * 24 * 3600 * 1000;
        const L3_DELETE_MAX_STRENGTH: f32 = 0.12;
        const L3_DELETE_MAX_UTILITY: f32 = 0.80;

        let mut to_demote: Vec<(MemoryId, u8)> = Vec::new();
        let mut to_delete: Vec<MemoryId> = Vec::new();

        {
            let states = self.states.read();
            for (&memory_id, state) in states.iter() {
                if state.deleted || state.pinned {
                    continue;
                }

                // Strength >= L3_DELETE_MAX_UTILITY means never delete
                let age_ms = now_ms - state.created_at_ms;
                let last_access_ago = now_ms - state.last_accessed_ms;
                // access_count serves as rehearsal proxy; cap at 8 for bonus calc
                let rehearsal = state.access_count.min(8) as i64;

                match state.tier {
                    0 => {
                        // L1 → L2
                        if age_ms >= L1_TO_L2_AGE_MS
                            && last_access_ago >= L1_TO_L2_LAST_ACCESS_MS
                            && state.strength < L1_TO_L2_MAX_STRENGTH
                        {
                            to_demote.push((memory_id, 1));
                        }
                    }
                    1 => {
                        // L2 → L3
                        let threshold =
                            L2_TO_L3_AGE_BASE_MS + rehearsal * L2_TO_L3_REHEARSAL_BONUS_MS;
                        if age_ms >= threshold
                            && last_access_ago >= L2_TO_L3_LAST_ACCESS_MS
                            && state.strength < L2_TO_L3_MAX_STRENGTH
                        {
                            to_demote.push((memory_id, 2));
                        }
                    }
                    2 => {
                        // L3 → delete
                        let threshold =
                            L3_DELETE_AGE_BASE_MS + rehearsal * L3_DELETE_REHEARSAL_BONUS_MS;
                        if age_ms >= threshold
                            && last_access_ago >= L3_DELETE_LAST_ACCESS_MS
                            && state.strength < L3_DELETE_MAX_STRENGTH
                            && state.strength < L3_DELETE_MAX_UTILITY
                        {
                            to_delete.push(memory_id);
                        }
                    }
                    _ => {}
                }
            }
        }

        let demoted = to_demote.len();
        let deleted = to_delete.len();

        for (id, new_tier) in to_demote {
            let op = Op::DemoteMemory(DemoteMemoryOp {
                memory_id: id,
                new_tier,
            });
            self.log.write().append(&op)?;
            if let Some(state) = self.states.write().get_mut(&id) {
                state.tier = new_tier;
            }
        }
        for id in to_delete {
            self.forget(id)?;
        }

        Ok((demoted, deleted))
    }

    /// Encode all memories that don't yet have a sparse code.
    pub fn encode_all_unindexed(&self) -> Result<usize> {
        let ids: Vec<MemoryId> = {
            let payloads = self.payloads.read();
            let cortical = self.cortical_idx.read();
            payloads
                .keys()
                .filter(|id| !cortical.mem_codes.contains_key(id))
                .copied()
                .collect()
        };
        let count = ids.len();
        for id in ids {
            self.encode_memory(id)?;
        }
        Ok(count)
    }

    /// Train a ProductQuantizer from the residuals of all encoded memories.
    /// Requires at least 256 memories with sparse codes.
    pub fn train_pq(&self) -> Result<()> {
        // Collect residuals: for each memory with a sparse code, decode and subtract
        let residuals: Vec<Vec<f32>> = {
            let payloads = self.payloads.read();
            let cortical = self.cortical_idx.read();
            let encoder = self.sparse_encoder.read();

            cortical
                .mem_codes
                .iter()
                .filter_map(|(&memory_id, code)| {
                    let embedding = payloads.get(&memory_id).map(|p| p.embedding.clone())?;
                    if embedding.len() != crate::ops::EMBED_DIM {
                        return None;
                    }
                    let decoded = encoder.decode(code);
                    let residual: Vec<f32> = embedding
                        .iter()
                        .zip(decoded.iter())
                        .map(|(e, d)| e - d)
                        .collect();
                    Some(residual)
                })
                .collect()
        };

        let pq = ProductQuantizer::train(&residuals, 20)
            .map_err(|e| crate::error::FieldError::Manifest(e))?;

        let codebook_bytes = bincode::serialize(&pq)
            .map_err(|e| crate::error::FieldError::Serialization(e.to_string()))?;

        let op = Op::TrainPQ(TrainPQOp { codebook_bytes });
        self.log.write().append(&op)?;

        self.cortical_idx.write().set_pq(pq);

        Ok(())
    }

    /// Encode PQ residual for a single memory. The PQ must already be trained.
    pub fn encode_pq_memory(&self, memory_id: MemoryId) -> Result<()> {
        let embedding = {
            let payloads = self.payloads.read();
            payloads.get(&memory_id).map(|p| p.embedding.clone())
        };
        let Some(embedding) = embedding else {
            return Ok(());
        };
        if embedding.len() != crate::ops::EMBED_DIM {
            return Ok(());
        }

        let decoded = {
            let cortical = self.cortical_idx.read();
            let code = match cortical.mem_codes.get(&memory_id) {
                Some(c) => c.clone(),
                None => return Ok(()),
            };
            self.sparse_encoder.read().decode(&code)
        };

        let residual: Vec<f32> = embedding
            .iter()
            .zip(decoded.iter())
            .map(|(e, d)| e - d)
            .collect();

        let codes = {
            let cortical = self.cortical_idx.read();
            let pq = match &cortical.pq {
                Some(pq) => pq,
                None => return Ok(()),
            };
            pq.quantize(&residual)
        };

        let pq_bytes: Vec<u8> = codes.to_vec();
        let op = Op::UpdateResidualPQ(UpdateResidualPQOp {
            memory_id,
            pq_bytes,
        });
        self.log.write().append(&op)?;

        self.cortical_idx.write().index_pq(memory_id, codes);

        Ok(())
    }

    /// Encode PQ residuals for all memories that have sparse codes but no PQ code.
    /// If PQ is not yet trained, trains it first.
    /// Returns the count of memories PQ-encoded.
    pub fn encode_all_pq(&self) -> Result<usize> {
        if !self.cortical_idx.read().is_pq_trained() {
            self.train_pq()?;
        }

        let ids: Vec<MemoryId> = {
            let cortical = self.cortical_idx.read();
            cortical
                .mem_codes
                .keys()
                .filter(|id| !cortical.mem_pq.contains_key(id))
                .copied()
                .collect()
        };

        let count = ids.len();
        for id in ids {
            self.encode_pq_memory(id)?;
        }

        Ok(count)
    }

    /// Return how many memories have PQ residual codes.
    pub fn pq_count(&self) -> usize {
        self.cortical_idx.read().pq_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open_test_field() -> (ChittaField, TempDir) {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        let field = ChittaField::open(data_dir).unwrap();
        (field, tmp)
    }

    #[test]
    fn test_put_get_roundtrip() {
        let (field, _tmp) = open_test_field();
        let embedding = vec![0.1f32; 768];
        let (id, hash) = field
            .put_memory(
                "wisdom",
                "test",
                b"hello world",
                &embedding,
                0.9,
                0.001,
                0,
                vec![],
                None,
                None,
            )
            .unwrap();

        let payload = field.get_memory(id).unwrap();
        assert_eq!(payload.content, b"hello world");
        assert_eq!(payload.kind, "wisdom");
        assert_eq!(payload.chunk_hash, hash);
    }

    #[test]
    fn test_forget() {
        let (field, _tmp) = open_test_field();
        let embedding = vec![0.0f32; 768];
        let (id, _) = field
            .put_memory(
                "wisdom",
                "test",
                b"to forget",
                &embedding,
                1.0,
                0.001,
                0,
                vec![],
                None,
                None,
            )
            .unwrap();
        field.forget(id).unwrap();
        assert!(matches!(
            field.get_memory(id),
            Err(crate::error::FieldError::Deleted(_))
        ));
    }

    #[test]
    fn test_replay_on_reopen() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");

        let id = {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            let embedding = vec![0.5f32; 768];
            let (id, _) = field
                .put_memory(
                    "episode",
                    "test",
                    b"persisted",
                    &embedding,
                    0.8,
                    0.002,
                    0,
                    vec![],
                    None,
                    None,
                )
                .unwrap();
            id
        };

        // Reopen and verify data survived.
        let field2 = ChittaField::open(data_dir).unwrap();
        let payload = field2.get_memory(id).unwrap();
        assert_eq!(payload.content, b"persisted");
    }

    #[test]
    fn test_assoc_edge() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.1f32; 768];
        let (id1, _) = field
            .put_memory(
                "wisdom",
                "test",
                b"a",
                &emb,
                1.0,
                0.001,
                0,
                vec![],
                None,
                None,
            )
            .unwrap();
        let (id2, _) = field
            .put_memory(
                "wisdom",
                "test",
                b"b",
                &emb,
                1.0,
                0.001,
                0,
                vec![],
                None,
                None,
            )
            .unwrap();
        field
            .add_assoc_edge(id1, id2, EdgeType::CoRetrieved, 0.7)
            .unwrap();
        let neighbors = field.list_neighbors(id1).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].dst, id2);
    }

    #[test]
    fn test_integration_add_triplet() {
        let tmp = TempDir::new().unwrap();
        let field = ChittaField::open(tmp.path().join("data")).unwrap();

        let id = field
            .add_triplet(
                "chitta".into(),
                "replaces".into(),
                "duckdb".into(),
                1.0,
                None,
                None,
            )
            .unwrap();
        assert!(id > 0);

        let results = field.query_subject("chitta").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object, "duckdb");
    }

    #[test]
    fn test_replay_triplets() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");

        {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            field
                .add_triplet("a".into(), "b".into(), "c".into(), 1.0, None, None)
                .unwrap();
        }

        let field2 = ChittaField::open(data_dir).unwrap();
        let results = field2.query_subject("a").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_integration_invalidate_triplet() {
        let tmp = TempDir::new().unwrap();
        let field = ChittaField::open(tmp.path().join("data")).unwrap();

        let id = field
            .add_triplet(
                "chitta".into(),
                "uses".into(),
                "duckdb".into(),
                1.0,
                None,
                None,
            )
            .unwrap();

        let before = field.query_subject("chitta").unwrap();
        assert_eq!(before.len(), 1);

        field.invalidate_triplet(id).unwrap();

        let after = field.query_subject("chitta").unwrap();
        assert_eq!(after.len(), 0);
    }

    #[test]
    fn test_integration_query_entity() {
        let tmp = TempDir::new().unwrap();
        let field = ChittaField::open(tmp.path().join("data")).unwrap();

        field
            .add_triplet(
                "alice".into(),
                "knows".into(),
                "bob".into(),
                1.0,
                None,
                None,
            )
            .unwrap();
        field
            .add_triplet(
                "charlie".into(),
                "knows".into(),
                "alice".into(),
                1.0,
                None,
                None,
            )
            .unwrap();
        field
            .add_triplet(
                "alice".into(),
                "works_at".into(),
                "anthropic".into(),
                1.0,
                None,
                None,
            )
            .unwrap();

        let results = field.query_entity("alice").unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_integration_recall_keyword() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.1f32; 768];

        field
            .put_memory(
                "wisdom",
                "test",
                b"rust ownership model prevents memory leaks automatically",
                &emb,
                1.0,
                0.001,
                0,
                vec![],
                None,
                None,
            )
            .unwrap();
        field
            .put_memory(
                "wisdom",
                "test",
                b"python garbage collector handles memory management",
                &emb,
                1.0,
                0.001,
                0,
                vec![],
                None,
                None,
            )
            .unwrap();

        let hits = field.recall_keyword("rust ownership", 5).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].kind, "wisdom");
        // "rust" and "ownership" only in doc 1
        assert!(hits[0].content.contains("rust"));
    }

    #[test]
    fn test_recall_effects_are_deferred_until_flush() {
        let (field, _tmp) = open_test_field();

        let mut emb1 = vec![0.0f32; 768];
        emb1[0] = 1.0;
        let mut emb2 = vec![0.0f32; 768];
        emb2[1] = 1.0;

        field
            .put_memory(
                "wisdom",
                "test",
                b"alpha memory",
                &emb1,
                1.0,
                0.001,
                0,
                vec![],
                None,
                None,
            )
            .unwrap();
        field
            .put_memory(
                "wisdom",
                "test",
                b"beta memory",
                &emb2,
                1.0,
                0.001,
                0,
                vec![],
                None,
                None,
            )
            .unwrap();

        let seqno_before = field.log.read().last_seqno();
        let hits = field.recall_semantic(&emb1, 2, Some("test")).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(field.log.read().last_seqno(), seqno_before);
        assert!(!field.pending_recall.lock().strengthen.is_empty());

        field.flush().unwrap();
        assert!(field.log.read().last_seqno() > seqno_before);
        assert!(field.pending_recall.lock().strengthen.is_empty());
    }

    // ── Status-aware recall tests ─────────────────────────────────────────────

    /// Superseded/Contradicted/Archived memories must be excluded from semantic recall.
    #[test]
    fn test_recall_excludes_invalidated_statuses() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.5f32; 768];

        let (id_active, _)     = field.put_memory("wisdom", "test", b"active memory",     &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id_superseded, _) = field.put_memory("wisdom", "test", b"superseded memory", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id_contradicted,_)= field.put_memory("wisdom", "test", b"contradicted memory",&emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id_archived, _)   = field.put_memory("wisdom", "test", b"archived memory",   &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

        field.set_memory_status(id_superseded,    crate::state::MemoryStatus::Superseded).unwrap();
        field.set_memory_status(id_contradicted,  crate::state::MemoryStatus::Contradicted).unwrap();
        field.set_memory_status(id_archived,      crate::state::MemoryStatus::Archived).unwrap();

        let hits = field.recall_semantic(&emb, 20, None).unwrap();
        let ids: Vec<_> = hits.iter().map(|h| h.memory_id).collect();

        assert!(ids.contains(&id_active),        "active memory must be recalled");
        assert!(!ids.contains(&id_superseded),   "superseded must be excluded");
        assert!(!ids.contains(&id_contradicted), "contradicted must be excluded");
        assert!(!ids.contains(&id_archived),     "archived must be excluded");
    }

    /// Verified memories score higher than Active; Proposed score lower.
    #[test]
    fn test_recall_status_score_ordering() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.5f32; 768];

        let (id_active,   _) = field.put_memory("wisdom", "test", b"active",   &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id_verified, _) = field.put_memory("wisdom", "test", b"verified", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id_proposed, _) = field.put_memory("wisdom", "test", b"proposed", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

        field.set_memory_status(id_verified, crate::state::MemoryStatus::Verified).unwrap();
        field.set_memory_status(id_proposed, crate::state::MemoryStatus::Proposed).unwrap();

        let hits = field.recall_semantic(&emb, 20, None).unwrap();
        let score = |id: MemoryId| hits.iter().find(|h| h.memory_id == id).map(|h| h.score).unwrap_or(0.0);

        assert!(score(id_verified) > score(id_active),  "verified must outscore active");
        assert!(score(id_active)   > score(id_proposed), "active must outscore proposed");
    }

    // ── Recall explainability tests ─────────────────────────────────────────

    #[test]
    fn test_recall_explain_fields_populated() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.5f32; 768];

        let (id, _) = field.put_memory("wisdom", "test", b"tool derived memory", &emb, 0.9, 0.001, 0, vec![], None, None).unwrap();
        field.set_epistemic_status(id, crate::state::EpistemicStatus::ToolDerived).unwrap();

        let hits = field.recall_semantic(&emb, 5, Some("test")).unwrap();
        let hit = hits.iter().find(|h| h.memory_id == id).expect("memory must be recalled");

        assert!(hit.semantic_weight > 0.0, "semantic_weight must be > 0");
        assert!((hit.status_mul - 1.0).abs() < f32::EPSILON, "Active status_mul must be 1.0");
        assert!((hit.epistemic_mul - 0.95).abs() < f32::EPSILON, "ToolDerived epistemic_mul must be 0.95");
        assert!(hit.strength_factor >= 0.5 && hit.strength_factor <= 1.0, "strength_factor must be in [0.5, 1.0]");
    }

    #[test]
    fn test_recall_explain_score_decomposition() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.5f32; 768];

        let (id, _) = field.put_memory("wisdom", "test", b"decomposition test", &emb, 0.8, 0.001, 0, vec![], None, None).unwrap();

        let hits = field.recall_semantic(&emb, 5, Some("test")).unwrap();
        let hit = hits.iter().find(|h| h.memory_id == id).expect("memory must be recalled");

        // Score includes PoE domain reliability and kind-based multipliers.
        // "wisdom" kind → kind_mul = 1.1, "test" realm with no corrections → default reliability.
        let poe_mul = field.learners.read().domain_reliability.reliability("test");
        let kind_mul = 1.1f32; // realm_kind_multiplier("wisdom")
        let expected = hit.semantic_weight * hit.strength_factor * hit.confidence * hit.status_mul * hit.epistemic_mul * poe_mul * kind_mul;
        assert!(
            (hit.score - expected).abs() < 1e-4,
            "score ({}) must equal semantic_weight * strength_factor * confidence * status_mul * epistemic_mul * poe_mul * kind_mul ({})",
            hit.score, expected
        );
    }

    #[test]
    fn test_recall_keyword_explain_fields() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.1f32; 768];

        field.put_memory("wisdom", "test", b"rust ownership borrow checker lifetime", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

        let hits = field.recall_keyword("rust ownership", 5).unwrap();
        assert!(!hits.is_empty(), "keyword recall must return results");
        let hit = &hits[0];

        assert!(hit.semantic_weight > 0.0, "semantic_weight must be bm25_score > 0");
        assert!(hit.status_mul > 0.0, "status_mul must be populated");
        assert!(hit.epistemic_mul > 0.0, "epistemic_mul must be populated");
        assert!(hit.strength_factor >= 0.5, "strength_factor must be >= 0.5");
    }

    // ── Contradiction engine tests ──────────────────────────────────────────

    #[test]
    fn test_get_conflicts_bidirectional() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.1f32; 768];
        let (id_a, _) = field.put_memory("wisdom", "test", b"memory A", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id_b, _) = field.put_memory("wisdom", "test", b"memory B", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

        field.add_triplet(id_a.to_string(), "contradicts".to_string(), id_b.to_string(), 1.0, None, None).unwrap();

        let conflicts_a = field.get_conflicts(id_a).unwrap();
        let conflicts_b = field.get_conflicts(id_b).unwrap();
        assert!(conflicts_a.contains(&id_b), "A must see B as conflict");
        assert!(conflicts_b.contains(&id_a), "B must see A as conflict");
    }

    #[test]
    fn test_get_supersession_chain_follows_edges() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.1f32; 768];
        let (id_a, _) = field.put_memory("wisdom", "test", b"original", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id_b, _) = field.put_memory("wisdom", "test", b"revision 1", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id_c, _) = field.put_memory("wisdom", "test", b"revision 2", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

        // "B supersedes A" means subject=B, predicate="supersedes", object=A
        field.add_triplet(id_b.to_string(), "supersedes".to_string(), id_a.to_string(), 1.0, None, None).unwrap();
        field.add_triplet(id_c.to_string(), "supersedes".to_string(), id_b.to_string(), 1.0, None, None).unwrap();

        let chain = field.get_supersession_chain(id_a).unwrap();
        assert_eq!(chain, vec![id_a, id_b, id_c], "chain must follow A -> B -> C");
    }

    #[test]
    fn test_get_supersession_chain_cycle_safe() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.1f32; 768];
        let (id_a, _) = field.put_memory("wisdom", "test", b"cycle A", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id_b, _) = field.put_memory("wisdom", "test", b"cycle B", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

        // Create a cycle: B supersedes A, A supersedes B
        field.add_triplet(id_b.to_string(), "supersedes".to_string(), id_a.to_string(), 1.0, None, None).unwrap();
        field.add_triplet(id_a.to_string(), "supersedes".to_string(), id_b.to_string(), 1.0, None, None).unwrap();

        let chain = field.get_supersession_chain(id_a).unwrap();
        assert!(chain.len() <= 21, "cycle must terminate within max depth");
        assert_eq!(chain[0], id_a, "chain must start with self");
    }

    #[test]
    fn test_get_confirmations() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.1f32; 768];
        let (id_x, _) = field.put_memory("wisdom", "test", b"confirmer", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id_y, _) = field.put_memory("wisdom", "test", b"confirmed", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

        // "X confirms Y" means subject=X, predicate="confirms", object=Y
        field.add_triplet(id_x.to_string(), "confirms".to_string(), id_y.to_string(), 1.0, None, None).unwrap();

        let confs = field.get_confirmations(id_y).unwrap();
        assert_eq!(confs, vec![id_x], "Y must show X as confirmer");
    }

    #[test]
    fn test_get_conflicts_empty() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.1f32; 768];
        let (id, _) = field.put_memory("wisdom", "test", b"lonely memory", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

        let conflicts = field.get_conflicts(id).unwrap();
        assert!(conflicts.is_empty(), "no contradictions should return empty vec");
    }

    // ── Regression tests for replay/contract correctness ─────────────────────

    fn put_test_memory(field: &ChittaField, content: &[u8]) -> MemoryId {
        let emb = vec![0.1f32; 768];
        field.put_memory("wisdom", "test", content, &emb, 1.0, 0.001, 0, vec![], None, None)
            .unwrap().0
    }

    /// Bug fix: UpdateState replay used now_ms=0, corrupting last_accessed_ms and
    /// last_strengthened_ms. After reopen the timestamps must reflect op_ts_ms, not epoch 0.
    #[test]
    fn test_replay_update_state_timestamps_nonzero() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");

        let id = {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            let id = put_test_memory(&field, b"state-replay");
            // touch=true writes an UpdateState op with real op_ts_ms
            field.update_state(id, None, None, None, true, None).unwrap();
            field.flush().unwrap();
            id
        };

        let field2 = ChittaField::open(data_dir).unwrap();
        let state = field2.get_state(id).unwrap();
        assert!(
            state.last_accessed_ms > 0,
            "last_accessed_ms must not be 0 after replay, got {}",
            state.last_accessed_ms
        );
    }

    /// Bug fix: UpdateMemoryContent replay did not clear embed_pending, so backfilled
    /// memories were re-queued as pending after every restart.
    #[test]
    fn test_replay_backfill_clears_embed_pending() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");

        let id = {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            // Empty embedding slice → embed_pending = true
            let (id, _) = field.put_memory("wisdom", "test", b"needs-embed", &[], 1.0, 0.001, 0, vec![], None, None).unwrap();
            let emb = vec![0.2f32; 768];
            field.backfill_embedding(id, &emb).unwrap();
            field.flush().unwrap();
            id
        };

        let field2 = ChittaField::open(data_dir).unwrap();
        assert!(
            !field2.pending_embeddings(100).contains(&id),
            "backfilled memory must not appear in pending_embeddings after replay"
        );
    }

    /// Bug fix: backfill_embedding() previously returned Ok(()) for nonexistent IDs.
    #[test]
    fn test_backfill_nonexistent_returns_not_found() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.0f32; 768];
        let fake_id: MemoryId = 0xdeadbeef_cafebabe;
        let result = field.backfill_embedding(fake_id, &emb);
        assert!(
            matches!(result, Err(crate::error::FieldError::NotFound(_))),
            "expected NotFound, got {:?}", result
        );
    }

    /// Bug fix: set_memory_status() and set_epistemic_status() wrote WAL before
    /// confirming the memory exists, leaving orphaned WAL entries on invalid IDs.
    #[test]
    fn test_set_status_invalid_id_no_wal_mutation() {
        let (field, _tmp) = open_test_field();
        let fake_id: MemoryId = 0xdeadbeef_00000001;
        let seqno_before = field.log.read().last_seqno();

        let r1 = field.set_memory_status(fake_id, crate::state::MemoryStatus::Archived);
        let r2 = field.set_epistemic_status(fake_id, crate::state::EpistemicStatus::ModelInferred);

        assert!(matches!(r1, Err(crate::error::FieldError::NotFound(_))));
        assert!(matches!(r2, Err(crate::error::FieldError::NotFound(_))));
        assert_eq!(
            field.log.read().last_seqno(), seqno_before,
            "WAL must not grow when ID is invalid"
        );
    }

    #[test]
    fn test_compact_wal_guard_rejects_small_store() {
        let (field, _tmp) = open_test_field();
        let embedding = vec![0.1f32; 768];
        for i in 0..50 {
            field
                .put_memory(
                    "wisdom",
                    "test",
                    format!("memory {}", i).as_bytes(),
                    &embedding,
                    0.9,
                    0.001,
                    0,
                    vec![],
                    None,
                    None,
                )
                .unwrap();
        }
        let result = field.compact_wal();
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("refusing compact_wal"),
            "expected guard error, got: {}", err_msg
        );
    }

    #[test]
    fn test_compact_wal_guard_allows_large_store() {
        let (field, _tmp) = open_test_field();
        let embedding = vec![0.1f32; 768];
        for i in 0..100 {
            field
                .put_memory(
                    "wisdom",
                    "test",
                    format!("memory {}", i).as_bytes(),
                    &embedding,
                    0.9,
                    0.001,
                    0,
                    vec![],
                    None,
                    None,
                )
                .unwrap();
        }
        let result = field.compact_wal();
        assert!(result.is_ok(), "compact_wal should succeed with 100+ memories, got: {:?}", result);
    }

    #[test]
    fn test_filter_level_signatures_reduces_terms() {
        let (field, _tmp) = open_test_field();
        field.set_filter_level(FilterLevel::Signatures);
        let code = b"fn foo(x: i32) -> i32 {\n    let y = x + 1;\n    y\n}";
        let (id, _) = field
            .put_memory("code", "test", code, &[], 0.8, 0.001, 0, vec![], None, None)
            .unwrap();
        let hits = field.recall_keyword("fn foo", 5).unwrap();
        assert!(hits.iter().any(|h| h.memory_id == id));
        let body_hits = field.recall_keyword("let y", 5).unwrap();
        assert!(!body_hits.iter().any(|h| h.memory_id == id));
    }

    #[test]
    fn test_recall_fallback_to_bm25() {
        let (field, _tmp) = open_test_field();
        for i in 0..15 {
            field
                .put_memory(
                    "wisdom",
                    "test",
                    format!("unique_term_{i} content here").as_bytes(),
                    &[],
                    0.8,
                    0.001,
                    0,
                    vec![],
                    None,
                    None,
                )
                .unwrap();
        }
        let hits = field
            .recall_with_fallback(&vec![0.0f32; 768], "unique_term_0", 5, None)
            .unwrap();
        assert!(!hits.is_empty(), "fallback should return results");
    }
}

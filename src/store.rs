use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};
use crate::error::{FieldError, Result};
use crate::field::{AssocEdge, ChittaField};
use crate::ids::{ArtifactId, ChunkHash, MemoryId, compute_chunk_hash};
use crate::ops::{
    AddAssocEdgeOp, DeleteMemoryOp, EdgeType, Op, PutPayloadOp, StateDeltaOp,
    UpsertArtifactOp, ArtifactRef, AddTripletOp, InvalidateTripletOp,
    UpsertSymbolOp, RemoveSymbolOp, AddSymCallEdgeOp, UpsertCodeFileOp, UpdateSparseCodeOp,
};
use crate::organ::triplet::TripletEntry;
use crate::organ::symbol::SymbolEntry;
use crate::payload::MemoryPayload;
use crate::state::MemoryState;
use crate::ops::EMBED_DIM;
use crate::recall::RecallHit;
use crate::learner::route::{Route, RouteLearner};

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
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
        if embedding.len() != EMBED_DIM {
            return Err(FieldError::InvalidEmbedDim {
                expected: EMBED_DIM,
                actual: embedding.len(),
            });
        }

        let chunk_hash = compute_chunk_hash(kind, realm, content, embedding);
        let memory_id = self.id_alloc.next_id();
        let ts = now_ms();

        let op = PutPayloadOp {
            memory_id,
            version: 0,
            chunk_hash,
            created_at_ms: ts,
            authored_at_ms,
            kind: kind.to_string(),
            realm: realm.to_string(),
            content: content.to_vec(),
            embedding_model: "bge-base-en-v1.5".to_string(),
            embedding: embedding.to_vec(),
            artifact_refs: artifact_refs.clone(),
            source_session,
            source_tool,
        };

        let op_enum = Op::PutPayload(op.clone());
        let seqno = self.log.write().append(&op_enum)?;

        let payload = MemoryPayload::from(op);
        let mut state = MemoryState::new(memory_id, chunk_hash, ts);
        state.confidence = confidence;
        state.decay_rate = decay_rate;

        self.payloads.write().insert(memory_id, payload);
        self.states.write().insert(memory_id, state);
        self.semantic_idx.write().upsert(memory_id, embedding.to_vec());
        let content_str = std::str::from_utf8(content).unwrap_or("").to_string();
        self.keyword_idx.write().index(memory_id, &content_str);

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

        Ok((memory_id, chunk_hash))
    }

    /// Retrieve the payload for a memory. Also records a touch access.
    pub fn get_memory(&self, memory_id: MemoryId) -> Result<MemoryPayload> {
        {
            let states = self.states.read();
            let state = states.get(&memory_id).ok_or(FieldError::NotFound(memory_id))?;
            if state.deleted {
                return Err(FieldError::Deleted(memory_id));
            }
        }

        let ts = now_ms();

        // Record access in plasticity learner and get recommended decay rate.
        let recommended_decay = self.learners.write().plasticity.record_access(memory_id, ts);

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
            let state = states.get(&memory_id).ok_or(FieldError::NotFound(memory_id))?;
            if state.deleted {
                return Err(FieldError::Deleted(memory_id));
            }
        }

        let delta = StateDeltaOp {
            memory_id,
            strength_delta,
            confidence_delta,
            decay_rate,
            touch,
            pin,
        };
        let _seqno = self.log.write().append(&Op::UpdateState(delta.clone()))?;

        let ts = now_ms();
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
            states.get(&memory_id).ok_or(FieldError::NotFound(memory_id))?;
        }

        let ts = now_ms();
        let op = Op::DeleteMemory(DeleteMemoryOp {
            memory_id,
            deleted_at_ms: ts,
        });
        let _seqno = self.log.write().append(&op)?;

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
                self.time_idx.write().remove(memory_id, payload.authored_at_ms);
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
            .push(AssocEdge { dst, edge_type, weight });

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

    /// Semantic recall: find k most similar memories to a query embedding.
    ///
    /// Applies realm filter and strength-weighted final scoring:
    ///   `score = semantic_score × (0.5 + 0.5 × effective_strength) × confidence`
    ///
    /// Queries `k * 3` candidates from the index to give the score re-ranking
    /// room to promote high-confidence results.
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

        // Build realm-filtered allowed set from payloads (realm lives in payload, not state).
        let allowed: HashSet<MemoryId> = {
            let payloads = self.payloads.read();
            let states = self.states.read();
            payloads
                .iter()
                .filter(|(id, p)| {
                    let not_deleted = states
                        .get(id)
                        .map(|s| !s.deleted)
                        .unwrap_or(false);
                    let realm_match = realm.map(|r| p.realm == r).unwrap_or(true);
                    not_deleted && realm_match
                })
                .map(|(id, _)| *id)
                .collect()
        };

        // Encode query into sparse code
        let query_code = self.sparse_encoder.read().encode(query_embedding);

        let now = now_ms();

        // Merge cortical and semantic results
        let mut seen: std::collections::HashMap<MemoryId, f32> = std::collections::HashMap::new();

        let cortical_empty = self.cortical_idx.read().is_empty();

        if cortical_empty {
            // Pure brute-force fallback: no cortical index yet
            let candidates = self.semantic_idx.read().search(
                query_embedding,
                k * 3,
                Some(&allowed),
            );
            let states_r = self.states.read();
            for hit in candidates {
                let eff = states_r.get(&hit.memory_id).map(|s| 0.5 + 0.5 * s.effective_strength(now)).unwrap_or(0.5);
                let conf = states_r.get(&hit.memory_id).map(|s| s.confidence).unwrap_or(1.0);
                seen.insert(hit.memory_id, hit.cosine_similarity * eff * conf);
            }
        } else {
            // Cortical index hits (fast path for indexed memories)
            {
                let cortical = self.cortical_idx.read();
                for (mem_id, score) in cortical.search(&query_code, k * 3, Some(&allowed)) {
                    seen.insert(mem_id, score);
                }
            }

            // Semantic (HNSW) fallback for memories not yet in cortical index
            let cortical_mem_ids: std::collections::HashSet<MemoryId> = {
                self.cortical_idx.read().mem_codes.keys().copied().collect()
            };
            let semantic_allowed: std::collections::HashSet<MemoryId> = allowed
                .iter()
                .filter(|id| !cortical_mem_ids.contains(id))
                .copied()
                .collect();

            if !semantic_allowed.is_empty() {
                let candidates = self.semantic_idx.read().search(
                    query_embedding,
                    k * 3,
                    Some(&semantic_allowed),
                );
                let states_r = self.states.read();
                for hit in candidates {
                    let eff = states_r.get(&hit.memory_id).map(|s| 0.5 + 0.5 * s.effective_strength(now)).unwrap_or(0.5);
                    let conf = states_r.get(&hit.memory_id).map(|s| s.confidence).unwrap_or(1.0);
                    seen.entry(hit.memory_id).or_insert(hit.cosine_similarity * eff * conf);
                }
            }
        }

        let states = self.states.read();
        let payloads = self.payloads.read();

        let mut hits: Vec<RecallHit> = seen
            .into_iter()
            .filter_map(|(memory_id, score)| {
                let state = states.get(&memory_id)?;
                let payload = payloads.get(&memory_id)?;
                let eff_strength = state.effective_strength(now);
                Some(RecallHit {
                    memory_id,
                    score,
                    semantic_score: score,
                    ts_ms: payload.created_at_ms,
                    kind: payload.kind.clone(),
                    realm: payload.realm.clone(),
                    strength: eff_strength,
                    confidence: state.confidence,
                    content: String::from_utf8(payload.content.clone()).unwrap_or_default(),
                })
            })
            .collect();

        hits.sort_unstable_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);

        // Reconsolidation: mini-strengthen each returned memory
        let hit_ids: Vec<MemoryId> = hits.iter().map(|h| h.memory_id).collect();
        drop(states);
        drop(payloads);
        for &id in &hit_ids {
            let _ = self.update_state(id, Some(0.01), None, None, true, None);
            let new_strength = self.states.read().get(&id).map(|s| s.strength);
            if let Some(s) = new_strength {
                self.cortical_idx.write().update_strength(id, s);
            }
        }
        // Add co-retrieval edges for top results
        for i in 0..hit_ids.len().min(5) {
            for j in (i + 1)..hit_ids.len().min(5) {
                let _ = self.add_assoc_edge(hit_ids[i], hit_ids[j], EdgeType::CoRetrieved, 0.05);
            }
        }

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
        let mut activation: std::collections::HashMap<MemoryId, f32> = std::collections::HashMap::new();

        // frontier: (memory_id, activation_score, hops_remaining)
        let mut frontier: Vec<(MemoryId, f32, usize)> = seed_ids
            .iter()
            .map(|&id| (id, 1.0, max_hops))
            .collect();

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
                    content: String::from_utf8(payload.content.clone()).unwrap_or_default(),
                })
            })
            .collect();

        hits.sort_unstable_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
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
        let entries = self.time_idx.read().range_query(start_ms, end_ms, realm, limit);
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
                    content: String::from_utf8(payload.content.clone()).unwrap_or_default(),
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
                let payload = payloads.get(&hit.memory_id)?;
                let eff_strength = state.effective_strength(now);
                let score = hit.bm25_score * (0.5 + 0.5 * eff_strength) * state.confidence;
                Some(RecallHit {
                    memory_id: hit.memory_id,
                    score,
                    semantic_score: 0.0,
                    ts_ms: payload.authored_at_ms,
                    kind: payload.kind.clone(),
                    realm: payload.realm.clone(),
                    strength: eff_strength,
                    confidence: state.confidence,
                    content: String::from_utf8(payload.content.clone()).unwrap_or_default(),
                })
            })
            .collect();

        hits.sort_unstable_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(k);

        // Reconsolidation: mini-strengthen each returned memory
        let hit_ids: Vec<MemoryId> = hits.iter().map(|h| h.memory_id).collect();
        drop(states);
        drop(payloads);
        for &id in &hit_ids {
            let _ = self.update_state(id, Some(0.01), None, None, true, None);
            let new_strength = self.states.read().get(&id).map(|s| s.strength);
            if let Some(s) = new_strength {
                self.cortical_idx.write().update_strength(id, s);
            }
        }
        // Add co-retrieval edges for top results
        for i in 0..hit_ids.len().min(5) {
            for j in (i + 1)..hit_ids.len().min(5) {
                let _ = self.add_assoc_edge(hit_ids[i], hit_ids[j], EdgeType::CoRetrieved, 0.05);
            }
        }

        Ok(hits)
    }

    /// Recall memories associated with a file path (exact match).
    pub fn recall_artifact(
        &self,
        path: &str,
        limit: usize,
    ) -> Result<Vec<RecallHit>> {
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
                    content: String::from_utf8(payload.content.clone()).unwrap_or_default(),
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

    /// Invalidate a triplet (marks it as expired at the current time).
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
        Ok(store.query_subject(subject, at_ms).into_iter().cloned().collect())
    }

    /// Query all currently-valid triplets with the given object.
    pub fn query_object(&self, object: &str) -> Result<Vec<TripletEntry>> {
        let at_ms = now_ms();
        let store = self.triplet_store.read();
        Ok(store.query_object(object, at_ms).into_iter().cloned().collect())
    }

    /// Query all currently-valid triplets where subject OR object matches.
    pub fn query_entity(&self, entity: &str) -> Result<Vec<TripletEntry>> {
        let at_ms = now_ms();
        let store = self.triplet_store.read();
        Ok(store.query_entity(entity, at_ms).into_iter().cloned().collect())
    }

    /// Apply feedback for a recall episode (route learning).
    pub fn feedback(&self, episode_id: u64, reward: f32) -> Result<()> {
        self.learners.write().route.feedback(episode_id, reward);
        Ok(())
    }

    /// Get recommended window size for a session type.
    pub fn recommended_window(&self, session_type: &str) -> usize {
        self.learners.read().context.recommended_window(session_type)
    }

    /// Record context outcome for a session type and window size.
    pub fn record_context_outcome(&self, session_type: &str, size: usize, outcome: f32) {
        self.learners.write().context.record_outcome(session_type, size, outcome);
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
        let op = Op::AddSymCallEdge(AddSymCallEdgeOp { caller_id, callee_id });
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
        let (file_id, _was_updated) = self.code_files.write().upsert(path, project, mtime, next_id_fn);
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

    /// Encode a memory's embedding into sparse codes and index into the cortical index.
    /// Persists via UpdateSparseCode op.
    pub fn encode_memory(&self, memory_id: MemoryId) -> Result<()> {
        let embedding = {
            let payloads = self.payloads.read();
            payloads.get(&memory_id).map(|p| p.embedding.clone())
        };
        let Some(embedding) = embedding else { return Ok(()); };
        if embedding.len() != EMBED_DIM { return Ok(()); }

        let code = self.sparse_encoder.read().encode(&embedding);
        if code.is_empty() { return Ok(()); }

        // Hebbian update
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
        let strength = self.states.read().get(&memory_id).map(|s| s.strength).unwrap_or(0.5);
        let kind = self.payloads.read().get(&memory_id).map(|p| p.kind.clone()).unwrap_or_default();
        let authored_at = self.payloads.read().get(&memory_id).map(|p| p.authored_at_ms).unwrap_or(ts_ms);
        self.cortical_idx.write().index(memory_id, &code, strength, authored_at, &kind);

        Ok(())
    }

    /// Encode all memories that don't yet have a sparse code.
    pub fn encode_all_unindexed(&self) -> Result<usize> {
        let ids: Vec<MemoryId> = {
            let payloads = self.payloads.read();
            let cortical = self.cortical_idx.read();
            payloads.keys()
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
        let (id, hash) = field.put_memory(
            "wisdom", "test", b"hello world", &embedding,
            0.9, 0.001, 0, vec![], None, None,
        ).unwrap();

        let payload = field.get_memory(id).unwrap();
        assert_eq!(payload.content, b"hello world");
        assert_eq!(payload.kind, "wisdom");
        assert_eq!(payload.chunk_hash, hash);
    }

    #[test]
    fn test_forget() {
        let (field, _tmp) = open_test_field();
        let embedding = vec![0.0f32; 768];
        let (id, _) = field.put_memory(
            "wisdom", "test", b"to forget", &embedding,
            1.0, 0.001, 0, vec![], None, None,
        ).unwrap();
        field.forget(id).unwrap();
        assert!(matches!(field.get_memory(id), Err(crate::error::FieldError::Deleted(_))));
    }

    #[test]
    fn test_replay_on_reopen() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");

        let id = {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            let embedding = vec![0.5f32; 768];
            let (id, _) = field.put_memory(
                "episode", "test", b"persisted", &embedding,
                0.8, 0.002, 0, vec![], None, None,
            ).unwrap();
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
        let (id1, _) = field.put_memory("wisdom", "test", b"a", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id2, _) = field.put_memory("wisdom", "test", b"b", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        field.add_assoc_edge(id1, id2, EdgeType::CoRetrieved, 0.7).unwrap();
        let neighbors = field.list_neighbors(id1).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].dst, id2);
    }

    #[test]
    fn test_integration_add_triplet() {
        let tmp = TempDir::new().unwrap();
        let field = ChittaField::open(tmp.path().join("data")).unwrap();

        let id = field.add_triplet(
            "chitta".into(), "replaces".into(), "duckdb".into(),
            1.0, None, None,
        ).unwrap();
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
            field.add_triplet("a".into(), "b".into(), "c".into(), 1.0, None, None).unwrap();
        }

        let field2 = ChittaField::open(data_dir).unwrap();
        let results = field2.query_subject("a").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_integration_invalidate_triplet() {
        let tmp = TempDir::new().unwrap();
        let field = ChittaField::open(tmp.path().join("data")).unwrap();

        let id = field.add_triplet(
            "chitta".into(), "uses".into(), "duckdb".into(),
            1.0, None, None,
        ).unwrap();

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

        field.add_triplet("alice".into(), "knows".into(), "bob".into(), 1.0, None, None).unwrap();
        field.add_triplet("charlie".into(), "knows".into(), "alice".into(), 1.0, None, None).unwrap();
        field.add_triplet("alice".into(), "works_at".into(), "anthropic".into(), 1.0, None, None).unwrap();

        let results = field.query_entity("alice").unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_integration_recall_keyword() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.1f32; 768];

        field.put_memory("wisdom", "test",
            b"rust ownership model prevents memory leaks automatically",
            &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        field.put_memory("wisdom", "test",
            b"python garbage collector handles memory management",
            &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

        let hits = field.recall_keyword("rust ownership", 5).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].kind, "wisdom");
        // "rust" and "ownership" only in doc 1
        assert!(hits[0].content.contains("rust"));
    }
}

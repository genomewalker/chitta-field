use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct ThemeRecord {
    pub theme_id: u64,
    pub name: String,
    pub centroid: Vec<f32>,
    pub realm: String,
    pub coherence: f32,
    pub member_ids: HashSet<u64>,
    pub created_at: i64,
}

#[derive(Debug, Default)]
pub struct ThemeStats {
    pub total_themes: usize,
    pub total_memberships: usize,
    pub orphan_count: usize,
    pub avg_size: f32,
    pub avg_coherence: f32,
}

#[derive(Debug, Default)]
pub struct ThemeMaintenanceResult {
    pub themes_split: usize,
    pub themes_merged: usize,
    pub memories_reassigned: usize,
}

#[derive(Debug, Default)]
pub struct ThemeOrgan {
    themes: HashMap<u64, ThemeRecord>,
    next_id: u64,
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

impl ThemeOrgan {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create(&mut self, theme_id: u64, name: String) {
        if theme_id >= self.next_id {
            self.next_id = theme_id + 1;
        }
        self.themes.insert(
            theme_id,
            ThemeRecord {
                theme_id,
                name,
                centroid: Vec::new(),
                realm: String::new(),
                coherence: 1.0,
                member_ids: HashSet::new(),
                created_at: 0,
            },
        );
    }

    /// Update centroid from a JSON string (used during log replay).
    /// Parses a JSON array of floats. Empty string clears the centroid.
    pub fn update_centroid(&mut self, theme_id: u64, centroid_json: String) {
        if let Some(t) = self.themes.get_mut(&theme_id) {
            if centroid_json.is_empty() {
                t.centroid = Vec::new();
            } else if let Ok(vals) = serde_json::from_str::<Vec<f32>>(&centroid_json) {
                t.centroid = vals;
            }
        }
    }

    /// Recompute centroid as mean of member embeddings.
    pub fn recompute_centroid(&mut self, theme_id: u64, embeddings: &HashMap<u64, Vec<f32>>) {
        let t = match self.themes.get_mut(&theme_id) {
            Some(t) => t,
            None => return,
        };
        let member_embs: Vec<&Vec<f32>> = t
            .member_ids
            .iter()
            .filter_map(|id| embeddings.get(id))
            .collect();
        if member_embs.is_empty() {
            return;
        }
        let dim = member_embs[0].len();
        let mut centroid = vec![0.0f32; dim];
        for emb in &member_embs {
            for (c, v) in centroid.iter_mut().zip(emb.iter()) {
                *c += v;
            }
        }
        let n = member_embs.len() as f32;
        for c in centroid.iter_mut() {
            *c /= n;
        }
        // Compute coherence: 1.0 if <3 members, else avg cosine sim to centroid
        if member_embs.len() < 3 {
            t.coherence = 1.0;
        } else {
            let sum: f32 = member_embs.iter().map(|e| cosine_sim(e, &centroid)).sum();
            t.coherence = sum / n;
        }
        t.centroid = centroid;
    }

    pub fn assign_member(&mut self, theme_id: u64, memory_id: u64) {
        if let Some(t) = self.themes.get_mut(&theme_id) {
            t.member_ids.insert(memory_id);
        }
    }

    pub fn remove_member(&mut self, theme_id: u64, memory_id: u64) {
        if let Some(t) = self.themes.get_mut(&theme_id) {
            t.member_ids.remove(&memory_id);
        }
    }

    pub fn get(&self, theme_id: u64) -> Option<&ThemeRecord> {
        self.themes.get(&theme_id)
    }

    pub fn list_all(&self) -> Vec<&ThemeRecord> {
        self.themes.values().collect()
    }

    pub fn theme_count(&self) -> usize {
        self.themes.len()
    }

    /// Stats for themes in the given realm. Empty realm means all realms.
    pub fn stats(&self, realm: &str, total_memory_count: usize) -> ThemeStats {
        let themes: Vec<&ThemeRecord> = if realm.is_empty() {
            self.themes.values().collect()
        } else {
            self.themes.values().filter(|t| t.realm == realm).collect()
        };

        let total_themes = themes.len();
        let total_memberships: usize = themes.iter().map(|t| t.member_ids.len()).sum();
        let avg_size = if total_themes == 0 {
            0.0
        } else {
            total_memberships as f32 / total_themes as f32
        };
        let avg_coherence = if total_themes == 0 {
            0.0
        } else {
            let sum: f32 = themes.iter().map(|t| t.coherence).sum();
            sum / total_themes as f32
        };

        // Collect all themed memory ids
        let all_themed: HashSet<u64> = if realm.is_empty() {
            self.themes
                .values()
                .flat_map(|t| t.member_ids.iter().copied())
                .collect()
        } else {
            self.themes
                .values()
                .filter(|t| t.realm == realm)
                .flat_map(|t| t.member_ids.iter().copied())
                .collect()
        };
        let orphan_count = total_memory_count.saturating_sub(all_themed.len());

        ThemeStats {
            total_themes,
            total_memberships,
            orphan_count,
            avg_size,
            avg_coherence,
        }
    }

    /// Find top-k themes by cosine similarity between query embedding and theme centroids.
    pub fn recall_by_embedding(&self, embedding: &[f32], k: usize, realm: &str) -> Vec<(u64, f32)> {
        let mut scored: Vec<(u64, f32)> = self
            .themes
            .values()
            .filter(|t| realm.is_empty() || t.realm == realm)
            .filter(|t| !t.centroid.is_empty())
            .map(|t| (t.theme_id, cosine_sim(embedding, &t.centroid)))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Assigns orphan memories (not in any theme) to the best matching theme.
    /// Creates a new theme if best cosine sim is below assignment_threshold.
    /// Returns (assigned, remaining_orphans).
    pub fn assign_orphans(
        &mut self,
        all_memory_ids: &[u64],
        embeddings: &HashMap<u64, Vec<f32>>,
        realm: &str,
        batch_size: usize,
        assignment_threshold: f32,
    ) -> (usize, usize) {
        // Collect all themed memory ids
        let themed: HashSet<u64> = self
            .themes
            .values()
            .flat_map(|t| t.member_ids.iter().copied())
            .collect();

        let orphans: Vec<u64> = all_memory_ids
            .iter()
            .copied()
            .filter(|id| !themed.contains(id))
            .collect();

        let total_orphans = orphans.len();
        let batch: Vec<u64> = orphans.into_iter().take(batch_size).collect();
        let mut assigned = 0usize;

        for mem_id in &batch {
            let emb = match embeddings.get(mem_id) {
                Some(e) if !e.is_empty() => e,
                _ => continue,
            };

            // Find best matching theme
            let best = self
                .themes
                .values()
                .filter(|t| realm.is_empty() || t.realm == realm)
                .filter(|t| !t.centroid.is_empty())
                .map(|t| (t.theme_id, cosine_sim(emb, &t.centroid)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((best_id, best_score)) = best {
                if best_score >= assignment_threshold {
                    self.themes
                        .get_mut(&best_id)
                        .unwrap()
                        .member_ids
                        .insert(*mem_id);
                    assigned += 1;
                    continue;
                }
            }

            // Create a new theme for this orphan
            let new_id = self.next_id;
            self.next_id += 1;
            let name = format!("theme_{}", new_id);
            let mut record = ThemeRecord {
                theme_id: new_id,
                name,
                centroid: emb.clone(),
                realm: realm.to_string(),
                coherence: 1.0,
                member_ids: HashSet::new(),
                created_at: 0,
            };
            record.member_ids.insert(*mem_id);
            self.themes.insert(new_id, record);
            assigned += 1;
        }

        let remaining = total_orphans.saturating_sub(batch.len());
        (assigned, remaining)
    }

    /// Split themes with >100 members (k-means k=2) and merge themes with
    /// centroid cosine_sim > 0.9.
    pub fn maintain(&mut self, embeddings: &HashMap<u64, Vec<f32>>) -> ThemeMaintenanceResult {
        let mut themes_split = 0usize;
        let mut themes_merged = 0usize;
        let mut memories_reassigned = 0usize;

        // --- Split pass ---
        let split_candidates: Vec<u64> = self
            .themes
            .values()
            .filter(|t| t.member_ids.len() > 100)
            .map(|t| t.theme_id)
            .collect();

        for theme_id in split_candidates {
            let (member_ids, realm) = {
                let t = match self.themes.get(&theme_id) {
                    Some(t) => t,
                    None => continue,
                };
                if t.member_ids.len() <= 100 {
                    continue;
                }
                (
                    t.member_ids.iter().copied().collect::<Vec<u64>>(),
                    t.realm.clone(),
                )
            };

            let members_with_emb: Vec<(u64, &Vec<f32>)> = member_ids
                .iter()
                .filter_map(|id| embeddings.get(id).map(|e| (*id, e)))
                .filter(|(_, e)| !e.is_empty())
                .collect();

            if members_with_emb.len() < 2 {
                continue;
            }

            // Pick two seeds: first and last in the list
            let seed_a = members_with_emb[0].1.clone();
            let seed_b = members_with_emb[members_with_emb.len() - 1].1.clone();
            let mut centroid_a = seed_a;
            let mut centroid_b = seed_b;

            // 5 iterations of k-means
            let mut cluster_a: Vec<u64> = Vec::new();
            let mut cluster_b: Vec<u64> = Vec::new();
            for _ in 0..5 {
                cluster_a.clear();
                cluster_b.clear();
                for (id, emb) in &members_with_emb {
                    let sa = cosine_sim(emb, &centroid_a);
                    let sb = cosine_sim(emb, &centroid_b);
                    if sa >= sb {
                        cluster_a.push(*id);
                    } else {
                        cluster_b.push(*id);
                    }
                }
                if !cluster_a.is_empty() {
                    centroid_a = mean_embedding(&cluster_a, embeddings);
                }
                if !cluster_b.is_empty() {
                    centroid_b = mean_embedding(&cluster_b, embeddings);
                }
            }

            if cluster_a.is_empty() || cluster_b.is_empty() {
                continue;
            }

            // Update the original theme with cluster_a
            {
                let t = self.themes.get_mut(&theme_id).unwrap();
                t.member_ids = cluster_a.iter().copied().collect();
                t.centroid = centroid_a;
                if t.member_ids.len() < 3 {
                    t.coherence = 1.0;
                } else {
                    let n = t.member_ids.len() as f32;
                    let sum: f32 = t
                        .member_ids
                        .iter()
                        .filter_map(|id| embeddings.get(id))
                        .map(|e| cosine_sim(e, &t.centroid))
                        .sum();
                    t.coherence = sum / n;
                }
            }

            // Create new theme for cluster_b
            let new_id = self.next_id;
            self.next_id += 1;
            let member_count_b = cluster_b.len();
            let coherence_b = if member_count_b < 3 {
                1.0
            } else {
                let n = member_count_b as f32;
                let sum: f32 = cluster_b
                    .iter()
                    .filter_map(|id| embeddings.get(id))
                    .map(|e| cosine_sim(e, &centroid_b))
                    .sum();
                sum / n
            };
            let record = ThemeRecord {
                theme_id: new_id,
                name: format!("theme_{}", new_id),
                centroid: centroid_b,
                realm,
                coherence: coherence_b,
                member_ids: cluster_b.iter().copied().collect(),
                created_at: 0,
            };
            memories_reassigned += member_count_b;
            self.themes.insert(new_id, record);
            themes_split += 1;
        }

        // --- Merge pass ---
        let theme_ids: Vec<u64> = self.themes.keys().copied().collect();
        let mut merged: HashSet<u64> = HashSet::new();

        'outer: for i in 0..theme_ids.len() {
            let id_a = theme_ids[i];
            if merged.contains(&id_a) {
                continue;
            }
            for j in (i + 1)..theme_ids.len() {
                let id_b = theme_ids[j];
                if merged.contains(&id_b) {
                    continue;
                }

                let sim = {
                    let ta = match self.themes.get(&id_a) {
                        Some(t) if !t.centroid.is_empty() => t,
                        _ => continue,
                    };
                    let tb = match self.themes.get(&id_b) {
                        Some(t) if !t.centroid.is_empty() => t,
                        _ => continue,
                    };
                    cosine_sim(&ta.centroid, &tb.centroid)
                };

                if sim > 0.9 {
                    // Merge b into a
                    let (members_b, _) = {
                        let tb = self.themes.get(&id_b).unwrap();
                        (
                            tb.member_ids.iter().copied().collect::<Vec<u64>>(),
                            tb.realm.clone(),
                        )
                    };
                    let reassigned = members_b.len();
                    {
                        let ta = self.themes.get_mut(&id_a).unwrap();
                        for mid in members_b {
                            ta.member_ids.insert(mid);
                        }
                    }
                    self.recompute_centroid(id_a, embeddings);
                    merged.insert(id_b);
                    themes_merged += 1;
                    memories_reassigned += reassigned;
                    if themes_merged > 0 && merged.len() >= theme_ids.len() / 2 {
                        break 'outer;
                    }
                }
            }
        }
        for id in &merged {
            self.themes.remove(id);
        }

        ThemeMaintenanceResult {
            themes_split,
            themes_merged,
            memories_reassigned,
        }
    }
}

fn mean_embedding(ids: &[u64], embeddings: &HashMap<u64, Vec<f32>>) -> Vec<f32> {
    let vecs: Vec<&Vec<f32>> = ids.iter().filter_map(|id| embeddings.get(id)).collect();
    if vecs.is_empty() {
        return Vec::new();
    }
    let dim = vecs[0].len();
    let mut result = vec![0.0f32; dim];
    for v in &vecs {
        for (r, x) in result.iter_mut().zip(v.iter()) {
            *r += x;
        }
    }
    let n = vecs.len() as f32;
    for r in result.iter_mut() {
        *r /= n;
    }
    result
}

impl crate::organ::OrganApply for ThemeOrgan {
    /// Organ-owned WAL replay (THEORY.md §8 Phase 2).
    fn apply(&mut self, op: crate::ops::Op) -> Option<crate::ops::Op> {
        use crate::ops::Op;
        match op {
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
                        self.create(ev.theme_id, name);
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
                        self.update_centroid(ev.theme_id, centroid);
                    }
                    "assign_member" => {
                        let memory_id = serde_json::from_str::<serde_json::Value>(&payload_str)
                            .ok()
                            .and_then(|v| v.get("memory_id").and_then(|m| m.as_u64()))
                            .unwrap_or(0);
                        if memory_id > 0 {
                            self.assign_member(ev.theme_id, memory_id);
                        }
                    }
                    "remove_member" => {
                        let memory_id = serde_json::from_str::<serde_json::Value>(&payload_str)
                            .ok()
                            .and_then(|v| v.get("memory_id").and_then(|m| m.as_u64()))
                            .unwrap_or(0);
                        if memory_id > 0 {
                            self.remove_member(ev.theme_id, memory_id);
                        }
                    }
                    _ => {}
                }
                    None
                }
            other => Some(other),
        }
    }
}

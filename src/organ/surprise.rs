//! Surprise Memory — prediction error tuples that reveal blind spots.
//!
//! Tracks divergences between expected and actual outcomes. Over time,
//! surprise patterns expose systematic blind spots and help calibrate
//! confidence. Inspired by free-energy principle prediction error signals.
//! Reference: Friston (2010), Clark (2013) predictive processing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurpriseEvent {
    pub id: u64,
    pub context_sketch: String,
    pub action: String,
    pub expected: Option<String>,
    pub actual: String,
    pub surprise_magnitude: f32,
    pub domain: String,
    pub timestamp_ms: i64,
    pub realm: String,
    pub session_id: Option<String>,
    pub source_memory_id: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlindSpot {
    pub domain: String,
    pub action: String,
    pub count: usize,
    pub avg_magnitude: f32,
    pub recent_ids: Vec<u64>,
}

pub struct SurpriseStats {
    pub total_events: usize,
    pub avg_magnitude: f32,
    pub by_domain: Vec<(String, usize)>,
}

pub struct SurpriseStore {
    next_id: u64,
    events: Vec<SurpriseEvent>,
    id_to_index: HashMap<u64, usize>,
    by_domain: HashMap<String, Vec<u64>>,
    by_source_memory: HashMap<u64, Vec<u64>>,
}

impl SurpriseStore {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            events: Vec::new(),
            id_to_index: HashMap::new(),
            by_domain: HashMap::new(),
            by_source_memory: HashMap::new(),
        }
    }

    pub fn record(
        &mut self,
        context_sketch: String,
        action: String,
        expected: Option<String>,
        actual: String,
        surprise_magnitude: f32,
        domain: String,
        realm: String,
        session_id: Option<String>,
        source_memory_id: Option<u64>,
        now_ms: i64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let event = SurpriseEvent {
            id,
            context_sketch,
            action,
            expected,
            actual,
            surprise_magnitude: surprise_magnitude.clamp(0.0, 1.0),
            domain,
            timestamp_ms: now_ms,
            realm,
            session_id,
            source_memory_id,
        };
        self.insert(event);
        id
    }

    pub fn replay_record(&mut self, event: SurpriseEvent) {
        if event.id >= self.next_id {
            self.next_id = event.id + 1;
        }
        self.insert(event);
    }

    fn insert(&mut self, event: SurpriseEvent) {
        let id = event.id;
        let idx = self.events.len();
        self.by_domain
            .entry(event.domain.clone())
            .or_default()
            .push(id);
        if let Some(mem_id) = event.source_memory_id {
            self.by_source_memory.entry(mem_id).or_default().push(id);
        }
        self.events.push(event);
        self.id_to_index.insert(id, idx);
    }

    pub fn get(&self, id: u64) -> Option<&SurpriseEvent> {
        self.id_to_index.get(&id).map(|&idx| &self.events[idx])
    }

    pub fn query(
        &self,
        domain: Option<&str>,
        realm: Option<&str>,
        min_magnitude: Option<f32>,
        since_ms: Option<i64>,
        limit: usize,
    ) -> Vec<&SurpriseEvent> {
        let candidates: Box<dyn Iterator<Item = &SurpriseEvent>> = if let Some(d) = domain {
            if let Some(ids) = self.by_domain.get(d) {
                Box::new(
                    ids.iter()
                        .rev()
                        .filter_map(|id| self.get(*id)),
                )
            } else {
                return Vec::new();
            }
        } else {
            Box::new(self.events.iter().rev())
        };

        candidates
            .filter(|e| realm.map_or(true, |r| e.realm == r))
            .filter(|e| min_magnitude.map_or(true, |m| e.surprise_magnitude >= m))
            .filter(|e| since_ms.map_or(true, |t| e.timestamp_ms >= t))
            .take(limit)
            .collect()
    }

    pub fn get_blind_spots(&self, realm: Option<&str>, limit: usize) -> Vec<BlindSpot> {
        let mut groups: HashMap<(String, String), (usize, f32, Vec<u64>)> = HashMap::new();

        for event in &self.events {
            if realm.map_or(false, |r| event.realm != r) {
                continue;
            }
            let key = (event.domain.clone(), event.action.clone());
            let entry = groups.entry(key).or_insert_with(|| (0, 0.0, Vec::new()));
            entry.0 += 1;
            entry.1 += event.surprise_magnitude;
            if entry.2.len() < 5 {
                entry.2.push(event.id);
            }
        }

        let mut spots: Vec<BlindSpot> = groups
            .into_iter()
            .map(|((domain, action), (count, total_mag, recent_ids))| {
                let avg_magnitude = if count > 0 {
                    total_mag / count as f32
                } else {
                    0.0
                };
                BlindSpot {
                    domain,
                    action,
                    count,
                    avg_magnitude,
                    recent_ids,
                }
            })
            .collect();

        spots.sort_by(|a, b| {
            let score_a = a.count as f32 * a.avg_magnitude;
            let score_b = b.count as f32 * b.avg_magnitude;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        spots.truncate(limit);
        spots
    }

    pub fn surprises_for_memory(&self, memory_id: u64) -> Vec<&SurpriseEvent> {
        self.by_source_memory
            .get(&memory_id)
            .map(|ids| ids.iter().filter_map(|id| self.get(*id)).collect())
            .unwrap_or_default()
    }

    pub fn stats(&self) -> SurpriseStats {
        let total_events = self.events.len();
        let avg_magnitude = if total_events > 0 {
            self.events.iter().map(|e| e.surprise_magnitude).sum::<f32>() / total_events as f32
        } else {
            0.0
        };
        let mut domain_counts: HashMap<&str, usize> = HashMap::new();
        for event in &self.events {
            *domain_counts.entry(&event.domain).or_default() += 1;
        }
        let by_domain: Vec<(String, usize)> = domain_counts
            .into_iter()
            .map(|(d, c)| (d.to_string(), c))
            .collect();

        SurpriseStats {
            total_events,
            avg_magnitude,
            by_domain,
        }
    }

    pub fn count(&self) -> usize {
        self.events.len()
    }
}

impl crate::organ::OrganApply for SurpriseStore {
    /// Organ-owned WAL replay (THEORY.md §8 Phase 2).
    fn apply(&mut self, op: crate::ops::Op) -> Option<crate::ops::Op> {
        use crate::ops::Op;
        match op {
            Op::RecordSurprise(s) => {
                self.replay_record(crate::organ::surprise::SurpriseEvent {
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
                    None
                }
            other => Some(other),
        }
    }
}

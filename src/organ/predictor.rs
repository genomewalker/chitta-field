//! Predictive Memory — sparse Markov chain over memory access sequences.
//!
//! Learns which memories tend to be accessed together and in what order.
//! Predictions pre-warm likely-needed memories and drive sleep replay.
//! Sleep = retrain predictor (restructure expectations from accumulated evidence).
//!
//! References:
//!   - Anderson & Schooler (1991). Reflections of the environment in memory.
//!   - GPT-5.4 + Opus brainstorm (2026-04-11) — access predictor as sleep substrate

use crate::ids::MemoryId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

const RING_BUFFER_SIZE: usize = 256;
const PRUNE_THRESHOLD: f32 = 0.01;
const MAX_TRANSITIONS_PER_SOURCE: usize = 32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPredictor {
    transitions: HashMap<MemoryId, Vec<(MemoryId, f32)>>,
    recent_access: VecDeque<MemoryId>,
    total_transitions: u64,
    last_trained_ms: i64,
}

impl AccessPredictor {
    pub fn new() -> Self {
        Self {
            transitions: HashMap::new(),
            recent_access: VecDeque::with_capacity(RING_BUFFER_SIZE),
            total_transitions: 0,
            last_trained_ms: 0,
        }
    }

    /// Record a memory access on the hot path. Updates the ring buffer
    /// and increments the raw transition count from the previous access.
    pub fn record_access(&mut self, memory_id: MemoryId) {
        if let Some(&prev) = self.recent_access.back() {
            if prev != memory_id {
                let entry = self.transitions.entry(prev).or_default();
                if let Some(pair) = entry.iter_mut().find(|(id, _)| *id == memory_id) {
                    pair.1 += 1.0;
                } else if entry.len() < MAX_TRANSITIONS_PER_SOURCE {
                    entry.push((memory_id, 1.0));
                }
                self.total_transitions += 1;
            }
        }
        if self.recent_access.len() >= RING_BUFFER_SIZE {
            self.recent_access.pop_front();
        }
        self.recent_access.push_back(memory_id);
    }

    /// Get predictions for what memory is likely needed next, given
    /// the most recently accessed memory. Returns up to `k` predictions
    /// sorted by probability descending.
    pub fn predict(&self, k: usize) -> Vec<(MemoryId, f32)> {
        let last = match self.recent_access.back() {
            Some(id) => *id,
            None => return Vec::new(),
        };
        self.predict_from(last, k)
    }

    /// Predict from a specific source memory.
    pub fn predict_from(&self, source: MemoryId, k: usize) -> Vec<(MemoryId, f32)> {
        let entry = match self.transitions.get(&source) {
            Some(e) => e,
            None => return Vec::new(),
        };

        let total: f32 = entry.iter().map(|(_, c)| c).sum();
        if total < 1.0 {
            return Vec::new();
        }

        let mut predictions: Vec<(MemoryId, f32)> = entry
            .iter()
            .map(|(id, count)| (*id, count / total))
            .filter(|(_, prob)| *prob >= PRUNE_THRESHOLD)
            .collect();

        predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        predictions.truncate(k);
        predictions
    }

    /// Check if a memory was predicted (used by PredictionFactor scoring).
    pub fn is_predicted(&self, memory_id: MemoryId) -> Option<f32> {
        let predictions = self.predict(16);
        predictions.iter().find(|(id, _)| *id == memory_id).map(|(_, prob)| *prob)
    }

    /// Sleep consolidation: rebuild transition matrix from the full access ring buffer.
    /// Normalizes probabilities and prunes low-probability transitions.
    /// This IS sleep replay — retraining restructures expectations.
    pub fn retrain(&mut self, now_ms: i64) {
        let mut new_transitions: HashMap<MemoryId, HashMap<MemoryId, f32>> = HashMap::new();

        let accesses: Vec<MemoryId> = self.recent_access.iter().copied().collect();
        for window in accesses.windows(2) {
            let (src, dst) = (window[0], window[1]);
            if src != dst {
                *new_transitions.entry(src).or_default().entry(dst).or_insert(0.0) += 1.0;
            }
        }

        // Also incorporate existing transitions with decay
        for (src, targets) in &self.transitions {
            let entry = new_transitions.entry(*src).or_default();
            for (dst, count) in targets {
                *entry.entry(*dst).or_insert(0.0) += count * 0.5; // 50% decay on old data
            }
        }

        // Normalize and prune
        self.transitions.clear();
        for (src, targets) in new_transitions {
            let total: f32 = targets.values().sum();
            if total < 1.0 {
                continue;
            }
            let mut normalized: Vec<(MemoryId, f32)> = targets
                .into_iter()
                .map(|(id, count)| (id, count / total))
                .filter(|(_, prob)| *prob >= PRUNE_THRESHOLD)
                .collect();
            normalized.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            normalized.truncate(MAX_TRANSITIONS_PER_SOURCE);
            // Store as raw counts again (will be normalized on predict)
            let renormalized: Vec<(MemoryId, f32)> = normalized
                .into_iter()
                .map(|(id, prob)| (id, prob * total))
                .collect();
            self.transitions.insert(src, renormalized);
        }

        self.last_trained_ms = now_ms;
    }

    pub fn total_transitions(&self) -> u64 {
        self.total_transitions
    }

    pub fn last_trained_ms(&self) -> i64 {
        self.last_trained_ms
    }

    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    pub fn recent_access_len(&self) -> usize {
        self.recent_access.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_predict() {
        let mut pred = AccessPredictor::new();
        // Create a pattern: 1 → 2 → 3, repeated
        for _ in 0..10 {
            pred.record_access(1);
            pred.record_access(2);
            pred.record_access(3);
        }

        // After accessing 1, should predict 2
        // Manually set last access to 1
        let preds = pred.predict_from(1, 5);
        assert!(!preds.is_empty());
        assert_eq!(preds[0].0, 2);

        // After accessing 2, should predict 3
        let preds = pred.predict_from(2, 5);
        assert!(!preds.is_empty());
        assert_eq!(preds[0].0, 3);
    }

    #[test]
    fn test_retrain() {
        let mut pred = AccessPredictor::new();
        for _ in 0..20 {
            pred.record_access(10);
            pred.record_access(20);
        }

        pred.retrain(1000);
        assert!(pred.last_trained_ms() == 1000);

        let preds = pred.predict_from(10, 5);
        assert!(!preds.is_empty());
        assert_eq!(preds[0].0, 20);
    }

    #[test]
    fn test_is_predicted() {
        let mut pred = AccessPredictor::new();
        for _ in 0..10 {
            pred.record_access(100);
            pred.record_access(200);
        }
        pred.record_access(100); // last access is 100

        assert!(pred.is_predicted(200).is_some());
        assert!(pred.is_predicted(999).is_none());
    }

    #[test]
    fn test_ring_buffer_limit() {
        let mut pred = AccessPredictor::new();
        for i in 0..300u64 {
            pred.record_access(i);
        }
        assert_eq!(pred.recent_access_len(), RING_BUFFER_SIZE);
    }
}

//! Asymmetric Hopfield Network for association-based pattern completion.
//!
//! Unifies co-activation statistics and prototype transitions into a single
//! energy-based attractor network with asymmetric couplings. FEP §3.2.
//!
//! Energy: E = -Σ_ij W_ij s_i s_j  (W_ij ≠ W_ji for non-equilibrium dynamics)
//!
//! The coupling matrix W is sparse, keyed by memory pairs with co-activation.
//! Settling iterates asynchronous updates until convergence or max steps.

use crate::ids::MemoryId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sparse asymmetric coupling between two memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coupling {
    pub weight: f32,        // W_ij (i → j coupling strength)
    pub co_activation: u32, // how often these were co-retrieved
    pub last_seen_ms: i64,
}

/// Asymmetric Hopfield network over memory activations.
#[derive(Serialize, Deserialize)]
pub struct HopfieldNetwork {
    /// Sparse coupling matrix: (src, dst) → coupling.
    /// W_ij ≠ W_ji in general (asymmetric, non-equilibrium).
    couplings: HashMap<(MemoryId, MemoryId), Coupling>,
    /// Self-bias (threshold) per memory.
    bias: HashMap<MemoryId, f32>,
    /// Running count of settle() calls for statistics.
    settle_count: u64,
}

impl Default for HopfieldNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl HopfieldNetwork {
    pub fn new() -> Self {
        Self {
            couplings: HashMap::new(),
            bias: HashMap::new(),
            settle_count: 0,
        }
    }

    /// Strengthen directed coupling from src → dst.
    /// Called when src is retrieved before dst in the same recall.
    /// Forward coupling gets full delta, reverse gets attenuated (0.3x). FEP §3.2.
    pub fn strengthen(&mut self, src: MemoryId, dst: MemoryId, delta: f32, ts_ms: i64) {
        if src == dst {
            return;
        }
        // Forward: src → dst
        let fwd = self.couplings.entry((src, dst)).or_insert(Coupling {
            weight: 0.0,
            co_activation: 0,
            last_seen_ms: ts_ms,
        });
        fwd.weight = (fwd.weight + delta).min(1.0);
        fwd.co_activation += 1;
        fwd.last_seen_ms = ts_ms;

        // Reverse: dst → src (attenuated)
        let rev = self.couplings.entry((dst, src)).or_insert(Coupling {
            weight: 0.0,
            co_activation: 0,
            last_seen_ms: ts_ms,
        });
        rev.weight = (rev.weight + delta * 0.3).min(1.0);
        rev.co_activation += 1;
        rev.last_seen_ms = ts_ms;
    }

    /// Record co-retrieval of a batch of memories (from a single recall).
    /// Strengthens pairwise couplings respecting order (earlier → later).
    pub fn record_co_retrieval(&mut self, ids: &[MemoryId], ts_ms: i64) {
        let delta = 0.02f32;
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                self.strengthen(ids[i], ids[j], delta, ts_ms);
            }
        }
    }

    /// Settle: given seed activations, iterate until convergence.
    /// Returns the settled activation pattern (memory_id → activation).
    ///
    /// Update rule (asynchronous): s_i ← σ(Σ_j W_ij s_j + b_i)
    /// where σ is sigmoid, ensuring activations stay in [0, 1].
    pub fn settle(
        &mut self,
        seed: &HashMap<MemoryId, f32>,
        max_steps: usize,
    ) -> HashMap<MemoryId, f32> {
        self.settle_count += 1;
        let mut state = seed.clone();
        let epsilon = 0.001f32;

        for _step in 0..max_steps {
            // Expand: discover neighbors of all currently activated nodes
            let active_ids: Vec<MemoryId> = state.keys().copied().collect();
            for &id in &active_ids {
                if state.get(&id).copied().unwrap_or(0.0) < 0.01 {
                    continue; // skip near-zero nodes
                }
                for (&(src, dst), _) in &self.couplings {
                    if src == id && !state.contains_key(&dst) {
                        state.insert(dst, 0.0);
                    }
                }
            }

            let mut max_delta = 0.0f32;
            let ids: Vec<MemoryId> = state.keys().copied().collect();

            for &id in &ids {
                // Seeds keep their activation (clamped)
                if seed.contains_key(&id) {
                    continue;
                }
                let mut input = self.bias.get(&id).copied().unwrap_or(0.0);
                for &other_id in &ids {
                    if other_id == id {
                        continue;
                    }
                    if let Some(coupling) = self.couplings.get(&(other_id, id)) {
                        input += coupling.weight * state.get(&other_id).copied().unwrap_or(0.0);
                    }
                }
                let new_val = sigmoid(input);
                let old_val = state.get(&id).copied().unwrap_or(0.0);
                max_delta = max_delta.max((new_val - old_val).abs());
                state.insert(id, new_val);
            }

            if max_delta < epsilon {
                break;
            }
        }

        state
    }

    /// Get the top-N activated memories from a settled state.
    pub fn top_activated(state: &HashMap<MemoryId, f32>, n: usize) -> Vec<(MemoryId, f32)> {
        let mut items: Vec<(MemoryId, f32)> = state.iter().map(|(&k, &v)| (k, v)).collect();
        items.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        items.truncate(n);
        items
    }

    /// Decay all couplings by a factor. Called periodically to prevent saturation.
    pub fn decay_all(&mut self, factor: f32) {
        self.couplings.retain(|_, c| {
            c.weight *= factor;
            c.weight > 0.001
        });
    }

    /// Remove all couplings involving a specific memory.
    pub fn remove_memory(&mut self, memory_id: MemoryId) {
        self.couplings.retain(|&(a, b), _| a != memory_id && b != memory_id);
        self.bias.remove(&memory_id);
    }

    pub fn coupling_count(&self) -> usize {
        self.couplings.len()
    }

    pub fn settle_count(&self) -> u64 {
        self.settle_count
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asymmetric_coupling() {
        let mut net = HopfieldNetwork::new();
        net.strengthen(1, 2, 0.5, 1000);

        // Forward should be stronger than reverse
        let fwd = net.couplings.get(&(1, 2)).unwrap().weight;
        let rev = net.couplings.get(&(2, 1)).unwrap().weight;
        assert!(fwd > rev, "forward {} should be > reverse {}", fwd, rev);
        assert!((fwd - 0.5).abs() < 0.001);
        assert!((rev - 0.15).abs() < 0.001);
    }

    #[test]
    fn test_settle_convergence() {
        let mut net = HopfieldNetwork::new();
        // Create a simple attractor: 1→2→3
        net.strengthen(1, 2, 0.8, 1000);
        net.strengthen(2, 3, 0.8, 1000);

        let mut seed = HashMap::new();
        seed.insert(1, 1.0);

        let settled = net.settle(&seed, 10);
        // Memory 2 should be activated (strongly coupled from 1)
        let act_2 = settled.get(&2).copied().unwrap_or(0.0);
        assert!(act_2 > 0.3, "memory 2 should be activated via 1→2, got {}", act_2);
        // Memory 3: 2-hop propagation with asymmetric attenuation, so weaker
        let act_3 = settled.get(&3).copied().unwrap_or(0.0);
        assert!(act_3 > act_2 * 0.1,
            "memory 3 should have nonzero activation via 2→3, got {}", act_3);
    }

    #[test]
    fn test_co_retrieval_ordering() {
        let mut net = HopfieldNetwork::new();
        net.record_co_retrieval(&[10, 20, 30], 1000);

        // 10→20 should be stronger than 20→10
        let fwd = net.couplings.get(&(10, 20)).unwrap().weight;
        let rev = net.couplings.get(&(20, 10)).unwrap().weight;
        assert!(fwd > rev);
    }
}

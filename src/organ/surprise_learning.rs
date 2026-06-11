use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ── Configuration ─────────────────────────────────────────────────────────

pub struct SurpriseLearningConfig {
    pub ignore_threshold: f32,
    pub gate_credit: f32,
    pub min_streak: u8,
    pub credit_decay: f32,
    pub post_apply_decay: f32,
    pub min_strength: f32,
    pub max_strength: f32,
    pub max_delta: f32,
    pub neg_hard_threshold: f32,
    pub neg_soft_threshold: f32,
    pub pos_threshold: f32,
    pub repeated_failure_needed: u8,
    pub repeated_failure_window: u8,
}

impl Default for SurpriseLearningConfig {
    fn default() -> Self {
        Self {
            ignore_threshold: 0.20,
            gate_credit: 0.75,
            min_streak: 2,
            credit_decay: 0.85,
            post_apply_decay: 0.35,
            min_strength: 0.05,
            max_strength: 1.50,
            max_delta: 0.08,
            neg_hard_threshold: 0.55,
            neg_soft_threshold: 0.25,
            pos_threshold: 0.30,
            repeated_failure_needed: 2,
            repeated_failure_window: 8,
        }
    }
}

// ── State ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurpriseLearningState {
    pub memory_id: u64,
    pub credit: f32,
    pub last_dir: i8,
    pub same_dir_streak: u8,
    pub last_surprise_id: u64,
    pub updated_ms: i64,
}

pub struct SurpriseCreditResult {
    pub memory_id: u64,
    pub strength_delta: f32,
    pub credit_before: f32,
    pub credit_after: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurpriseLearningStats {
    pub tracked_memories: usize,
    pub tracked_failure_pairs: usize,
    pub total_gates_passed: u64,
    pub total_credits_updated: u64,
}

// ── Store ─────────────────────────────────────────────────────────────────

pub struct SurpriseLearningStore {
    by_memory: HashMap<u64, SurpriseLearningState>,
    failure_tracker: HashMap<(String, String), VecDeque<u64>>,
    config: SurpriseLearningConfig,
    total_gates_passed: u64,
    total_credits_updated: u64,
}

impl SurpriseLearningStore {
    pub fn new() -> Self {
        Self {
            by_memory: HashMap::new(),
            failure_tracker: HashMap::new(),
            config: SurpriseLearningConfig::default(),
            total_gates_passed: 0,
            total_credits_updated: 0,
        }
    }

    /// Quadratic evidence function: ignores low surprise, amplifies high.
    fn compute_evidence(&self, magnitude: f32) -> f32 {
        if magnitude < self.config.ignore_threshold {
            return 0.0;
        }
        let x = (magnitude - self.config.ignore_threshold)
            / (1.0 - self.config.ignore_threshold);
        x * x
    }

    /// Update rolling credit for a memory. dir = +1 (actual/correct), -1 (expected/wrong).
    /// Returns Some(result) if the hysteresis gate passes and strength should change.
    pub fn update_credit(
        &mut self,
        memory_id: u64,
        surprise_id: u64,
        magnitude: f32,
        dir: i8,
        now_ms: i64,
    ) -> Option<SurpriseCreditResult> {
        let e = self.compute_evidence(magnitude);
        if e == 0.0 {
            return None;
        }

        self.total_credits_updated += 1;

        let state = self.by_memory.entry(memory_id).or_insert_with(|| {
            SurpriseLearningState {
                memory_id,
                credit: 0.0,
                last_dir: 0,
                same_dir_streak: 0,
                last_surprise_id: 0,
                updated_ms: 0,
            }
        });

        let credit_before = state.credit;
        state.credit = self.config.credit_decay * state.credit + (dir as f32) * e;

        if state.last_dir == dir {
            state.same_dir_streak = state.same_dir_streak.saturating_add(1).min(8);
        } else {
            state.same_dir_streak = 1;
        }
        state.last_dir = dir;
        state.last_surprise_id = surprise_id;
        state.updated_ms = now_ms;

        // Hysteresis gate
        if state.credit.abs() < self.config.gate_credit {
            return None;
        }
        if state.same_dir_streak < self.config.min_streak {
            return None;
        }

        self.total_gates_passed += 1;

        // Compute strength delta
        let excess = (state.credit.abs() - self.config.gate_credit).min(1.0);
        let delta_mag = self.config.max_delta.min(0.02 + 0.06 * excess);
        let delta = if state.credit > 0.0 { delta_mag } else { -delta_mag };

        // Post-apply bleed
        let credit_after = state.credit * self.config.post_apply_decay;
        state.credit = credit_after;

        Some(SurpriseCreditResult {
            memory_id,
            strength_delta: delta,
            credit_before,
            credit_after,
        })
    }

    /// WAL replay: restore a credit state directly.
    pub fn replay_credit(&mut self, state: SurpriseLearningState) {
        self.by_memory.insert(state.memory_id, state);
    }

    /// Check if negative feedback should be sent to integration kernel (Move 2).
    pub fn should_send_negative_feedback(
        &self,
        domain: &str,
        source: &str,
        magnitude: f32,
    ) -> bool {
        if magnitude >= self.config.neg_hard_threshold {
            return true;
        }
        if magnitude >= self.config.neg_soft_threshold {
            let key = (domain.to_string(), source.to_string());
            if let Some(recent) = self.failure_tracker.get(&key) {
                return recent.len() >= self.config.repeated_failure_needed as usize;
            }
        }
        false
    }

    /// Check if positive feedback should be sent (Move 2).
    pub fn should_send_positive_feedback(&self, magnitude: f32) -> bool {
        magnitude >= self.config.pos_threshold
    }

    /// Record a failure in the rolling window for a (domain, source) pair.
    pub fn record_failure(&mut self, domain: &str, source: &str, surprise_id: u64) {
        let key = (domain.to_string(), source.to_string());
        let window = self
            .failure_tracker
            .entry(key)
            .or_insert_with(VecDeque::new);
        window.push_back(surprise_id);
        while window.len() > self.config.repeated_failure_window as usize {
            window.pop_front();
        }
    }

    pub fn get_state(&self, memory_id: u64) -> Option<&SurpriseLearningState> {
        self.by_memory.get(&memory_id)
    }

    pub fn stats(&self) -> SurpriseLearningStats {
        SurpriseLearningStats {
            tracked_memories: self.by_memory.len(),
            tracked_failure_pairs: self.failure_tracker.len(),
            total_gates_passed: self.total_gates_passed,
            total_credits_updated: self.total_credits_updated,
        }
    }
}

impl crate::organ::OrganApply for SurpriseLearningStore {
    /// Organ-owned WAL replay (THEORY.md §8 Phase 2).
    fn apply(&mut self, op: crate::ops::Op) -> Option<crate::ops::Op> {
        use crate::ops::Op;
        match op {
            Op::UpdateSurpriseCredit(c) => {
                self.replay_credit(
                    crate::organ::surprise_learning::SurpriseLearningState {
                        memory_id: c.memory_id,
                        credit: c.credit,
                        last_dir: c.last_dir,
                        same_dir_streak: c.same_dir_streak,
                        last_surprise_id: c.last_surprise_id,
                        updated_ms: c.updated_ms,
                    },
                );
                    None
                }
            other => Some(other),
        }
    }
}

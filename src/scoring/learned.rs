use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedFactorWeight {
    pub name: String,
    pub delta: f32,
    pub min_delta: f32,
    pub max_delta: f32,
    pub last_updated_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedScoringModel {
    pub model_version: u64,
    pub baseline_version: String,
    pub factors: HashMap<String, LearnedFactorWeight>,
    pub ewma_loss: f32,
    pub outcome_count: u64,
    pub last_calibrated_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedScoringStats {
    pub model_version: u64,
    pub baseline_version: String,
    pub factor_count: usize,
    pub ewma_loss: f32,
    pub outcome_count: u64,
    pub last_calibrated_ms: i64,
    pub factors: Vec<LearnedFactorWeight>,
}

impl LearnedScoringModel {
    pub fn new(baseline_version: String) -> Self {
        Self {
            model_version: 0,
            baseline_version,
            factors: HashMap::new(),
            ewma_loss: 0.0,
            outcome_count: 0,
            last_calibrated_ms: 0,
        }
    }

    /// Compute effective weight: baseline + learned delta, clamped.
    pub fn effective_weight(&self, factor_name: &str, baseline_value: f32) -> f32 {
        if let Some(lw) = self.factors.get(factor_name) {
            let effective = baseline_value + lw.delta;
            let min = baseline_value + lw.min_delta;
            let max = baseline_value + lw.max_delta;
            effective.clamp(min, max)
        } else {
            baseline_value
        }
    }

    /// Get learned delta for a factor (0.0 if not learned).
    pub fn get_delta(&self, factor_name: &str) -> f32 {
        self.factors
            .get(factor_name)
            .map(|lw| lw.delta)
            .unwrap_or(0.0)
    }

    /// Apply a batch update from the calibration job.
    pub fn apply_update(
        &mut self,
        weights_json: &str,
        model_version: u64,
        ewma_loss: f32,
        outcome_count: u64,
        now_ms: i64,
    ) {
        if let Ok(updates) = serde_json::from_str::<HashMap<String, f32>>(weights_json) {
            for (name, delta) in updates {
                let entry = self.factors.entry(name.clone()).or_insert_with(|| {
                    LearnedFactorWeight {
                        name,
                        delta: 0.0,
                        min_delta: -0.5,
                        max_delta: 0.5,
                        last_updated_ms: 0,
                    }
                });
                entry.delta = delta.clamp(entry.min_delta, entry.max_delta);
                entry.last_updated_ms = now_ms;
            }
        }
        self.model_version = model_version;
        self.ewma_loss = ewma_loss;
        self.outcome_count = outcome_count;
        self.last_calibrated_ms = now_ms;
    }

    /// WAL replay: restore full model state.
    pub fn replay_update(&mut self, model: LearnedScoringModel) {
        *self = model;
    }

    pub fn stats(&self) -> LearnedScoringStats {
        LearnedScoringStats {
            model_version: self.model_version,
            baseline_version: self.baseline_version.clone(),
            factor_count: self.factors.len(),
            ewma_loss: self.ewma_loss,
            outcome_count: self.outcome_count,
            last_calibrated_ms: self.last_calibrated_ms,
            factors: self.factors.values().cloned().collect(),
        }
    }
}

impl crate::organ::OrganApply for LearnedScoringModel {
    /// Organ-owned WAL replay (THEORY.md §8 Phase 2).
    fn apply(&mut self, op: crate::ops::Op) -> Option<crate::ops::Op> {
        use crate::ops::Op;
        match op {
            Op::UpdateScorerModel(m) => {
                self.apply_update(
                    &m.weights_json,
                    m.model_version,
                    m.mean_loss,
                    m.outcome_count,
                    m.applied_at_ms,
                );
                    None
                }
            other => Some(other),
        }
    }
}

use crate::learner::bandit::GaussianPrior;
use std::collections::HashMap;

pub struct ContextLearner {
    /// Per-session-type: Gaussian prior over optimal window size
    window_priors: HashMap<String, GaussianPrior>,
    default_window: usize,
}

impl ContextLearner {
    pub fn new() -> Self {
        Self {
            window_priors: HashMap::new(),
            default_window: 10,
        }
    }

    /// Get recommended window size for a session type.
    pub fn recommended_window(&self, session_type: &str) -> usize {
        self.window_priors
            .get(session_type)
            .map(|p| p.mean().round() as usize)
            .unwrap_or(self.default_window)
            .clamp(3, 50)
    }

    /// Record that a window of `size` led to `outcome` quality (0.0-1.0).
    pub fn record_outcome(&mut self, session_type: &str, size: usize, outcome: f32) {
        // If outcome is good, update prior toward this size
        // Weight the update by the outcome quality
        let weighted_size = size as f64 * outcome as f64
            + self.recommended_window(session_type) as f64 * (1.0 - outcome as f64);
        self.window_priors
            .entry(session_type.to_string())
            .or_insert(GaussianPrior::new(10.0, 3.0))
            .update(weighted_size);
    }
}

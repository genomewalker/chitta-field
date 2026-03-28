pub mod bandit;
pub mod context;
pub mod domain_reliability;
pub mod plasticity;
pub mod route;

pub use bandit::{BetaPrior, GaussianPrior};
pub use context::ContextLearner;
pub use domain_reliability::DomainReliability;
pub use plasticity::PlasticityLearner;
pub use route::{QueryIntent, Route, RouteLearner};

/// Manages all learners together.
pub struct LearnerSet {
    pub plasticity: PlasticityLearner,
    pub route: RouteLearner,
    pub context: ContextLearner,
    pub domain_reliability: DomainReliability,
}

impl LearnerSet {
    pub fn new() -> Self {
        Self {
            plasticity: PlasticityLearner::new(),
            route: RouteLearner::new(),
            context: ContextLearner::new(),
            domain_reliability: DomainReliability::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plasticity_frequent_access() {
        let mut p = PlasticityLearner::new();
        // Access every hour for 10 times
        let start = 0i64;
        for i in 0..10 {
            p.record_access(1, start + i * 3_600_000);
        }
        let rate = p.recommended_decay_rate(1);
        // Frequently accessed should decay slower than default
        assert!(
            rate < 0.001,
            "frequent access should lower decay rate, got {}",
            rate
        );
    }

    #[test]
    fn test_route_learner_feedback_shifts_preference() {
        let mut r = RouteLearner::new();
        // Give strong positive feedback for Keyword route on Code queries
        for i in 0..20u64 {
            let (ep, _route) = r.select_route(QueryIntent::Code, i * 1000);
            // Force feedback as if Keyword always worked
            r.arms
                .entry((QueryIntent::Code, Route::Keyword))
                .or_insert(BetaPrior::new())
                .update(1.0);
            r.pending.remove(&ep);
        }
        let best = r.best_route(&QueryIntent::Code);
        assert_eq!(best, Route::Keyword);
    }

    #[test]
    fn test_context_learner_convergence() {
        let mut c = ContextLearner::new();
        // Record that window=20 is great for "code" sessions
        for _ in 0..10 {
            c.record_outcome("code", 20, 0.9);
        }
        let w = c.recommended_window("code");
        // Should have shifted toward ~20
        assert!(w > 12, "should converge toward 20, got {}", w);
    }

    #[test]
    fn test_detect_intent() {
        assert_eq!(
            RouteLearner::detect_intent("what happened yesterday"),
            QueryIntent::Temporal
        );
        assert_eq!(
            RouteLearner::detect_intent("fix the rust code in main.rs"),
            QueryIntent::FileRef
        );
        assert_eq!(
            RouteLearner::detect_intent("that was wrong actually"),
            QueryIntent::Correction
        );
    }
}

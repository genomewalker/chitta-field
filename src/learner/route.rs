use crate::learner::bandit::BetaPrior;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Route {
    Semantic, // HNSW only
    Keyword,  // BM25 only
    Temporal, // Time-based
    Artifact, // File-based
    Hybrid,   // Semantic + BM25
    Full,     // All channels
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum QueryIntent {
    General,    // Default
    Code,       // Code/programming queries
    Temporal,   // Time-related ("yesterday", "last week")
    FileRef,    // File/path references
    Correction, // Factual corrections
}

pub struct RouteLearner {
    /// Beta prior per (intent, route) pair
    pub arms: HashMap<(QueryIntent, Route), BetaPrior>,
    /// Episode log: query_id -> (intent, route chosen, pending outcome)
    pub pending: HashMap<u64, (QueryIntent, Route)>,
    next_episode_id: u64,
}

impl RouteLearner {
    pub fn new() -> Self {
        Self {
            arms: HashMap::new(),
            pending: HashMap::new(),
            next_episode_id: 1,
        }
    }

    /// Select the best route for a query intent using Thompson sampling.
    /// Returns (episode_id, selected_route).
    pub fn select_route(&mut self, intent: QueryIntent, now_ms: u64) -> (u64, Route) {
        let routes = [
            Route::Semantic,
            Route::Keyword,
            Route::Temporal,
            Route::Artifact,
            Route::Hybrid,
            Route::Full,
        ];
        let seed = now_ms ^ (self.next_episode_id * 2654435761);

        let best = routes
            .iter()
            .max_by(|a, b| {
                let sa = self
                    .arms
                    .entry((intent.clone(), (*a).clone()))
                    .or_insert(BetaPrior::new())
                    .sample(seed);
                let sb = self
                    .arms
                    .entry((intent.clone(), (*b).clone()))
                    .or_insert(BetaPrior::new())
                    .sample(seed.wrapping_add(1));
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
            .clone();

        let episode_id = self.next_episode_id;
        self.next_episode_id += 1;
        self.pending.insert(episode_id, (intent, best.clone()));
        (episode_id, best)
    }

    /// Apply outcome feedback to a pending episode.
    pub fn feedback(&mut self, episode_id: u64, reward: f32) {
        if let Some((intent, route)) = self.pending.remove(&episode_id) {
            self.arms
                .entry((intent, route))
                .or_insert(BetaPrior::new())
                .update(reward);
        }
    }

    /// Get the current best route for an intent (deterministic, highest mean).
    pub fn best_route(&self, intent: &QueryIntent) -> Route {
        let routes = [
            Route::Semantic,
            Route::Keyword,
            Route::Temporal,
            Route::Artifact,
            Route::Hybrid,
            Route::Full,
        ];
        routes
            .iter()
            .max_by(|a, b| {
                let sa = self
                    .arms
                    .get(&(intent.clone(), (*a).clone()))
                    .map(|p| p.mean())
                    .unwrap_or(0.5);
                let sb = self
                    .arms
                    .get(&(intent.clone(), (*b).clone()))
                    .map(|p| p.mean())
                    .unwrap_or(0.5);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
            .clone()
    }

    /// Detect query intent from text heuristics.
    pub fn detect_intent(query: &str) -> QueryIntent {
        let q = query.to_lowercase();
        if q.contains("yesterday")
            || q.contains("last week")
            || q.contains("ago")
            || q.contains("when")
        {
            QueryIntent::Temporal
        } else if q.contains(".cpp")
            || q.contains(".rs")
            || q.contains(".py")
            || q.contains("fn ")
            || q.contains("impl ")
        {
            QueryIntent::FileRef
        } else if q.contains("wrong")
            || q.contains("correct")
            || q.contains("actually")
            || q.contains("mistake")
        {
            QueryIntent::Correction
        } else if q.contains("rust")
            || q.contains("function")
            || q.contains("struct")
            || q.contains("code")
        {
            QueryIntent::Code
        } else {
            QueryIntent::General
        }
    }
}

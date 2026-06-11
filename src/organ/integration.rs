//! Integration Kernel — recall source arbitration with learned weights.
//!
//! Learns which recall sources (semantic, keyword, temporal, constraint, etc.)
//! are most useful for different query domains. Provides justification traces
//! and feedback-driven weight adaptation. Reference: mixture-of-experts
//! gating (Shazeer et al. 2017), Bayesian model averaging.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

const TRACE_RING_SIZE: usize = 128;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceWeight {
    pub source: String,
    pub weight: f32,
    pub query_domain: String,
    pub success_count: u64,
    pub total_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationTrace {
    pub query_sketch: String,
    pub sources_used: Vec<String>,
    pub top_results: Vec<(u64, f32, String)>,
    pub timestamp_ms: i64,
}

pub struct IntegrationStats {
    pub total_queries: u64,
    pub source_rates: Vec<(String, String, f32, u64)>,
}

pub struct IntegrationKernel {
    source_weights: HashMap<(String, String), SourceWeight>,
    traces: VecDeque<IntegrationTrace>,
    total_queries: u64,
}

impl IntegrationKernel {
    pub fn new() -> Self {
        Self {
            source_weights: HashMap::new(),
            traces: VecDeque::with_capacity(TRACE_RING_SIZE),
            total_queries: 0,
        }
    }

    pub fn record_feedback(
        &mut self,
        query_domain: &str,
        source: &str,
        was_useful: bool,
    ) -> SourceWeight {
        let key = (source.to_string(), query_domain.to_string());
        let entry = self.source_weights.entry(key).or_insert_with(|| SourceWeight {
            source: source.to_string(),
            weight: 1.0,
            query_domain: query_domain.to_string(),
            success_count: 0,
            total_count: 0,
        });
        entry.total_count += 1;
        if was_useful {
            entry.success_count += 1;
        }
        entry.weight =
            (entry.success_count as f32 / entry.total_count as f32 * 2.0).clamp(0.0, 2.0);
        entry.clone()
    }

    pub fn replay_feedback(
        &mut self,
        source: String,
        query_domain: String,
        weight: f32,
        success_count: u64,
        total_count: u64,
    ) {
        let key = (source.clone(), query_domain.clone());
        self.source_weights.insert(
            key,
            SourceWeight {
                source,
                weight,
                query_domain,
                success_count,
                total_count,
            },
        );
    }

    pub fn update_source_weight(
        &mut self,
        source: &str,
        query_domain: &str,
        weight: f32,
    ) -> bool {
        let key = (source.to_string(), query_domain.to_string());
        if let Some(entry) = self.source_weights.get_mut(&key) {
            entry.weight = weight.clamp(0.0, 2.0);
            true
        } else {
            self.source_weights.insert(
                key,
                SourceWeight {
                    source: source.to_string(),
                    weight: weight.clamp(0.0, 2.0),
                    query_domain: query_domain.to_string(),
                    success_count: 0,
                    total_count: 0,
                },
            );
            true
        }
    }

    pub fn replay_update_weight(&mut self, source: String, query_domain: String, weight: f32) {
        let key = (source.clone(), query_domain.clone());
        if let Some(entry) = self.source_weights.get_mut(&key) {
            entry.weight = weight;
        } else {
            self.source_weights.insert(
                key,
                SourceWeight {
                    source,
                    weight,
                    query_domain,
                    success_count: 0,
                    total_count: 0,
                },
            );
        }
    }

    pub fn get_source_weights(&self, domain: Option<&str>) -> Vec<&SourceWeight> {
        self.source_weights
            .values()
            .filter(|w| domain.map_or(true, |d| w.query_domain == d))
            .collect()
    }

    pub fn record_trace(
        &mut self,
        query_sketch: String,
        sources_used: Vec<String>,
        top_results: Vec<(u64, f32, String)>,
        now_ms: i64,
    ) {
        self.total_queries += 1;
        if self.traces.len() >= TRACE_RING_SIZE {
            self.traces.pop_front();
        }
        self.traces.push_back(IntegrationTrace {
            query_sketch,
            sources_used,
            top_results,
            timestamp_ms: now_ms,
        });
    }

    pub fn get_traces(&self, limit: usize) -> Vec<&IntegrationTrace> {
        self.traces.iter().rev().take(limit).collect()
    }

    pub fn weight_for(&self, source: &str, domain: &str) -> f32 {
        self.source_weights
            .get(&(source.to_string(), domain.to_string()))
            .map(|w| w.weight)
            .unwrap_or(1.0)
    }

    pub fn stats(&self) -> IntegrationStats {
        let source_rates: Vec<(String, String, f32, u64)> = self
            .source_weights
            .values()
            .map(|w| {
                let rate = if w.total_count > 0 {
                    w.success_count as f32 / w.total_count as f32
                } else {
                    0.0
                };
                (
                    w.source.clone(),
                    w.query_domain.clone(),
                    rate,
                    w.total_count,
                )
            })
            .collect();

        IntegrationStats {
            total_queries: self.total_queries,
            source_rates,
        }
    }

    pub fn total_queries(&self) -> u64 {
        self.total_queries
    }
}

impl crate::organ::OrganApply for IntegrationKernel {
    /// Organ-owned WAL replay (THEORY.md §8 Phase 2).
    fn apply(&mut self, op: crate::ops::Op) -> Option<crate::ops::Op> {
        use crate::ops::Op;
        match op {
            Op::UpdateSourceWeight(w) => {
                self.replay_update_weight(w.source, w.query_domain, w.weight);
                    None
                }
            Op::RecordFeedback(f) => {
                self.replay_feedback(
                    f.source,
                    f.query_domain,
                    f.new_weight,
                    f.success_count,
                    f.total_count,
                );
                    None
                }
            other => Some(other),
        }
    }
}

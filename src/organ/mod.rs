pub mod analytics;
pub mod artifact;
pub mod cdawg;
pub mod event_tape;
pub mod callgraph;
pub mod codefile;
pub mod cortex;
pub mod hopfield;
pub mod keyword;
pub mod lite_encoder;
pub mod msg;
pub mod pq;
pub mod prototype;
pub mod session;
pub mod symbol;
pub mod task;
pub mod temporal;
pub mod theme_organ;
pub mod transcript;
pub mod triplet;
pub mod user_model;
pub mod skill;
pub mod agent;
pub mod constraint;
pub mod trigger;
pub mod predictor;
pub mod surprise;
pub mod epistemic_debt;
pub mod integration;
pub mod surprise_learning;
pub mod wisdom_promotion;
pub mod intervention;
pub mod agent_protocol;
pub mod wisdom_lineage;
pub mod symbol_events;
pub mod sequitur;
pub mod refutation_ledger;
pub mod intervention_store;
pub mod decision_tape;
pub mod hypothesis_market;
pub mod turiya_monitor;
pub mod fep_prior;

pub mod memory_kind;
pub mod query_router;

pub mod provenance;
pub mod reconciler;
pub mod observer;
pub mod interaction_ledger;
pub mod predicate_store;

/// Organ-owned WAL replay (THEORY.md §8 Phase 2): an organ applies the op
/// variants it owns and consumes them (returns None); everything else passes
/// through (Some(op)) to the next organ or the central multi-structure match
/// in apply_op. Taking `Op` by value preserves the replay path's move
/// semantics — no clones.
pub(crate) trait OrganApply {
    fn apply(&mut self, op: crate::ops::Op) -> Option<crate::ops::Op>;
}

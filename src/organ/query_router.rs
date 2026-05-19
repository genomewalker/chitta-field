/// CEC Phase 16 — CPU-Native Query Router.
///
/// Parses a typed RecallRequest into a dispatch strategy without any LLM call.
/// Only NeedsDisambiguation (unbound slots in the typed grammar) escalates to the LLM.
use serde::{Serialize, Deserialize};
use crate::organ::memory_kind::MemoryKind;

/// Fully-typed recall request parsed from caller inputs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecallRequest {
    pub kind_filter:   Option<MemoryKind>,
    pub realm:         Option<String>,
    /// Exact triplet subject — compiles to relational lookup.
    pub subject:       Option<String>,
    /// Exact triplet predicate — compiles to relational lookup.
    pub predicate:     Option<String>,
    /// Fuzzy free-text — compiles to ANN + BM25 lane.
    pub freetext:      Option<String>,
    pub time_from_ms:  Option<i64>,
    pub time_to_ms:    Option<i64>,
    /// CEC causal query — CDAWG antecedent lookup.
    pub causal_tool:   Option<String>,
    pub causal_entity: Option<String>,
    pub k: usize,
}

/// Named hole left by the typed grammar — requires LLM disambiguation.
#[derive(Debug)]
pub struct UnboundSlot {
    pub name:    &'static str,
    pub context: String,
}

/// Dispatch strategy derived from a RecallRequest.
#[derive(Debug)]
pub enum DispatchKind {
    /// Pure relational: triplet subject/predicate lookup. Zero token cost.
    Exact,
    /// ANN + BM25 over embeddings.
    Fuzzy,
    /// EventTape time-range query.
    Temporal,
    /// CDAWG causal antecedent query.
    Causal,
    /// Relational + fuzzy join.
    Hybrid,
    /// Parse left named holes — caller must fill via LLM before re-routing.
    NeedsDisambiguation(Vec<UnboundSlot>),
}

impl DispatchKind {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Fuzzy => "fuzzy",
            Self::Temporal => "temporal",
            Self::Causal => "causal",
            Self::Hybrid => "hybrid",
            Self::NeedsDisambiguation(_) => "needs_disambiguation",
        }
    }

    pub fn is_exact_family(&self) -> bool {
        matches!(self, Self::Exact | Self::Temporal | Self::Causal)
    }
}

pub struct QueryRouter;

impl QueryRouter {
    pub fn new() -> Self { Self }

    /// Parse a RecallRequest into a dispatch strategy.
    /// Fully-bound requests return Exact/Temporal/Causal (zero token cost).
    /// Underspecified requests return NeedsDisambiguation with named unbound slots.
    pub fn route(&self, req: &RecallRequest) -> DispatchKind {
        // Causal path: requires both tool and entity.
        if req.causal_tool.is_some() {
            return if req.causal_entity.is_some() {
                DispatchKind::Causal
            } else {
                DispatchKind::NeedsDisambiguation(vec![UnboundSlot {
                    name: "causal_entity",
                    context: format!("causal_tool={:?}", req.causal_tool),
                }])
            };
        }

        // Temporal path: at least one bound on the time axis.
        let has_time = req.time_from_ms.is_some() || req.time_to_ms.is_some();

        // Exact path: triplet subject or predicate present.
        let has_exact = req.subject.is_some() || req.predicate.is_some();
        let has_fuzzy = req.freetext.is_some();

        if has_exact && has_fuzzy { return DispatchKind::Hybrid; }
        if has_exact { return DispatchKind::Exact; }
        if has_time && !has_fuzzy { return DispatchKind::Temporal; }
        if has_time && has_fuzzy { return DispatchKind::Hybrid; }
        if has_fuzzy { return DispatchKind::Fuzzy; }

        // Nothing bound — ambiguous request.
        DispatchKind::NeedsDisambiguation(vec![UnboundSlot {
            name: "query_target",
            context: "no subject, predicate, freetext, causal pair, or time range provided".into(),
        }])
    }
}

impl Default for QueryRouter {
    fn default() -> Self { Self::new() }
}

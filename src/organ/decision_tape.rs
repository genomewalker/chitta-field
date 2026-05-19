/// CEC Phase 10 — DecisionTape: explicit branch-point memory.
///
/// Records what the agent chose AND what it considered and rejected, with
/// the reason. Gives `recall_true_counterfactual` real data instead of
/// CDAWG sibling-edge inference.
use serde::{Serialize, Deserialize};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RejectionReason {
    LowerQValue    = 0,
    HigherFailRate = 1,
    RefutedRule    = 2,
    CostEstimate   = 3,
    OutOfScope     = 4,
    UserOverride   = 5,
}

impl RejectionReason {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::LowerQValue,
            1 => Self::HigherFailRate,
            2 => Self::RefutedRule,
            3 => Self::CostEstimate,
            4 => Self::OutOfScope,
            _ => Self::UserOverride,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::LowerQValue    => "lower_q_value",
            Self::HigherFailRate => "higher_fail_rate",
            Self::RefutedRule    => "refuted_rule",
            Self::CostEstimate   => "cost_estimate",
            Self::OutOfScope     => "out_of_scope",
            Self::UserOverride   => "user_override",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionPoint {
    pub turn_id:          u32,
    pub chosen_sym:       u64,
    /// (alternative_sym, rejection_reason as u8)
    pub rejected:         Vec<(u64, u8)>,
    /// chosen_confidence - best_alternative_confidence
    pub confidence_delta: f32,
    pub ts_ms:            i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DecisionTape {
    pub points: Vec<DecisionPoint>,
}

impl DecisionTape {
    pub fn new() -> Self { Self::default() }

    pub fn log(
        &mut self,
        turn_id:          u32,
        chosen_sym:       u64,
        rejected:         Vec<(u64, u8)>,
        confidence_delta: f32,
        ts_ms:            i64,
    ) {
        self.points.push(DecisionPoint { turn_id, chosen_sym, rejected, confidence_delta, ts_ms });
    }

    /// Return all DecisionPoints where `sym` was the chosen action or a rejected alternative.
    pub fn decisions_for_sym(&self, sym: u64, k: usize) -> Vec<&DecisionPoint> {
        self.points.iter().rev()
            .filter(|dp| dp.chosen_sym == sym || dp.rejected.iter().any(|(s, _)| *s == sym))
            .take(k)
            .collect()
    }

    /// For counterfactual: return points where `sym` was rejected, along with
    /// the chosen alternative and its confidence_delta.
    pub fn rejected_alternatives(&self, sym: u64, k: usize) -> Vec<(&DecisionPoint, u8)> {
        self.points.iter().rev()
            .filter_map(|dp| {
                dp.rejected.iter()
                    .find(|(s, _)| *s == sym)
                    .map(|(_, reason)| (dp, *reason))
            })
            .take(k)
            .collect()
    }
}

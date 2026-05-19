/// CEC Phase 10 — Hypothesis Market.
///
/// Each Sequitur rule is a hypothesis: "sym_a is reliably followed by sym_b".
/// Wilson score confidence intervals + information gain turn the refutation
/// ledger from a passive graveyard into an experiment planner.
///
/// probe_value = H(p_hat) * (1 - |p_hat - 0.5| * 2)  [maximised at p_hat=0.5]
/// Rules with high probe_value are maximally uncertain — prime targets for a
/// deliberate disambiguation observation.
use serde::{Serialize, Deserialize};
use crate::organ::refutation_ledger::RefutationLedger;

/// Wilson score interval (z=1.96, 95% CI).
fn wilson(successes: u32, total: u32) -> (f32, f32) {
    if total == 0 { return (0.0, 1.0); }
    let n = total as f32;
    let p = successes as f32 / n;
    let z  = 1.96_f32;
    let z2 = z * z;
    let center    = (p + z2 / (2.0 * n)) / (1.0 + z2 / n);
    let halfwidth = (z / (1.0 + z2 / n))
        * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
    ((center - halfwidth).clamp(0.0, 1.0), (center + halfwidth).clamp(0.0, 1.0))
}

fn binary_entropy(p: f32) -> f32 {
    if p <= 0.0 || p >= 1.0 { return 0.0; }
    let q = 1.0 - p;
    -(p * p.log2() + q * q.log2())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypothesis {
    pub rule_id:       u32,
    /// p̂ = support / (support + contradict)
    pub p_hat:         f32,
    /// Wilson lower bound (conservative)
    pub wilson_lower:  f32,
    /// Wilson upper bound
    pub wilson_upper:  f32,
    /// Binary entropy H(p_hat)
    pub info_gain:     f32,
    /// Expected bits from one more observation — zero at certainty, max at p_hat=0.5
    pub probe_value:   f32,
    pub last_probe_ts: i64,
}

impl Hypothesis {
    fn compute(rule_id: u32, support: u32, contradict: u32, last_ts: i64) -> Self {
        let total = support + contradict;
        let p_hat = if total == 0 { 0.0 } else { support as f32 / total as f32 };
        let (wilson_lower, wilson_upper) = wilson(support, total);
        let info_gain  = binary_entropy(p_hat);
        let certainty  = (p_hat - 0.5).abs() * 2.0; // 0 at p=0.5, 1 at p=0 or 1
        let probe_value = info_gain * (1.0 - certainty);
        Self { rule_id, p_hat, wilson_lower, wilson_upper, info_gain, probe_value, last_probe_ts: last_ts }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HypothesisMarket {
    pub hypotheses: Vec<Hypothesis>,
}

impl HypothesisMarket {
    pub fn new() -> Self { Self::default() }

    /// Rebuild from the refutation ledger. Called after each consolidation_pass.
    pub fn update_from_ledger(&mut self, ledger: &RefutationLedger) {
        self.hypotheses.clear();
        for entries in ledger.antecedent_index_entries() {
            for e in entries {
                // Only consider rules with at least 3 observations
                if e.support + e.contradict < 3 { continue; }
                self.hypotheses.push(Hypothesis::compute(
                    e.rule_id, e.support, e.contradict, e.last_ts,
                ));
            }
        }
        // Sort by probe_value descending so top_probes is O(1)
        self.hypotheses.sort_by(|a, b| {
            b.probe_value.partial_cmp(&a.probe_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Top-k rules by expected information gain. These are the best candidates
    /// for a deliberate disambiguation observation.
    pub fn top_probes(&self, k: usize) -> &[Hypothesis] {
        &self.hypotheses[..self.hypotheses.len().min(k)]
    }

    pub fn stats_json(&self, k: usize) -> String {
        let probes = self.top_probes(k);
        let rows: Vec<String> = probes.iter().map(|h| {
            format!(
                "{{\"rule_id\":{},\"p_hat\":{:.3},\"wilson_lower\":{:.3},\
                 \"wilson_upper\":{:.3},\"info_gain\":{:.3},\"probe_value\":{:.3}}}",
                h.rule_id, h.p_hat, h.wilson_lower, h.wilson_upper, h.info_gain, h.probe_value
            )
        }).collect();
        format!(
            "{{\"total\":{},\"top_k\":[{}]}}",
            self.hypotheses.len(), rows.join(",")
        )
    }
}

/// CEC Phase 11 — Turīya Monitor (witness meta-cognition).
///
/// A pure read-only organ that watches the other CEC organs and emits a
/// system-health vector after each consolidation_pass(). No writes to any
/// other organ. No actuation. Pure witness.
///
/// "Turīya" (Sanskrit: the fourth) is the witnessing consciousness that
/// observes the three states (waking/dreaming/deep-sleep) without being
/// caught in any of them. Here it watches the manas organs without
/// intervening in them.
use std::collections::VecDeque;
use serde::{Serialize, Deserialize};

use crate::organ::cdawg::CdawgOrgan;
use crate::organ::event_tape::EventTape;
use crate::organ::refutation_ledger::{RefutationLedger, RefutStatus};
use crate::organ::hypothesis_market::HypothesisMarket;
use crate::organ::fep_prior::FepPriorOrgan;

const MAX_SAMPLES: usize = 100;

/// One time-stamped snapshot of CEC organ health.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthSample {
    pub ts_ms:             i64,
    /// Number of CDAWG states (proxy for explored action-space breadth).
    pub cdawg_states:      usize,
    /// Total events ingested into the EventTape.
    pub tape_events:       usize,
    /// Sequitur rules in the refutation ledger (promoted + being tracked).
    pub tracked_rules:     usize,
    /// Rules whose refutation ratio crossed 0.4 (actively falsified).
    pub refuted_rules:     usize,
    /// Hypotheses in the HypothesisMarket with at least 3 observations.
    pub hypotheses:        usize,
    /// Mean Wilson probe_value across all hypotheses (0=certain, 1=max uncertainty).
    pub mean_probe_value:  f32,
    /// Variance of Q-values across top-50 CDAWG states. Low → degenerate.
    pub q_variance:        f32,
    /// CDAWG states added since previous sample (0 = stale exploration).
    pub delta_states:      i64,
    /// Events added since previous sample.
    pub delta_events:      i64,
    /// Rules promoted since previous sample.
    pub delta_rules:       i64,
    /// FEP EWMA of KL(q||prior_z). Rising trend → world-model drifting.
    pub fep_drift:         f32,
    /// FEP EWMA of emission shock events. High → known contexts, novel outcomes.
    pub fep_shock:         f32,
}

/// High-level diagnosis derived from a HealthSample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Diagnosis {
    Healthy,
    /// CDAWG is not growing despite new events → agent repeating the same actions.
    Stale,
    /// Q-value variance collapsed → all paths look equally (un)promising.
    QCollapse,
    /// More rules refuted than live → learned knowledge is being invalidated faster
    /// than it is created.
    RefutationFlood,
    /// Mean probe_value > 0.65 → most hypotheses are maximally uncertain.
    /// Experimentation (Phase 14) should be triggered.
    HighUncertainty,
    /// FEP EWMA drift > 0.5 → generative model predictions are degrading
    /// even though Q-values look stable. World-model going silently stale.
    FepContextDrift,
    /// FEP emission shock EWMA > 0.3 → known contexts producing novel outcomes.
    /// Learned rules are being violated in practice.
    FepEmissionShock,
}

impl HealthSample {
    pub fn diagnose(&self) -> Diagnosis {
        // Priority order: most actionable first.
        if self.refuted_rules > 0 && self.tracked_rules > 0
            && self.refuted_rules * 2 >= self.tracked_rules
        {
            return Diagnosis::RefutationFlood;
        }
        if self.delta_states == 0 && self.delta_events > 20 {
            return Diagnosis::Stale;
        }
        if self.q_variance < 0.005 && self.cdawg_states > 50 {
            return Diagnosis::QCollapse;
        }
        if self.mean_probe_value > 0.65 && self.hypotheses >= 5 {
            return Diagnosis::HighUncertainty;
        }
        if self.fep_drift > 0.5 {
            return Diagnosis::FepContextDrift;
        }
        if self.fep_shock > 0.3 {
            return Diagnosis::FepEmissionShock;
        }
        Diagnosis::Healthy
    }

    pub fn diagnosis_label(&self) -> &'static str {
        match self.diagnose() {
            Diagnosis::Healthy          => "healthy",
            Diagnosis::Stale            => "stale",
            Diagnosis::QCollapse        => "q_collapse",
            Diagnosis::RefutationFlood  => "refutation_flood",
            Diagnosis::HighUncertainty  => "high_uncertainty",
            Diagnosis::FepContextDrift  => "fep_context_drift",
            Diagnosis::FepEmissionShock => "fep_emission_shock",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuriyaMonitor {
    /// Rolling window of health snapshots (newest last).
    pub samples:       VecDeque<HealthSample>,
    pub sample_count:  u64,
}

impl Default for TuriyaMonitor {
    fn default() -> Self {
        Self { samples: VecDeque::new(), sample_count: 0 }
    }
}

impl TuriyaMonitor {
    pub fn new() -> Self { Self::default() }

    /// Take a health sample from the current organ state.
    /// Called at the end of consolidation_pass() — no writes to other organs.
    pub fn sample(
        &mut self,
        ts_ms:   i64,
        cdawg:   &CdawgOrgan,
        tape:    &EventTape,
        ledger:  &RefutationLedger,
        market:  &HypothesisMarket,
        fep:     &FepPriorOrgan,
    ) {
        let cdawg_states = cdawg.state_count();
        let tape_events  = tape.events.len();

        let (tracked_rules, refuted_rules) = {
            let mut total = 0usize;
            let mut refuted = 0usize;
            for entries in ledger.antecedent_index_entries() {
                for e in entries {
                    if e.support + e.contradict >= 3 {
                        total += 1;
                        if let RefutStatus::Refuted(_) = ledger.status(e.rule_id) {
                            refuted += 1;
                        }
                    }
                }
            }
            (total, refuted)
        };

        let (hypotheses, mean_probe_value) = {
            let h = &market.hypotheses;
            let n = h.len();
            let mean = if n == 0 { 0.0 }
                else { h.iter().map(|h| h.probe_value).sum::<f32>() / n as f32 };
            (n, mean)
        };

        // Q-variance from top-50 states via context=[] prefix (all states).
        let q_vals: Vec<f32> = cdawg.top_q_states(&[], 50)
            .into_iter().map(|(_, q, _)| q).collect();
        let q_variance = variance(&q_vals);

        let prev = self.samples.back();
        let delta_states = cdawg_states as i64 - prev.map(|s| s.cdawg_states as i64).unwrap_or(cdawg_states as i64);
        let delta_events = tape_events  as i64 - prev.map(|s| s.tape_events  as i64).unwrap_or(tape_events  as i64);
        let delta_rules  = tracked_rules as i64 - prev.map(|s| s.tracked_rules as i64).unwrap_or(tracked_rules as i64);

        let s = HealthSample {
            ts_ms,
            cdawg_states,
            tape_events,
            tracked_rules,
            refuted_rules,
            hypotheses,
            mean_probe_value,
            q_variance,
            delta_states,
            delta_events,
            delta_rules,
            fep_drift:  fep.ewma_drift,
            fep_shock:  fep.ewma_shock,
        };

        self.samples.push_back(s);
        if self.samples.len() > MAX_SAMPLES {
            self.samples.pop_front();
        }
        self.sample_count += 1;
    }

    /// Latest sample, if any.
    pub fn latest(&self) -> Option<&HealthSample> {
        self.samples.back()
    }

    /// JSON status report for the `turiya_status` tool.
    pub fn status_json(&self) -> String {
        let Some(latest) = self.latest() else {
            return r#"{"status":"no_samples","diagnosis":"unknown","samples":0}"#.to_string();
        };
        let diagnosis = latest.diagnosis_label();

        // Trend: are we growing or plateauing? Compare last sample to N-5 sample.
        let trend = if self.samples.len() >= 5 {
            let old = &self.samples[self.samples.len() - 5];
            let ds = latest.cdawg_states as i64 - old.cdawg_states as i64;
            let de = latest.tape_events  as i64 - old.tape_events  as i64;
            if ds > 0 && de > 0 { "growing" }
            else if de > 0 && ds == 0 { "stale_exploration" }
            else if de == 0 { "idle" }
            else { "unknown" }
        } else {
            "insufficient_history"
        };

        // Recent diagnosis streak.
        let recent_streak: Vec<&str> = self.samples.iter().rev().take(5)
            .map(|s| s.diagnosis_label())
            .collect();

        format!(
            r#"{{"status":"ok","diagnosis":"{diagnosis}","trend":"{trend}","samples":{samples},"latest":{{"ts_ms":{ts},"cdawg_states":{cs},"tape_events":{te},"tracked_rules":{tr},"refuted_rules":{rr},"hypotheses":{hy},"mean_probe_value":{mpv:.4},"q_variance":{qv:.6},"delta_states":{ds},"delta_events":{de},"fep_drift":{fd:.4},"fep_shock":{fs:.4}}},"recent_diagnoses":{rd_json}}}"#,
            samples   = self.sample_count,
            ts        = latest.ts_ms,
            cs        = latest.cdawg_states,
            te        = latest.tape_events,
            tr        = latest.tracked_rules,
            rr        = latest.refuted_rules,
            hy        = latest.hypotheses,
            mpv       = latest.mean_probe_value,
            qv        = latest.q_variance,
            ds        = latest.delta_states,
            de        = latest.delta_events,
            fd        = latest.fep_drift,
            fs        = latest.fep_shock,
            rd_json   = format_str_array(&recent_streak),
        )
    }
}

fn variance(vals: &[f32]) -> f32 {
    if vals.len() < 2 { return 0.0; }
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32
}

fn format_str_array(items: &[&str]) -> String {
    let inner: Vec<String> = items.iter().map(|s| format!("\"{s}\"")).collect();
    format!("[{}]", inner.join(","))
}

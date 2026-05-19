/// CEC Phase 15 — Free Energy Principle priors (FepPriorOrgan).
///
/// A non-actuating generative model P(sym_t | context_t) over the EventTape
/// symbol stream. The CDAWG active state plus its k-deep suffix chain forms
/// the latent context. Factored emission: P(tool | state) · P(outcome | state, tool).
/// Entity is marginalized (too high-cardinality; would swamp the signal).
///
/// Online update: Bayesian Dirichlet-multinomial count update with exponential
/// forgetting (state_decay multiplier applied at each step).
///
/// Free energy per step (variational bound, tractable):
///   F_t = E_q[-log P(sym_t | z)] + KL(q_t(z) || prior_z)
///
/// Two diagnostic signals emitted to TuriyaMonitor:
///   fep_context_drift  — EWMA of KL(q_t || prior_z): world-model going stale
///   fep_emission_shock — high NLL with low KL: known context, novel symbol (failure)
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use crate::organ::cdawg::CdawgOrgan;
use crate::organ::event_tape::TurnEvent;

const K_SUFFIX: usize = 4;
const ALPHA_PRIOR: f32 = 0.1;   // Dirichlet prior pseudocount per symbol
const STATE_DECAY: f32 = 0.9995; // per-event forgetting on Dirichlet counts
const EWMA_ALPHA: f32 = 0.05;   // smoothing factor for drift / shock EWMAs

/// Dirichlet sufficient statistics for a categorical distribution over a sparse symbol set.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct DirichletRow {
    /// Accumulated (possibly fractional) counts per symbol key.
    counts: HashMap<u64, f32>,
    /// Sum of all counts (not including implicit prior).
    total: f32,
}

impl DirichletRow {
    fn observe(&mut self, sym: u64, decay: f32) {
        for v in self.counts.values_mut() { *v *= decay; }
        self.total *= decay;
        *self.counts.entry(sym).or_insert(0.0) += 1.0;
        self.total += 1.0;
    }

    /// P(sym) under Dirichlet-multinomial with `vocab_size` unique symbols seen so far.
    fn prob(&self, sym: u64) -> f32 {
        let vocab = self.counts.len().max(1) as f32;
        let count = self.counts.get(&sym).copied().unwrap_or(0.0);
        (count + ALPHA_PRIOR) / (self.total + ALPHA_PRIOR * vocab)
    }

    fn nll(&self, sym: u64) -> f32 { -self.prob(sym).max(1e-9).ln() }
}

/// Factored emission model for one CDAWG state.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct StateEmission {
    /// P(tool_id | this state).
    tool: DirichletRow,
    /// P(outcome | this state, tool_id).
    outcome_given_tool: HashMap<u16, DirichletRow>,
}

impl StateEmission {
    fn observe(&mut self, tool_id: u16, outcome: u8, decay: f32) {
        self.tool.observe(tool_id as u64, decay);
        self.outcome_given_tool
            .entry(tool_id)
            .or_default()
            .observe(outcome as u64, decay);
    }

    /// -log P(sym) factored as tool + outcome|tool. Entity is marginalized.
    fn nll(&self, tool_id: u16, outcome: u8) -> f32 {
        let nll_tool = self.tool.nll(tool_id as u64);
        let nll_outcome = self.outcome_given_tool
            .get(&tool_id)
            .map(|row| row.nll(outcome as u64))
            .unwrap_or(-ALPHA_PRIOR.ln()); // prior when unseen
        nll_tool + nll_outcome
    }
}

/// Output of one FEP observation step.
#[derive(Debug, Clone)]
pub struct FepStep {
    /// Negative log-likelihood: E_q[-log P(sym | z)].
    pub nll: f32,
    /// KL(q_t(z) || prior_z): context-distribution shift.
    pub kl_z: f32,
    /// Total variational free energy: nll + kl_z.
    pub free_energy: f32,
    /// True when NLL is high but KL is low: known context, novel symbol.
    pub emission_shock: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FepPriorOrgan {
    /// Emission model per CDAWG state id.
    state_emission: HashMap<u32, StateEmission>,
    /// Previous posterior over (state_id, weight) — the recognition model q_{t-1}.
    prev_q: Vec<(u32, f32)>,
    /// EWMA of KL(q || prior_z) — rising trend means world-model drift.
    pub ewma_drift: f32,
    /// EWMA of emission shock events (1.0 = shock, 0.0 = no shock).
    pub ewma_shock: f32,
    /// Total observations processed.
    pub obs_count: u64,
}

impl Default for FepPriorOrgan {
    fn default() -> Self {
        Self {
            state_emission: HashMap::new(),
            prev_q: Vec::new(),
            ewma_drift: 0.0,
            ewma_shock: 0.0,
            obs_count: 0,
        }
    }
}

impl FepPriorOrgan {
    pub fn new() -> Self { Self::default() }

    /// Observe a new EventTape event and return the FEP step metrics.
    /// Called from consolidation_pass() after CDAWG extend() for the same event.
    pub fn observe(&mut self, ev: &TurnEvent, cdawg: &CdawgOrgan) -> FepStep {
        let (tool_id, outcome) = unpack_tool_outcome(ev.pack());

        // Build candidate state set: active state + up-to-K suffix chain.
        let candidates = self.suffix_candidates(cdawg);

        // Prior over candidates: uniform over the candidate set.
        let n = candidates.len().max(1) as f32;
        let prior_z: Vec<(u32, f32)> = candidates.iter().map(|&s| (s, 1.0 / n)).collect();

        // Emission NLL per candidate.
        let nll_per_state: Vec<f32> = candidates.iter().map(|&sid| {
            self.state_emission
                .get(&sid)
                .map(|e| e.nll(tool_id, outcome))
                .unwrap_or(-ALPHA_PRIOR.ln() * 2.0)
        }).collect();

        // Posterior q_t ∝ prior_z * exp(-nll).
        let log_weights: Vec<f32> = prior_z.iter().zip(nll_per_state.iter())
            .map(|((_, pw), &nll)| pw.ln() - nll)
            .collect();
        let log_max = log_weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let unnorm: Vec<f32> = log_weights.iter().map(|&lw| (lw - log_max).exp()).collect();
        let z_sum: f32 = unnorm.iter().sum::<f32>().max(1e-30);
        let q_t: Vec<(u32, f32)> = candidates.iter().zip(unnorm.iter())
            .map(|(&s, &u)| (s, u / z_sum))
            .collect();

        // Expected NLL = E_q[-log P(sym | z)].
        let nll = q_t.iter().zip(nll_per_state.iter())
            .map(|((_, qw), &nll)| qw * nll)
            .sum::<f32>();

        // KL(q_t || prior_z).
        let kl_z = kl_discrete(&q_t, &prior_z);

        let free_energy = nll + kl_z;
        let emission_shock = nll > 2.5 && kl_z < 0.3;

        // Update EWMAs.
        self.ewma_drift = EWMA_ALPHA * kl_z + (1.0 - EWMA_ALPHA) * self.ewma_drift;
        self.ewma_shock = EWMA_ALPHA * (if emission_shock { 1.0 } else { 0.0 })
            + (1.0 - EWMA_ALPHA) * self.ewma_shock;

        // Update Dirichlet counts for all states in q_t (weighted by posterior).
        for &(sid, qw) in &q_t {
            if qw < 1e-4 { continue; }
            self.state_emission
                .entry(sid)
                .or_default()
                .observe(tool_id, outcome, STATE_DECAY.powf(qw));
        }

        self.prev_q = q_t;
        self.obs_count += 1;

        FepStep { nll, kl_z, free_energy, emission_shock }
    }

    /// Compute FEP free energy for a hypothetical symbol (read-only, no state mutation).
    pub fn predict_free_energy(&self, ev: &TurnEvent, cdawg: &CdawgOrgan) -> FepStep {
        let (tool_id, outcome) = unpack_tool_outcome(ev.pack());
        let candidates = self.suffix_candidates_ro(cdawg);
        let n = candidates.len().max(1) as f32;

        let nll_per_state: Vec<f32> = candidates.iter().map(|&sid| {
            self.state_emission
                .get(&sid)
                .map(|e| e.nll(tool_id, outcome))
                .unwrap_or(-ALPHA_PRIOR.ln() * 2.0)
        }).collect();

        let prior_z: Vec<(u32, f32)> = candidates.iter().map(|&s| (s, 1.0 / n)).collect();
        let log_weights: Vec<f32> = prior_z.iter().zip(nll_per_state.iter())
            .map(|((_, pw), &nll)| pw.ln() - nll)
            .collect();
        let log_max = log_weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let unnorm: Vec<f32> = log_weights.iter().map(|&lw| (lw - log_max).exp()).collect();
        let z_sum = unnorm.iter().sum::<f32>().max(1e-30);
        let q_t: Vec<(u32, f32)> = candidates.iter().zip(unnorm.iter())
            .map(|(&s, &u)| (s, u / z_sum))
            .collect();

        let nll = q_t.iter().zip(nll_per_state.iter())
            .map(|((_, qw), &nll)| qw * nll)
            .sum::<f32>();
        let kl_z = kl_discrete(&q_t, &prior_z);
        let emission_shock = nll > 2.5 && kl_z < 0.3;

        FepStep { nll, kl_z, free_energy: nll + kl_z, emission_shock }
    }

    /// Observe from a packed u64 symbol (hot path — no TurnEvent allocation needed).
    pub fn observe_packed(&mut self, sym: u64, cdawg: &CdawgOrgan) -> FepStep {
        let (tool_id, outcome) = unpack_tool_outcome(sym);
        let candidates = suffix_chain(cdawg, K_SUFFIX);
        let n = candidates.len().max(1) as f32;
        let prior_z: Vec<(u32, f32)> = candidates.iter().map(|&s| (s, 1.0 / n)).collect();
        let nll_per_state: Vec<f32> = candidates.iter().map(|&sid| {
            self.state_emission.get(&sid)
                .map(|e| e.nll(tool_id, outcome))
                .unwrap_or(-ALPHA_PRIOR.ln() * 2.0)
        }).collect();
        let log_weights: Vec<f32> = prior_z.iter().zip(nll_per_state.iter())
            .map(|((_, pw), &nll)| pw.ln() - nll).collect();
        let log_max = log_weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let unnorm: Vec<f32> = log_weights.iter().map(|&lw| (lw - log_max).exp()).collect();
        let z_sum = unnorm.iter().sum::<f32>().max(1e-30);
        let q_t: Vec<(u32, f32)> = candidates.iter().zip(unnorm.iter())
            .map(|(&s, &u)| (s, u / z_sum)).collect();
        let nll = q_t.iter().zip(nll_per_state.iter())
            .map(|((_, qw), &nll)| qw * nll).sum::<f32>();
        let kl_z = kl_discrete(&q_t, &prior_z);
        let emission_shock = nll > 2.5 && kl_z < 0.3;
        self.ewma_drift = EWMA_ALPHA * kl_z + (1.0 - EWMA_ALPHA) * self.ewma_drift;
        self.ewma_shock = EWMA_ALPHA * (if emission_shock { 1.0 } else { 0.0 })
            + (1.0 - EWMA_ALPHA) * self.ewma_shock;
        for &(sid, qw) in &q_t {
            if qw < 1e-4 { continue; }
            self.state_emission.entry(sid).or_default()
                .observe(tool_id, outcome, STATE_DECAY.powf(qw));
        }
        self.prev_q = q_t;
        self.obs_count += 1;
        FepStep { nll, kl_z, free_energy: nll + kl_z, emission_shock }
    }

    /// Rebuild the entire FEP model from an EventTape, running alongside a fresh CDAWG.
    /// Returns the final CDAWG (which callers can discard — it's just a replay vehicle).
    pub fn rebuild_from_tape(&mut self, tape: &super::event_tape::EventTape) {
        use super::cdawg::CdawgOrgan;
        self.reset();
        let mut replay = CdawgOrgan::new();
        for ev in &tape.events {
            let sym = ev.pack();
            replay.extend(sym, ev.turn_id);
            self.observe_packed(sym, &replay);
        }
    }

    /// Reset model — called when EventTape is rebuilt from scratch.
    pub fn reset(&mut self) {
        self.state_emission.clear();
        self.prev_q.clear();
        self.ewma_drift = 0.0;
        self.ewma_shock = 0.0;
        self.obs_count = 0;
    }

    pub fn state_emission_len(&self) -> usize { self.state_emission.len() }

    /// JSON status for fep_status tool.
    pub fn status_json(&self) -> String {
        format!(
            r#"{{"obs_count":{obs},"ewma_drift":{drift:.4},"ewma_shock":{shock:.4},"states_modeled":{states},"context_drift":{cd},"emission_shock":{es}}}"#,
            obs    = self.obs_count,
            drift  = self.ewma_drift,
            shock  = self.ewma_shock,
            states = self.state_emission.len(),
            cd     = self.ewma_drift > 0.5,
            es     = self.ewma_shock > 0.3,
        )
    }

    fn suffix_candidates(&self, cdawg: &CdawgOrgan) -> Vec<u32> {
        suffix_chain(cdawg, K_SUFFIX)
    }

    fn suffix_candidates_ro(&self, cdawg: &CdawgOrgan) -> Vec<u32> {
        suffix_chain(cdawg, K_SUFFIX)
    }
}

fn suffix_chain(cdawg: &CdawgOrgan, k: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(k + 1);
    let mut s = cdawg.last;
    for _ in 0..=k {
        out.push(s);
        let link = cdawg.states[s as usize].link;
        if link == crate::organ::cdawg::NULL_STATE || link == s { break; }
        s = link;
    }
    out
}

/// Extract (tool_id: u16, outcome: u8) from a packed EventTape u64 symbol.
/// Packing: bits 63-48 = tool_id, bits 47-40 = outcome, bits 39-8 = entity_key, bits 7-0 = 0.
fn unpack_tool_outcome(sym: u64) -> (u16, u8) {
    let tool_id = (sym >> 48) as u16;
    let outcome = ((sym >> 40) & 0xFF) as u8;
    (tool_id, outcome)
}

/// KL divergence KL(q || p) for discrete distributions given as (id, weight) pairs.
fn kl_discrete(q: &[(u32, f32)], p: &[(u32, f32)]) -> f32 {
    let p_map: HashMap<u32, f32> = p.iter().cloned().collect();
    q.iter().map(|&(id, qw)| {
        if qw < 1e-9 { return 0.0; }
        let pw = p_map.get(&id).copied().unwrap_or(1e-9);
        qw * (qw / pw).ln()
    }).sum()
}

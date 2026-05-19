/// Causal Refutation Ledger — tracks which Sequitur rules are being falsified.
///
/// Each promoted rule has an antecedent sym_a and expected consequent sym_b.
/// Every time sym_a appears followed by something OTHER than sym_b, that's a
/// contradiction. Hysteresis thresholds (0.4 refute, 0.2 reinstate) govern status.
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::organ::sequitur::SequiturRule;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefutEntry {
    pub rule_id:        u32,
    pub expected_sym_b: u64,
    pub support:        u32,
    pub contradict:     u32,
    pub last_ts:        i64,
}

impl RefutEntry {
    pub fn refute_ratio(&self) -> f32 {
        let denom = self.support + self.contradict;
        if denom == 0 { 0.0 } else { self.contradict as f32 / denom as f32 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RefutStatus {
    Live,
    Refuted(i64),  // ts_ms of refutation
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefutationLedger {
    // sym_a → entries for rules whose antecedent is sym_a
    antecedent_index: HashMap<u64, Vec<RefutEntry>>,
    pub rule_status:  HashMap<u32, RefutStatus>,
}

impl Default for RefutationLedger {
    fn default() -> Self {
        Self { antecedent_index: HashMap::new(), rule_status: HashMap::new() }
    }
}

impl RefutationLedger {
    pub fn new() -> Self { Self::default() }

    pub fn seed_from_rules(&mut self, rules: &[SequiturRule]) {
        self.antecedent_index.clear();
        // Preserve existing status entries — only add new ones
        for r in rules {
            self.rule_status.entry(r.id).or_insert(RefutStatus::Live);
            self.antecedent_index
                .entry(r.sym_a)
                .or_default()
                .push(RefutEntry {
                    rule_id:        r.id,
                    expected_sym_b: r.sym_b,
                    support:        r.support,
                    contradict:     0,
                    last_ts:        0,
                });
        }
    }

    /// Called after every EventTape append (sym_a → sym_b observation).
    /// Returns rule_ids whose status changed (Live↔Refuted).
    pub fn observe(&mut self, sym_a: u64, sym_b: u64, ts_ms: i64) -> Vec<(u32, RefutStatus)> {
        let entries = match self.antecedent_index.get_mut(&sym_a) {
            Some(e) => e,
            None => return Vec::new(),
        };

        let mut changed = Vec::new();
        for entry in entries.iter_mut() {
            entry.last_ts = ts_ms;
            if sym_b == entry.expected_sym_b {
                entry.support += 1;
            } else {
                entry.contradict += 1;
            }
            let ratio = entry.refute_ratio();
            let id = entry.rule_id;
            let current = self.rule_status.get(&id).cloned().unwrap_or(RefutStatus::Live);
            let next = match &current {
                RefutStatus::Live if ratio > 0.4 => Some(RefutStatus::Refuted(ts_ms)),
                RefutStatus::Refuted(_) if ratio < 0.2 => Some(RefutStatus::Live),
                _ => None,
            };
            if let Some(s) = next {
                self.rule_status.insert(id, s.clone());
                changed.push((id, s));
            }
        }
        changed
    }

    /// Iterate all antecedent entry groups (for HypothesisMarket).
    pub fn antecedent_index_entries(&self) -> impl Iterator<Item = &Vec<RefutEntry>> {
        self.antecedent_index.values()
    }

    pub fn refute_ratio_for_rule(&self, rule_id: u32) -> f32 {
        self.antecedent_index.values().flatten()
            .find(|e| e.rule_id == rule_id)
            .map(|e| e.refute_ratio())
            .unwrap_or(0.0)
    }

    pub fn status(&self, rule_id: u32) -> RefutStatus {
        self.rule_status.get(&rule_id).cloned().unwrap_or(RefutStatus::Live)
    }

    /// Top-k rules by refute_ratio, for diagnostic output.
    pub fn top_refuted(&self, k: usize) -> Vec<(u32, f32, RefutStatus)> {
        let mut rows: Vec<(u32, f32, RefutStatus)> = self.antecedent_index
            .values()
            .flatten()
            .filter_map(|e| {
                let ratio = e.refute_ratio();
                let status = self.status(e.rule_id);
                if e.support + e.contradict >= 3 { Some((e.rule_id, ratio, status)) } else { None }
            })
            .collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rows.truncate(k);
        rows
    }

    /// Stats summary for the `refutation_stats` tool.
    pub fn stats_json(&self, tape: &crate::organ::event_tape::EventTape, k: usize) -> String {
        let top = self.top_refuted(k);
        let live_count = self.rule_status.values().filter(|s| **s == RefutStatus::Live).count();
        let refuted_count = self.rule_status.len() - live_count;
        let rows: Vec<String> = top.iter().map(|(id, ratio, status)| {
            // Find the entry to get sym_a
            let entry_opt = self.antecedent_index.values().flatten().find(|e| e.rule_id == *id);
            let sym_repr = entry_opt.map(|e| {
                let ta = tape.tool_name((e.expected_sym_b >> 40) as u16);
                let ea = tape.entity_name((e.expected_sym_b & 0xffff_ffff) as u32);
                format!("{ta}({ea})")
            }).unwrap_or_else(|| format!("rule:{id}"));
            let st = match status { RefutStatus::Live => "live", RefutStatus::Refuted(_) => "refuted" };
            format!("  rule_{id}: {sym_repr} ratio={ratio:.2} status={st}")
        }).collect();
        format!(
            "refutation_stats: live={live_count} refuted={refuted_count} top_k=[\n{}\n]",
            rows.join("\n")
        )
    }
}

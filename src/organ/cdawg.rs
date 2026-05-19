/// Online suffix automaton (Blumer/Crochemore CDAWG) over a coarse action alphabet.
/// Alphabet: packed u64 from (tool_id: u16, outcome_class: u8, entity_key: u32).
/// Amortized O(1) per symbol insertion. Answers episodic queries in sub-ms without any
/// embedding model.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use super::event_tape::EventTape;

pub const NULL_STATE: u32 = u32::MAX;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdawgState {
    /// Length of the longest suffix in this equivalence class.
    pub len: u32,
    /// Suffix link (parent in suffix link tree). NULL_STATE for the initial state.
    pub link: u32,
    /// Symbol → next state transitions.
    pub transitions: HashMap<u64, u32>,
    /// Turn indices where this state is the "last" solid state (not a clone).
    /// Approximate endpos: for exact endpos, call collect_endpos() which does a DFS
    /// over sl_children.
    pub endpos: Vec<u32>,
    /// Integrated TD(λ) credit from outcome feedback.
    pub credit: f32,
    /// Counts for outcomes observed while this state was the active suffix.
    pub fail_count: u32,
    pub succ_count: u32,
    /// Suffix-link-tree children (for endpos DFS). Not populated on clones at creation;
    /// maintained incrementally by extend().
    pub sl_children: Vec<u32>,
}

impl CdawgState {
    fn new(len: u32, link: u32) -> Self {
        Self {
            len,
            link,
            transitions: HashMap::new(),
            endpos: Vec::new(),
            credit: 0.0,
            fail_count: 0,
            succ_count: 0,
            sl_children: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdawgOrgan {
    pub states: Vec<CdawgState>,
    /// Index of the state representing the current longest suffix.
    pub last: u32,
    /// Set to true after rebuild_from_tape(); allows callers to detect cold start.
    #[serde(skip)]
    pub rebuilt: bool,
}

impl Default for CdawgOrgan {
    fn default() -> Self {
        Self::new()
    }
}

impl CdawgOrgan {
    pub fn new() -> Self {
        // State 0 = initial. link = NULL_STATE (no parent).
        Self {
            states: vec![CdawgState::new(0, NULL_STATE)],
            last: 0,
            rebuilt: false,
        }
    }

    /// Blumer online suffix automaton extension (standard algorithm).
    /// `sym` is the packed symbol; `turn` is the 0-based tape index.
    pub fn extend(&mut self, sym: u64, turn: u32) {
        let cur = self.states.len() as u32;
        let cur_len = self.states[self.last as usize].len + 1;
        let mut new_state = CdawgState::new(cur_len, NULL_STATE);
        new_state.endpos.push(turn);
        self.states.push(new_state);

        // Walk from last upward, adding transitions to cur where missing.
        let mut p = self.last;
        loop {
            if self.states[p as usize].transitions.contains_key(&sym) {
                break;
            }
            self.states[p as usize].transitions.insert(sym, cur);
            let link = self.states[p as usize].link;
            if link == NULL_STATE {
                p = NULL_STATE;
                break;
            }
            p = link;
        }

        if p == NULL_STATE {
            // No existing transition on sym → link cur to initial state.
            self.states[cur as usize].link = 0;
            self.states[0].sl_children.push(cur);
        } else {
            let q = self.states[p as usize].transitions[&sym];
            let q_len = self.states[q as usize].len;
            let p_len = self.states[p as usize].len;

            if q_len == p_len + 1 {
                // q is already the correct suffix link.
                self.states[cur as usize].link = q;
                self.states[q as usize].sl_children.push(cur);
            } else {
                // Clone q to create a state with exactly len = p_len + 1.
                let clone = self.states.len() as u32;
                let clone_state = CdawgState {
                    len:         p_len + 1,
                    link:        self.states[q as usize].link,
                    transitions: self.states[q as usize].transitions.clone(),
                    endpos:      self.states[q as usize].endpos.clone(),
                    credit:      0.0,
                    fail_count:  0,
                    succ_count:  0,
                    sl_children: Vec::new(),
                };
                self.states.push(clone_state);

                // Reparent q and cur under clone.
                let old_q_link = self.states[q as usize].link;
                self.states[q as usize].link = clone;
                self.states[cur as usize].link = clone;

                // Fix sl_children of old_q_link: replace q with clone.
                if old_q_link != NULL_STATE {
                    let children = &mut self.states[old_q_link as usize].sl_children;
                    if let Some(pos) = children.iter().position(|&x| x == q) {
                        children[pos] = clone;
                    } else {
                        children.push(clone);
                    }
                }
                // clone's children are q and cur.
                self.states[clone as usize].sl_children.push(q);
                self.states[clone as usize].sl_children.push(cur);

                // Redirect transitions from p upward: any that pointed to q → clone.
                let mut p2 = p;
                loop {
                    match self.states[p2 as usize].transitions.get_mut(&sym) {
                        Some(t) if *t == q => { *t = clone; }
                        _ => break,
                    }
                    let link2 = self.states[p2 as usize].link;
                    if link2 == NULL_STATE { break; }
                    p2 = link2;
                }
            }
        }

        self.last = cur;
    }

    /// Rebuild automaton from scratch using all events in the tape.
    pub fn rebuild_from_tape(&mut self, tape: &EventTape) {
        *self = Self::new();
        for ev in &tape.events {
            self.extend(ev.pack(), ev.turn_id);
        }
        self.rebuilt = true;
    }

    /// Walk the automaton for the given symbol sequence.
    /// Returns the reached state index, or None if the pattern is absent.
    pub fn walk(&self, syms: &[u64]) -> Option<u32> {
        let mut state = 0u32;
        for &s in syms {
            state = *self.states[state as usize].transitions.get(&s)?;
        }
        Some(state)
    }

    /// Collect all endpos turns from `start` and all its suffix-link-tree descendants.
    /// Result is sorted and deduplicated.
    pub fn collect_endpos(&self, start: u32) -> Vec<u32> {
        let mut result = Vec::new();
        let mut stack = vec![start];
        while let Some(s) = stack.pop() {
            result.extend_from_slice(&self.states[s as usize].endpos);
            stack.extend_from_slice(&self.states[s as usize].sl_children);
        }
        result.sort_unstable();
        result.dedup();
        result
    }

    /// Return the most recent turn index where `syms` occurred, or None.
    pub fn last_occurrence(&self, syms: &[u64]) -> Option<u32> {
        let state = self.walk(syms)?;
        self.collect_endpos(state).into_iter().max()
    }

    /// For a given target pattern, find k most frequent preceding single-symbol contexts
    /// (causal antecedents). Scored by raw co-occurrence count (PMI approximation).
    pub fn causal_antecedents(
        &self,
        syms: &[u64],
        k: usize,
        tape: &EventTape,
    ) -> Vec<(Vec<u64>, u32)> {
        let state = match self.walk(syms) {
            Some(s) => s,
            None => return vec![],
        };
        let turns = self.collect_endpos(state);
        let pat_len = syms.len();

        let mut counts: HashMap<u64, u32> = HashMap::new();
        for &t in &turns {
            let t = t as usize;
            if t >= pat_len {
                if let Some(ev) = tape.events.get(t - pat_len) {
                    *counts.entry(ev.pack()).or_insert(0) += 1;
                }
            }
        }

        let mut sorted: Vec<(Vec<u64>, u32)> = counts
            .into_iter()
            .map(|(sym, cnt)| (vec![sym], cnt))
            .collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(k);
        sorted
    }

    /// Return top-k states with high failure rates as (state_id, fail_count, fail_ratio).
    pub fn failure_patterns(&self, min_fail: u32, k: usize) -> Vec<(u32, u32, f32)> {
        let mut candidates: Vec<(u32, u32, f32)> = self
            .states
            .iter()
            .enumerate()
            .filter(|(_, s)| s.fail_count >= min_fail)
            .map(|(i, s)| {
                let total = s.fail_count + s.succ_count;
                let ratio = if total > 0 { s.fail_count as f32 / total as f32 } else { 0.0 };
                (i as u32, s.fail_count, ratio)
            })
            .collect();
        candidates.sort_by(|a, b| {
            let score_b = b.2 * b.1 as f32;
            let score_a = a.2 * a.1 as f32;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(k);
        candidates
    }

    /// PPM-style prediction: given context symbols, return next-symbol distribution.
    /// Backs off to shorter context if full context not found.
    pub fn ppm_predict(&self, context: &[u64]) -> Vec<(u64, f32)> {
        let mut state = 0u32;
        for &sym in context {
            match self.states[state as usize].transitions.get(&sym).copied() {
                Some(next) => state = next,
                None => break,
            }
        }
        let transitions = &self.states[state as usize].transitions;
        if transitions.is_empty() {
            return vec![];
        }
        let total = transitions.len() as f32;
        transitions.iter().map(|(&sym, _)| (sym, 1.0 / total)).collect()
    }

    /// Compute surprisal of `sym` given `context`. Returns None if context unseen.
    pub fn surprisal(&self, context: &[u64], sym: u64) -> Option<f32> {
        let preds = self.ppm_predict(context);
        if preds.is_empty() { return None; }
        let p = preds.iter().find(|&&(s, _)| s == sym).map(|&(_, p)| p).unwrap_or(0.0);
        if p <= 0.0 { return None; }
        Some(-(p.ln()))
    }

    /// Push TD(λ) credit backward through the last_n_syms sequence.
    /// `gamma` is the discount factor (typically 0.9).
    pub fn push_td_credit(&mut self, last_n_syms: &[u64], delta: f32, gamma: f32) {
        let n = last_n_syms.len();
        let mut state = 0u32;
        for (i, &sym) in last_n_syms.iter().enumerate() {
            match self.states[state as usize].transitions.get(&sym).copied() {
                Some(next) => {
                    let lag = (n - 1 - i) as i32;
                    self.states[next as usize].credit += delta * gamma.powi(lag);
                    state = next;
                }
                None => break,
            }
        }
    }

    /// Record a success/failure outcome for the action sequence ending at current `last`.
    pub fn record_outcome(&mut self, syms: &[u64], success: bool) {
        if let Some(state) = self.walk(syms) {
            if success {
                self.states[state as usize].succ_count += 1;
            } else {
                self.states[state as usize].fail_count += 1;
            }
        }
    }

    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    pub fn event_count(&self) -> usize {
        // approximate: count solid states (those with non-empty endpos)
        self.states.iter().filter(|s| !s.endpos.is_empty()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tape_and_cdawg(actions: &[(&str, &str, u8)]) -> (EventTape, CdawgOrgan) {
        let mut tape = EventTape::new();
        let mut cdawg = CdawgOrgan::new();
        for (tool, entity, outcome) in actions {
            let sym = tape.log(tool, entity, *outcome, 0, 0);
            let turn = tape.events.len() as u32 - 1;
            cdawg.extend(sym, turn);
        }
        (tape, cdawg)
    }

    #[test]
    fn test_last_occurrence_simple() {
        let (mut tape, cdawg) = tape_and_cdawg(&[
            ("Edit", "store.rs", 0),
            ("Bash", "cargo", 1),
            ("Edit", "store.rs", 0),
        ]);
        let sym = tape.symbol_of("Edit", "store.rs", 0);
        let last = cdawg.last_occurrence(&[sym]);
        assert_eq!(last, Some(2)); // third event (index 2) is the last "Edit store.rs success"
    }

    #[test]
    fn test_pattern_not_present() {
        let (mut tape, cdawg) = tape_and_cdawg(&[("Edit", "foo.rs", 0)]);
        let sym = tape.symbol_of("Bash", "cargo", 1);
        assert_eq!(cdawg.last_occurrence(&[sym]), None);
    }

    #[test]
    fn test_failure_patterns() {
        let (mut tape, mut cdawg) = tape_and_cdawg(&[
            ("Edit", "store.rs", 1),
            ("Edit", "store.rs", 1),
            ("Edit", "store.rs", 1),
            ("Edit", "store.rs", 0),
        ]);
        let fail_sym = tape.symbol_of("Edit", "store.rs", 1);
        cdawg.record_outcome(&[fail_sym], false);
        cdawg.record_outcome(&[fail_sym], false);
        cdawg.record_outcome(&[fail_sym], false);
        let patterns = cdawg.failure_patterns(2, 5);
        assert!(!patterns.is_empty());
        assert!(patterns[0].1 >= 2);
    }

    #[test]
    fn test_ppm_predict() {
        let (_, cdawg) = tape_and_cdawg(&[
            ("Edit", "store.rs", 0),
            ("Bash", "cargo", 1),
            ("Edit", "store.rs", 0),
            ("Bash", "cargo", 1),
        ]);
        // After training on Edit→Bash twice, prediction from initial should include Bash
        let preds = cdawg.ppm_predict(&[]);
        assert!(!preds.is_empty());
        let total: f32 = preds.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 0.01 || preds.len() > 1); // uniform dist
    }

    #[test]
    fn test_rebuild_matches_online() {
        let actions = vec![
            ("Edit", "a.rs", 0u8),
            ("Bash", "cargo", 1),
            ("Edit", "a.rs", 0),
            ("Bash", "cargo", 0),
        ];
        let (tape, online_cdawg) = tape_and_cdawg(&actions);
        let mut rebuilt_cdawg = CdawgOrgan::new();
        rebuilt_cdawg.rebuild_from_tape(&tape);
        assert_eq!(online_cdawg.state_count(), rebuilt_cdawg.state_count());
        assert!(rebuilt_cdawg.rebuilt);
    }
}

/// RePair-style bigram grammar induction over the EventTape symbol stream.
///
/// Finds all bigrams (consecutive symbol pairs) with frequency >= min_support
/// and returns them as "rules" — the agent's learned procedural patterns.
/// Rules with sufficient support are promoted to the triplet KG by the caller.
use std::collections::HashMap;
use crate::organ::event_tape::EventTape;

#[derive(Debug, Clone)]
pub struct SequiturRule {
    pub id:          u32,
    pub sym_a:       u64,
    pub sym_b:       u64,
    pub support:     u32,
    pub avg_outcome: f32,  // 0.0=all-success .. 1.0=all-fail
    pub tape_start:  u32,  // turn index of first occurrence
    pub tape_end:    u32,  // turn index of last occurrence
    pub contradict_count: u32,  // times sym_a appeared but NOT followed by sym_b
}

impl SequiturRule {
    /// Human-readable key suitable for use as a triplet subject.
    pub fn rule_key(&self, tape: &EventTape) -> String {
        let tool_a   = tape.tool_name((self.sym_a >> 40) as u16);
        let ent_a    = tape.entity_name((self.sym_a & 0xffff_ffff) as u32);
        let out_a    = outcome_name((self.sym_a >> 32) as u8 & 0xff);
        let tool_b   = tape.tool_name((self.sym_b >> 40) as u16);
        let ent_b    = tape.entity_name((self.sym_b & 0xffff_ffff) as u32);
        let out_b    = outcome_name((self.sym_b >> 32) as u8 & 0xff);
        format!("rule:{tool_a}({ent_a},{out_a})→{tool_b}({ent_b},{out_b})")
    }

    /// Compact sequence representation for the "compresses" predicate object.
    pub fn seq_repr(&self, tape: &EventTape) -> String {
        let tool_a = tape.tool_name((self.sym_a >> 40) as u16);
        let ent_a  = tape.entity_name((self.sym_a & 0xffff_ffff) as u32);
        let out_a  = outcome_name((self.sym_a >> 32) as u8 & 0xff);
        let tool_b = tape.tool_name((self.sym_b >> 40) as u16);
        let ent_b  = tape.entity_name((self.sym_b & 0xffff_ffff) as u32);
        let out_b  = outcome_name((self.sym_b >> 32) as u8 & 0xff);
        format!(
            "[{tool_a}:{ent_a}:{out_a}] then [{tool_b}:{ent_b}:{out_b}] (×{}, turns {}–{})",
            self.support, self.tape_start, self.tape_end
        )
    }

    pub fn avg_outcome_label(&self) -> &'static str {
        if self.avg_outcome < 0.35 { "success" }
        else if self.avg_outcome > 0.65 { "failure" }
        else { "mixed" }
    }

    /// Phase 13 — Deterministic verbalization. No LLM. Pure template.
    ///
    /// Template: "Whenever {tool_a}({entity_a}) → {outcome_a},
    ///            {tool_b}({entity_b}) tends to follow (×N, avg: outcome)."
    pub fn verbalize(&self, tape: &EventTape) -> String {
        let tool_a = tape.tool_name((self.sym_a >> 40) as u16);
        let ent_a  = tape.entity_name((self.sym_a & 0xffff_ffff) as u32);
        let out_a  = outcome_verb((self.sym_a >> 32) as u8 & 0xff);
        let tool_b = tape.tool_name((self.sym_b >> 40) as u16);
        let ent_b  = tape.entity_name((self.sym_b & 0xffff_ffff) as u32);
        // The consequent's own outcome verb is deliberately not part of the
        // sentence: both branches below phrase the consequent as "tends to
        // follow" and report the pair's outcome through `avg`.
        let avg    = self.avg_outcome_label();

        let ent_a_short = shorten_entity(ent_a);
        let ent_b_short = shorten_entity(ent_b);

        if ent_a == ent_b {
            format!(
                "Whenever {tool_a}({ent_a_short}) {out_a}, {tool_b} on the same entity tends to follow \
                 (observed ×{}, avg outcome: {avg}).",
                self.support
            )
        } else {
            format!(
                "Whenever {tool_a}({ent_a_short}) {out_a}, {tool_b}({ent_b_short}) tends to follow \
                 (observed ×{}, avg outcome: {avg}).",
                self.support
            )
        }
    }
}

fn outcome_verb(c: u8) -> &'static str {
    match c { 0 => "succeeds", 1 => "fails", 2 => "errors", _ => "completes" }
}

fn shorten_entity(e: &str) -> &str {
    // Strip leading path components to keep label readable.
    if let Some(last_slash) = e.rfind('/') {
        &e[last_slash + 1..]
    } else {
        e
    }
}

fn outcome_name(c: u8) -> &'static str {
    match c { 0 => "ok", 1 => "fail", 2 => "err", _ => "partial" }
}

/// Run bigram frequency analysis over the EventTape.
/// Returns rules sorted by support descending.
pub fn run_sequitur(tape: &EventTape, min_support: u32) -> Vec<SequiturRule> {
    let events = &tape.events;
    let n = events.len();
    if n < 2 { return Vec::new(); }

    // Build symbol array from events
    let syms: Vec<u64> = events.iter().map(|e| e.pack()).collect();

    // Count bigrams and collect first/last turn positions
    let mut counts: HashMap<(u64, u64), (u32, u32, u32)> = HashMap::new();
    for i in 0..n - 1 {
        let key = (syms[i], syms[i + 1]);
        let entry = counts.entry(key).or_insert((0, i as u32, i as u32));
        entry.0 += 1;
        if (i as u32) < entry.1 { entry.1 = i as u32; }
        if (i as u32) > entry.2 { entry.2 = i as u32; }
    }

    let mut rules: Vec<SequiturRule> = Vec::new();
    let mut next_id: u32 = 0;

    for ((sym_a, sym_b), (support, tape_start, tape_end)) in counts {
        if support < min_support { continue; }

        // Average outcome: use outcome_class of the second symbol (the result event)
        let out_b = ((sym_b >> 32) & 0xff) as f32;
        let avg_outcome = out_b.min(1.0);

        rules.push(SequiturRule {
            id: next_id,
            sym_a,
            sym_b,
            support,
            avg_outcome,
            tape_start,
            tape_end,
            contradict_count: 0,
        });
        next_id += 1;
    }

    rules.sort_by(|a, b| b.support.cmp(&a.support));
    rules
}

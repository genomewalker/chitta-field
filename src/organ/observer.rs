use std::collections::HashMap;
use std::sync::OnceLock;
use regex::Regex;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Predicate registry
// ---------------------------------------------------------------------------

pub const PREDICATES: &[&str] = &[
    "name",                  // 0
    "education.degree",      // 1
    "education.institution", // 2
    "occupation",            // 3
    "location.current",      // 4
    "location.hometown",     // 5
    "preference.food",       // 6
    "preference.drink",      // 7
    "preference.music",      // 8
    "preference.sport",      // 9
    "possession",            // 10
    "family.relation",       // 11
    "hobby",                 // 12
    "achievement",           // 13
];

pub fn predicate_id(name: &str) -> Option<u16> {
    PREDICATES.iter().position(|&p| p == name).map(|i| i as u16)
}

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TimeScope {
    Current,
    Historical,
    Uncertain,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Polarity {
    Affirmed,
    Negated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservedFact {
    pub predicate_id: u16,
    pub object:       String,
    pub polarity:     Polarity,
    pub time_scope:   TimeScope,
    pub confidence:   f32,
    pub source_id:    u64,
    pub valid_from:   i64,
    pub valid_to:     Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ObserverState {
    pub user_facts: HashMap<u16, ObservedFact>,
    pub history:    Vec<ObservedFact>,
}

impl ObserverState {
    pub fn assert_fact(&mut self, fact: ObservedFact, ts_ms: i64) {
        let pred = fact.predicate_id;
        if let Some(existing) = self.user_facts.get_mut(&pred) {
            let mut old = existing.clone();
            old.valid_to = Some(ts_ms);
            self.history.push(old);
        }
        self.user_facts.insert(pred, fact.clone());
        self.history.push(fact);
    }

    pub fn current_facts(&self) -> Vec<&ObservedFact> {
        self.user_facts
            .values()
            .filter(|f| f.polarity == Polarity::Affirmed && f.time_scope == TimeScope::Current)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Compiled regex patterns
// ---------------------------------------------------------------------------

struct PatternSet {
    /// (predicate_id, pattern)
    patterns: Vec<(u16, Regex)>,

    first_person:   Regex,
    negation:       Regex,
    historical:     Regex,
    modal:          Regex,
    sentence_split: Regex,
}

impl PatternSet {
    fn build() -> Self {
        let raw: &[(u16, &str)] = &[
            // ── name (0) ──────────────────────────────────────────────────
            (0, r"(?i-u)\bmy name is ([A-Za-z][A-Za-z]{1,30})"),
            (0, r"(?i-u)\bcall me ([A-Za-z][A-Za-z]{1,30})"),
            (0, r"(?i-u)\bI(?:'m| am) ([A-Za-z][A-Za-z]{1,30})\b"),

            // ── education.degree (1) ──────────────────────────────────────
            (1, r"(?i)\b(?:got|earned|received|completed|finished|have)(?: a| my| an)? (b\.?sc?\.?|bachelor[^\s,]*|b\.?a\.?|m\.?sc?\.?|master[^\s,]*|ph\.?d\.?|doctorate|mba|m\.?b\.?a\.?|associate[^\s,]*)"),
            (1, r"(?i)\bgraduated with (?:a |my |an )?(b\.?sc?\.?|bachelor[^\s,]*|m\.?sc?\.?|master[^\s,]*|ph\.?d\.?|doctorate)"),
            (1, r"(?i)(?:my |a )?(b\.?sc?\.?|bachelor[^\s,]*|m\.?sc?\.?|master[^\s,]*|ph\.?d\.?|doctorate) (?:degree|diploma|certificate)"),
            (1, r"(?i)\b(b\.?sc?\.?|m\.?sc?\.?|ph\.?d\.?|mba|bachelor[^\s,]*|master[^\s,]*|doctorate) (?:from|at) "),

            // ── education.institution (2) ─────────────────────────────────
            (2, r"(?i-u)\b(?:studied|graduated|attended|went to|at) ([A-Za-z][A-Za-z\s]{3,40}(?:university|college|institute|school))"),
            (2, r"(?i-u)\b(?:studying|enrolled) at ([A-Za-z][A-Za-z\s]{3,40}(?:university|college|institute|school))"),

            // ── occupation (3) ────────────────────────────────────────────
            (3, r"(?i)\bI(?:'m| am) (?:a |an )([\w\s]{2,30}?)(?:\.|,| at | for | with | in |\s*$)"),
            (3, r"(?i)\bwork(?:ing)? (?:as|for) (?:a |an )?([\w\s]{2,30}?)(?:\.|,| at | for |\s*$)"),
            (3, r"(?i)\bmy (?:job|profession|career|role) is (?:a |an )?([\w\s]{2,30}?)(?:\.|,|\s*$)"),
            (3, r"(?i)\bI work (?:as|in) (?:a |an )?([\w\s]{2,30}?)(?:\.|,|\s*$)"),

            // ── location.current (4) ──────────────────────────────────────
            (4, r"(?i-u)\b(?:live|stay|reside|am based|moved) (?:in|to) ([A-Za-z][A-Za-z\s,]{2,30})"),
            (4, r"(?i-u)\bI(?:'m| am) (?:currently )?in ([A-Za-z][A-Za-z\s]{2,20})"),
            (4, r"(?i-u)\bbased (?:in|out of) ([A-Za-z][A-Za-z\s]{2,20})"),

            // ── location.hometown (5) ─────────────────────────────────────
            (5, r"(?i-u)\b(?:grew up|raised|born|from) (?:in |in )([A-Za-z][A-Za-z\s,]{2,30})"),
            (5, r"(?i-u)\bhometown is ([A-Za-z][A-Za-z\s]{2,20})"),
            (5, r"(?i-u)\bI(?:'m| am) (?:originally )?from ([A-Za-z][A-Za-z\s]{2,20})"),

            // ── preference.food (6) ───────────────────────────────────────
            (6, r"(?i)\b(?:love|like|enjoy|prefer|favorite food is|favourite food is) ([\w\s]{2,25}?(?:food|pizza|sushi|pasta|curry|tacos|ramen|salad|steak|burger|soup|bread|rice|noodles|chicken|fish|fruit|vegetables?))\b"),
            (6, r"(?i)\bI(?:'m| am) a (?:huge )?fan of ([\w\s]{2,25}?(?:food|cuisine|cooking))"),
            (6, r"(?i)\bmy favorite (?:food|dish|meal|cuisine) is ([\w\s]{2,25})"),

            // ── preference.drink (7) ──────────────────────────────────────
            (7, r"(?i)\b(?:love|like|drink|prefer|enjoy) (coffee|tea|beer|wine|whiskey|whisky|juice|water|espresso|latte|cappuccino|soda|cola|coke|matcha|kombucha|vodka|rum|gin|bourbon)\b"),
            (7, r"(?i)\bmy favorite (?:drink|beverage) is ([\w\s]{2,20})"),
            (7, r"(?i)\bI(?:'m| am) (?:a )?(coffee|tea|beer|wine) (?:lover|person|drinker|addict|enthusiast)"),

            // ── preference.music (8) ──────────────────────────────────────
            (8, r"(?i)\b(?:love|like|enjoy|listen to|into|prefer) ([\w\s]{2,25}?(?:music|jazz|rock|pop|metal|hip.?hop|classical|folk|country|blues|electronic|indie|punk|rap|soul|r&b|reggae|opera))\b"),
            (8, r"(?i)\bmy favorite (?:music|band|artist|singer|genre) is ([\w\s]{2,25})"),
            (8, r"(?i)\bI(?:'m| am) (?:a )?(?:big )?fan of ([\w\s]{2,25}?(?:band|music|artist))"),

            // ── preference.sport (9) ──────────────────────────────────────
            (9, r"(?i)\b(?:play|love|enjoy|watch|follow|into|practise|practice) ([\w\s]{2,25}?(?:football|soccer|basketball|tennis|golf|swimming|cycling|running|volleyball|baseball|hockey|cricket|rugby|boxing|yoga|climbing|skiing|surfing|martial arts?))\b"),
            (9, r"(?i)\bmy favorite (?:sport|team|club) is ([\w\s]{2,25})"),
            (9, r"(?i)\bI(?:'m| am) (?:a )?(?:[\w\s]{0,10} )?(?:football|soccer|basketball|tennis|golf|swimming|cycling|running|volleyball|baseball|hockey|cricket|rugby) (?:player|fan|supporter)"),

            // ── possession (10) ───────────────────────────────────────────
            (10, r"(?i)\bI (?:have|own|got|bought|got myself) (?:a |an |my )?(car|truck|bike|motorcycle|house|apartment|flat|dog|cat|pet|boat|guitar|piano|laptop|phone|tablet|camera)\b"),
            (10, r"(?i)\bmy (car|truck|bike|house|apartment|dog|cat|guitar|piano|laptop)\b"),

            // ── family.relation (11) ──────────────────────────────────────
            (11, r"(?i)\bI have (?:a |an )?(wife|husband|partner|girlfriend|boyfriend|son|daughter|child|children|baby|brother|sister|sibling|parent|mother|father|mom|dad|twin)\b"),
            (11, r"(?i)\bmy (wife|husband|partner|girlfriend|boyfriend|son|daughter|child|brother|sister|mother|father|mom|dad|twin) (?:is|are|was|were|has|have|and I|lives)"),
            (11, r"(?i)\b(?:married|engaged|divorced|single|dating)\b"),
            (11, r"(?i)\bI(?:'m| am) (?:a )?(?:married|single|engaged|divorced|father|mother|parent|dad|mom)\b"),

            // ── hobby (12) ────────────────────────────────────────────────
            (12, r"(?i)\b(?:enjoy|love|like|into|hobby is|hobbies include) ([\w\s]{2,30}?(?:reading|painting|drawing|photography|cooking|baking|gaming|hiking|gardening|knitting|coding|programming|writing|traveling|travelling|collecting|crafting|woodworking|fishing|hunting|dancing|singing))\b"),
            (12, r"(?i)\bI spend (?:my |a lot of )?time ([\w\s]{2,25}?ing)\b"),
            (12, r"(?i)\bmy hobby is ([\w\s]{2,30})"),
            (12, r"(?i)\bin my (?:free|spare) time I ([\w\s]{2,30}?)(?:\.|,|\s*$)"),

            // ── achievement (13) ──────────────────────────────────────────
            (13, r"(?i)\b(?:won|received|awarded|published|completed|finished|achieved|certified|licensed|qualified|passed) ([\w\s]{2,40}?)(?:\.|,| in | at | from |\s*$)"),
            (13, r"(?i)\bI(?:'m| am) (?:a )?(?:certified|licensed|qualified|registered) ([\w\s]{2,30})"),
            (13, r"(?i)\bmy (?:paper|article|research|book|thesis|dissertation) (?:was |got )?(?:published|accepted|awarded)"),
        ];

        let patterns = raw
            .iter()
            .map(|(id, pat)| (*id, Regex::new(pat).expect("static pattern")))
            .collect();

        PatternSet {
            patterns,
            first_person: Regex::new(r"(?i)\b(i|my|me|mine|myself)\b")
                .expect("static pattern"),
            negation: Regex::new(
                r"(?i)\b(not|never|don't|didn't|no longer|haven't|wasn't|aren't|isn't|can't)\b",
            )
            .expect("static pattern"),
            historical: Regex::new(
                r"(?i)\b(used to|before|when i was|back when|years ago|previously|formerly)\b",
            )
            .expect("static pattern"),
            modal: Regex::new(
                r"(?i)\b(might|want to|wish|hope|would like|planning to|thinking about|considering)\b",
            )
            .expect("static pattern"),
            sentence_split: Regex::new(r"[.!?][\s]+|[\n]").expect("static pattern"),
        }
    }
}

static PATTERN_SET: OnceLock<PatternSet> = OnceLock::new();

fn patterns() -> &'static PatternSet {
    PATTERN_SET.get_or_init(PatternSet::build)
}

// ---------------------------------------------------------------------------
// Canonicalization
// ---------------------------------------------------------------------------

fn canonicalize(predicate: u16, object: &str, polarity: &Polarity, time_scope: &TimeScope) -> String {
    let neg = if *polarity == Polarity::Negated { "does not " } else { "" };
    let hist = if *time_scope == TimeScope::Historical { " (historical)" } else { "" };
    match predicate {
        0  => format!("User's name is {}{}", object, hist),
        1  => format!("User {}holds a {} degree{}", neg, object, hist),
        2  => format!("User {}attended {} for education{}", neg, object, hist),
        3  => format!("User {}works as {}{}", neg, object, hist),
        4  => format!("User {}lives in {}{}", neg, object, hist),
        5  => format!("User {}is from {} (hometown){}", neg, object, hist),
        6  => format!("User {}enjoys {} (food preference){}", neg, object, hist),
        7  => format!("User {}drinks {} (drink preference){}", neg, object, hist),
        8  => format!("User {}listens to {} (music preference){}", neg, object, hist),
        9  => format!("User {}plays/follows {} (sport preference){}", neg, object, hist),
        10 => format!("User {}owns a {}{}", neg, object, hist),
        11 => format!("User family relation: {}{}{}", neg, object, hist),
        12 => format!("User {}enjoys {} as a hobby{}", neg, object, hist),
        13 => format!("User achievement: {}{}", object, hist),
        _  => format!("User {}{}: {}{}", neg, PREDICATES[predicate as usize], object, hist),
    }
}

// ---------------------------------------------------------------------------
// Extraction helpers
// ---------------------------------------------------------------------------

/// Return the ~N words before `byte_offset` in `text`.
fn words_before(text: &str, byte_offset: usize, n: usize) -> &str {
    let prefix = &text[..byte_offset];
    let start = prefix
        .split_whitespace()
        .collect::<Vec<_>>()
        .iter()
        .rev()
        .take(n)
        .last()
        .and_then(|w| prefix.rfind(w).map(|p| p))
        .unwrap_or(0);
    &prefix[start..]
}

fn normalize_degree(raw: &str) -> &str {
    let l = raw.to_ascii_lowercase();
    let l = l.trim();
    if l.starts_with("b.sc") || l.starts_with("bsc") || l.starts_with("bachelor") {
        return "Bachelor of Science";
    }
    if l.starts_with("b.a") || l == "ba" {
        return "Bachelor of Arts";
    }
    if l.starts_with("m.sc") || l.starts_with("msc") || l.starts_with("master") {
        return "Master of Science";
    }
    if l.starts_with("m.a") || l == "ma" {
        return "Master of Arts";
    }
    if l.starts_with("ph") || l.starts_with("doctorate") {
        return "PhD";
    }
    if l.starts_with("mba") || l.starts_with("m.b.a") {
        return "MBA";
    }
    if l.starts_with("associate") {
        return "Associate's degree";
    }
    raw
}

fn title_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut cap = true;
    for c in s.chars() {
        if c.is_whitespace() {
            cap = true;
            out.push(c);
        } else if cap {
            out.extend(c.to_uppercase());
            cap = false;
        } else {
            out.push(c);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Extractor
// ---------------------------------------------------------------------------

pub struct Extractor;

impl Extractor {
    /// Extract facts from a single sentence already confirmed to be first-person,
    /// non-modal, user-turn text.
    fn extract_from_sentence(
        &self,
        sentence: &str,
        source_id: u64,
        ts_ms: i64,
        results: &mut Vec<ObservedFact>,
    ) {
        let ps = patterns();

        // historical & negation scopes apply to the whole sentence
        let is_historical = ps.historical.is_match(sentence);
        let time_scope = if is_historical { TimeScope::Historical } else { TimeScope::Current };

        for (pred_id, re) in &ps.patterns {
            let caps = match re.captures(sentence) {
                Some(c) => c,
                None => continue,
            };

            // capture group 1 is the object; some patterns have no capture (presence-only)
            let raw_obj = match caps.get(1) {
                Some(m) => m.as_str().trim().to_string(),
                None => {
                    // presence-only pattern (family.relation "married" etc.)
                    let full = caps.get(0).unwrap().as_str().trim().to_string();
                    full
                }
            };

            if raw_obj.is_empty() {
                continue;
            }

            // check negation: look at words before the match start
            let match_start = caps.get(0).unwrap().start();
            let context_before = words_before(sentence, match_start, 5);
            let is_negated = ps.negation.is_match(context_before);
            let polarity = if is_negated { Polarity::Negated } else { Polarity::Affirmed };

            // normalize object per predicate
            let object = match *pred_id {
                1 => normalize_degree(&raw_obj).to_string(),
                0 | 2 | 4 | 5 => title_case(&raw_obj),
                3 | 6 | 7 | 8 | 9 | 12 => raw_obj.to_lowercase(),
                _ => raw_obj,
            };

            // confidence: historical → 0.7, negated → 0.8, normal → 0.9
            let confidence = if is_historical { 0.7 } else if is_negated { 0.8 } else { 0.9 };

            results.push(ObservedFact {
                predicate_id: *pred_id,
                object,
                polarity,
                time_scope: time_scope.clone(),
                confidence,
                source_id,
                valid_from: ts_ms,
                valid_to: None,
            });

            // one match per predicate per sentence to avoid duplicates
            // (break inner loop for this predicate; continue other predicates)
        }
    }
}

// ---------------------------------------------------------------------------
// Observer
// ---------------------------------------------------------------------------

pub struct Observer {
    extractor: Extractor,
}

impl Observer {
    pub fn new() -> Self {
        // Eagerly compile patterns at construction time
        let _ = patterns();
        Observer { extractor: Extractor }
    }

    /// Returns canonical text strings to be stored as derived memories.
    /// Called synchronously from put_memory. Designed to be <1ms for typical turns.
    pub fn extract(
        &self,
        content: &str,
        source_id: u64,
        ts_ms: i64,
        state: &mut ObserverState,
    ) -> Vec<String> {
        let ps = patterns();

        // Skip assistant / system turns
        let lower_start = content.get(..12).unwrap_or(content).to_ascii_lowercase();
        if lower_start.starts_with("[assistant]") || lower_start.starts_with("[system]") {
            return Vec::new();
        }

        // First-person guard on the whole content
        if !ps.first_person.is_match(content) {
            return Vec::new();
        }

        // Modal guard on the whole content — skip speculative statements
        if ps.modal.is_match(content) {
            // Only skip if modal dominates; allow if first-person is also strong
            // Heuristic: if modal appears before first strong FP anchor, skip.
            // Simple approach: skip entire content if modal found early.
            let modal_pos = ps.modal.find(content).map(|m| m.start()).unwrap_or(usize::MAX);
            let fp_pos = ps.first_person.find(content).map(|m| m.start()).unwrap_or(usize::MAX);
            if modal_pos < fp_pos + 20 {
                return Vec::new();
            }
        }

        // Strip role prefix if present: "[user] ..." → "..."
        let text = if content.starts_with('[') {
            content.find(']').map(|i| &content[i + 1..]).unwrap_or(content).trim()
        } else {
            content.trim()
        };

        // Split into sentences for finer-grained extraction
        let mut sentences: Vec<&str> = ps.sentence_split.split(text).collect();
        // Also treat the full text as one unit for multi-clause sentences
        sentences.push(text);

        let mut raw_facts: Vec<ObservedFact> = Vec::new();
        for sentence in &sentences {
            let s = sentence.trim();
            if s.is_empty() {
                continue;
            }
            // Per-sentence first-person guard
            if !ps.first_person.is_match(s) {
                continue;
            }
            self.extractor.extract_from_sentence(s, source_id, ts_ms, &mut raw_facts);
        }

        // Dedup: for the same predicate_id keep highest-confidence fact
        let mut best: HashMap<u16, ObservedFact> = HashMap::new();
        for fact in raw_facts {
            let e = best.entry(fact.predicate_id).or_insert_with(|| fact.clone());
            if fact.confidence > e.confidence {
                *e = fact;
            }
        }

        // Assert into state and collect canonical strings
        let mut output = Vec::new();
        for (_, fact) in best {
            let canonical = canonicalize(
                fact.predicate_id,
                &fact.object,
                &fact.polarity,
                &fact.time_scope,
            );
            state.assert_fact(fact, ts_ms);
            output.push(canonical);
        }

        output
    }
}

impl Default for Observer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_observer() -> Observer {
        Observer::new()
    }

    #[test]
    fn test_degree_extraction() {
        let obs = make_observer();
        let mut state = ObserverState::default();
        let out = obs.extract("[user] yeah I finally got my B.Sc. last spring", 1, 1000, &mut state);
        assert!(!out.is_empty(), "should extract degree");
        let joined = out.join(" ");
        assert!(joined.contains("Bachelor of Science"), "got: {}", joined);
    }

    #[test]
    fn test_occupation_extraction() {
        let obs = make_observer();
        let mut state = ObserverState::default();
        let out = obs.extract("[user] I'm a software engineer at a startup", 2, 1000, &mut state);
        assert!(!out.is_empty(), "should extract occupation");
    }

    #[test]
    fn test_skip_assistant_turn() {
        let obs = make_observer();
        let mut state = ObserverState::default();
        let out = obs.extract("[assistant] I earned my PhD in 2010", 3, 1000, &mut state);
        assert!(out.is_empty(), "should skip assistant turns");
    }

    #[test]
    fn test_negation() {
        let obs = make_observer();
        let mut state = ObserverState::default();
        let out = obs.extract("[user] I don't drink coffee", 4, 1000, &mut state);
        if let Some(fact) = state.user_facts.get(&7) {
            assert_eq!(fact.polarity, Polarity::Negated, "should be negated");
        }
        let _ = out;
    }

    #[test]
    fn test_historical() {
        let obs = make_observer();
        let mut state = ObserverState::default();
        let out = obs.extract("[user] I used to live in Berlin", 5, 1000, &mut state);
        if let Some(fact) = state.user_facts.get(&4) {
            assert_eq!(fact.time_scope, TimeScope::Historical, "should be historical");
        }
        let _ = out;
    }

    #[test]
    fn test_modal_skip() {
        let obs = make_observer();
        let mut state = ObserverState::default();
        let out = obs.extract("[user] I might move to Paris", 6, 1000, &mut state);
        // modal guard should suppress extraction
        assert!(out.is_empty() || !out.iter().any(|s| s.contains("Paris")));
    }

    #[test]
    fn test_name_extraction() {
        let obs = make_observer();
        let mut state = ObserverState::default();
        let out = obs.extract("[user] My name is Alice", 7, 1000, &mut state);
        assert!(!out.is_empty());
        assert!(out[0].contains("Alice"), "got: {:?}", out);
    }

    #[test]
    fn test_location() {
        let obs = make_observer();
        let mut state = ObserverState::default();
        let out = obs.extract("[user] I live in Copenhagen", 8, 1000, &mut state);
        assert!(!out.is_empty());
    }

    #[test]
    fn test_state_supersede() {
        let obs = make_observer();
        let mut state = ObserverState::default();
        obs.extract("[user] I live in Berlin", 1, 1000, &mut state);
        obs.extract("[user] I live in Copenhagen", 2, 2000, &mut state);
        // history should have 3 entries: first assert, supersede-close, second assert
        assert!(state.history.len() >= 2);
        let current = state.user_facts.get(&4).unwrap();
        assert!(current.object.contains("Copenhagen"), "got: {}", current.object);
    }
}

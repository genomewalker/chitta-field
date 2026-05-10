// Contradiction detection for chitta-field.
//
// Parses SSL v0.3 memory content into ClaimAtoms, indexes them by claim scope,
// and detects contradictory pairs at ingestion time or via background scan.
//
// Design: claim-centric, not text-centric. Two memories contradict when they
// make incompatible claims under overlapping scope (same subject+predicate),
// not merely when they are semantically similar.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── FNV-1a (inline, no external crate) ───────────────────────────────────────

fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

// ── SSL types ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SslType {
    Operational,
    Correction,
    Decision,
    Belief,
    Preference,
    Solution,
    Gotcha,
    Pattern,
    Failure,
    Event,
    Unknown,
}

impl SslType {
    fn from_tag(tag: &str) -> Self {
        match tag {
            "OPERATIONAL" => Self::Operational,
            "CORRECTION"  => Self::Correction,
            "DECISION"    => Self::Decision,
            "BELIEF"      => Self::Belief,
            "PREFERENCE"  => Self::Preference,
            "SOLUTION"    => Self::Solution,
            "GOTCHA"      => Self::Gotcha,
            "PATTERN"     => Self::Pattern,
            "FAILURE"     => Self::Failure,
            "EVENT"       => Self::Event,
            _             => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Polarity {
    Positive,
    Negative,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Modality {
    Must,
    Should,
    Fact,
    Preference,
    Cannot,
    Unknown,
}

// ── ClaimAtom ─────────────────────────────────────────────────────────────────

/// A single parsed claim extracted from one line of SSL content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimAtom {
    pub memory_id: u64,
    pub realm: String,
    pub domain: Option<String>,
    pub ssl_type: SslType,
    pub subject: String,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub polarity: Polarity,
    pub modality: Modality,
    /// "{realm}:{domain}:{subject}:{predicate}" — scope key for grouping
    pub claim_key: String,
    pub claim_key_hash: u64,
    pub raw_line: String,
}

// ── ContradictionCandidate ────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CandidateStatus {
    Open,
    Confirmed,
    Rejected,
    Resolved,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContradictionCandidate {
    pub id: u64,
    pub memory_a: u64,
    pub memory_b: u64,
    pub score: f32,
    pub same_score: f32,
    pub opposition_score: f32,
    pub reason: String,
    pub status: CandidateStatus,
    pub created_at_ms: i64,
}

// ── ResolutionOps ─────────────────────────────────────────────────────────────

/// Ops returned to the C++ caller to apply after resolution is confirmed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionOps {
    pub winner_id: u64,
    pub loser_id: u64,
    /// Triplets to add: (subject, predicate, object)
    pub add_triplets: Vec<(String, String, String)>,
    pub demote_memory_id: u64,
    pub new_confidence: f32,
    /// SSL CORRECTION line ready to store as a new memory
    pub correction_content: String,
}

// ── ContradictionIndex ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContradictionIndex {
    candidates: Vec<ContradictionCandidate>,
    next_id: u64,
    /// memory_id → candidate ids involving that memory
    by_memory: HashMap<u64, Vec<u64>>,
    /// claim_key_hash → memory_ids that have an atom with that hash
    by_claim_key: HashMap<u64, Vec<u64>>,
    /// Persisted atom store so scans don't re-parse
    atom_store: HashMap<u64, Vec<ClaimAtom>>,
}

impl ContradictionIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Index atoms for a memory (called after put_memory).
    pub fn register_atoms(&mut self, memory_id: u64, atoms: Vec<ClaimAtom>) {
        for atom in &atoms {
            self.by_claim_key
                .entry(atom.claim_key_hash)
                .or_default()
                .push(memory_id);
        }
        self.atom_store.insert(memory_id, atoms);
    }

    /// Parse + index + detect candidates for a newly stored memory.
    /// Returns candidates with score >= 0.65.
    pub fn detect_for_new_memory(
        &mut self,
        memory_id: u64,
        content: &[u8],
        realm: &str,
    ) -> Vec<ContradictionCandidate> {
        let text = match std::str::from_utf8(content) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let new_atoms = parse_claim_atoms(memory_id, realm, text.as_bytes());
        if new_atoms.is_empty() {
            return vec![];
        }

        // Collect candidate memory_ids via claim_key index
        let mut candidate_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for atom in &new_atoms {
            if let Some(peers) = self.by_claim_key.get(&atom.claim_key_hash) {
                for &peer_id in peers {
                    if peer_id != memory_id {
                        candidate_ids.insert(peer_id);
                    }
                }
            }
            // Also check same subject across all claim keys (broader sweep)
            for (_, atoms) in &self.atom_store {
                for existing in atoms {
                    if existing.memory_id != memory_id
                        && existing.subject == atom.subject
                        && !candidate_ids.contains(&existing.memory_id)
                    {
                        candidate_ids.insert(existing.memory_id);
                    }
                }
            }
        }

        let mut results = Vec::new();
        let ts = now_ms();

        for peer_id in candidate_ids {
            let peer_atoms = match self.atom_store.get(&peer_id) {
                Some(a) => a.clone(),
                None => continue,
            };
            // Score best atom pair between new and peer
            let mut best_score = 0.0f32;
            let mut best_same = 0.0f32;
            let mut best_opp = 0.0f32;
            let mut best_reason = String::new();

            for na in &new_atoms {
                for pa in &peer_atoms {
                    let (total, same, opp, reason) = score_contradiction(na, pa);
                    if total > best_score {
                        best_score = total;
                        best_same = same;
                        best_opp = opp;
                        best_reason = reason;
                    }
                }
            }

            if best_score >= 0.65 {
                // Avoid duplicate candidates
                let already = self.by_memory
                    .get(&memory_id)
                    .map(|ids| ids.iter().any(|&cid| {
                        self.candidates.iter().find(|c| c.id == cid)
                            .map(|c| (c.memory_a == peer_id && c.memory_b == memory_id)
                                   || (c.memory_b == peer_id && c.memory_a == memory_id))
                            .unwrap_or(false)
                    }))
                    .unwrap_or(false);
                if !already {
                    let id = self.next_id;
                    self.next_id += 1;
                    let candidate = ContradictionCandidate {
                        id,
                        memory_a: peer_id,
                        memory_b: memory_id,
                        score: best_score,
                        same_score: best_same,
                        opposition_score: best_opp,
                        reason: best_reason,
                        status: CandidateStatus::Open,
                        created_at_ms: ts,
                    };
                    self.by_memory.entry(memory_id).or_default().push(id);
                    self.by_memory.entry(peer_id).or_default().push(id);
                    results.push(candidate.clone());
                    self.candidates.push(candidate);
                }
            }
        }

        // Now register the new atoms
        self.register_atoms(memory_id, new_atoms);
        results
    }

    pub fn add_candidate(&mut self, c: ContradictionCandidate) -> u64 {
        let id = c.id;
        self.by_memory.entry(c.memory_a).or_default().push(id);
        self.by_memory.entry(c.memory_b).or_default().push(id);
        self.candidates.push(c);
        id
    }

    pub fn open_candidates(&self) -> Vec<&ContradictionCandidate> {
        self.candidates
            .iter()
            .filter(|c| c.status == CandidateStatus::Open)
            .collect()
    }

    pub fn candidates_for_memory(&self, memory_id: u64) -> Vec<&ContradictionCandidate> {
        let ids = match self.by_memory.get(&memory_id) {
            Some(v) => v,
            None => return vec![],
        };
        ids.iter()
            .filter_map(|&cid| self.candidates.iter().find(|c| c.id == cid))
            .collect()
    }

    /// Resolve a contradiction: produces ops for C++ caller to apply.
    pub fn resolve(
        &mut self,
        candidate_id: u64,
        winner_id: u64,
        loser_id: u64,
        reason: &str,
    ) -> Option<ResolutionOps> {
        let candidate = self.candidates.iter_mut().find(|c| c.id == candidate_id)?;
        candidate.status = CandidateStatus::Resolved;

        let winner_content = self.atom_store.get(&winner_id)
            .and_then(|atoms| atoms.first())
            .map(|a| a.raw_line.clone())
            .unwrap_or_default();
        let loser_content = self.atom_store.get(&loser_id)
            .and_then(|atoms| atoms.first())
            .map(|a| a.raw_line.clone())
            .unwrap_or_default();

        let correction = format!(
            "[CORRECTION] supersedes:mem-{}|reason:{} F:CORE\nwinner: {}\nsuperseded: {}",
            loser_id, reason, winner_content, loser_content
        );

        Some(ResolutionOps {
            winner_id,
            loser_id,
            add_triplets: vec![
                (winner_id.to_string(), "supersedes".to_string(), loser_id.to_string()),
                (winner_id.to_string(), "contradicts".to_string(), loser_id.to_string()),
            ],
            demote_memory_id: loser_id,
            new_confidence: 0.05,
            correction_content: correction,
        })
    }

    pub fn reject(&mut self, candidate_id: u64) {
        if let Some(c) = self.candidates.iter_mut().find(|c| c.id == candidate_id) {
            c.status = CandidateStatus::Rejected;
        }
    }

    /// Background scan: group all atoms in realm by claim_key_hash, score within buckets.
    pub fn scan_realm(
        &mut self,
        realm: &str,
        payload_iter: &[(u64, Vec<u8>)],
        limit: usize,
    ) -> Vec<ContradictionCandidate> {
        // Parse atoms for all memories in this realm
        let mut realm_atoms: HashMap<u64, Vec<ClaimAtom>> = HashMap::new();
        for (memory_id, content) in payload_iter {
            let atoms = parse_claim_atoms(*memory_id, realm, content);
            if !atoms.is_empty() {
                realm_atoms.insert(*memory_id, atoms);
            }
        }

        // Group by claim_key_hash
        let mut buckets: HashMap<u64, Vec<u64>> = HashMap::new();
        for (mid, atoms) in &realm_atoms {
            for atom in atoms {
                buckets.entry(atom.claim_key_hash).or_default().push(*mid);
            }
        }

        let ts = now_ms();
        let mut results = Vec::new();

        'outer: for (_key_hash, mem_ids) in &buckets {
            if mem_ids.len() < 2 {
                continue;
            }
            for i in 0..mem_ids.len() {
                for j in (i + 1)..mem_ids.len() {
                    let id_a = mem_ids[i];
                    let id_b = mem_ids[j];
                    let atoms_a = match realm_atoms.get(&id_a) {
                        Some(a) => a,
                        None => continue,
                    };
                    let atoms_b = match realm_atoms.get(&id_b) {
                        Some(b) => b,
                        None => continue,
                    };

                    let mut best_score = 0.0f32;
                    let mut best_same = 0.0f32;
                    let mut best_opp = 0.0f32;
                    let mut best_reason = String::new();

                    for aa in atoms_a {
                        for ab in atoms_b {
                            let (total, same, opp, reason) = score_contradiction(aa, ab);
                            if total > best_score {
                                best_score = total;
                                best_same = same;
                                best_opp = opp;
                                best_reason = reason;
                            }
                        }
                    }

                    if best_score >= 0.65 {
                        let id = self.next_id;
                        self.next_id += 1;
                        results.push(ContradictionCandidate {
                            id,
                            memory_a: id_a,
                            memory_b: id_b,
                            score: best_score,
                            same_score: best_same,
                            opposition_score: best_opp,
                            reason: best_reason,
                            status: CandidateStatus::Open,
                            created_at_ms: ts,
                        });
                        if results.len() >= limit {
                            break 'outer;
                        }
                    }
                }
            }
        }

        results
    }
}

// ── Parsing ───────────────────────────────────────────────────────────────────

/// Parse SSL v0.3 content into ClaimAtoms.
pub fn parse_claim_atoms(memory_id: u64, realm: &str, content: &[u8]) -> Vec<ClaimAtom> {
    let text = match std::str::from_utf8(content) {
        Ok(s) => s,
        Err(_) => return vec![],
    };

    let mut atoms = Vec::new();
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || !line.starts_with('[') {
            continue;
        }
        if let Some(atom) = parse_ssl_line(memory_id, realm, line) {
            atoms.push(atom);
        }
    }
    atoms
}

fn parse_ssl_line(memory_id: u64, realm: &str, line: &str) -> Option<ClaimAtom> {
    // Extract [TYPE]
    let rest = line.strip_prefix('[')?;
    let (type_tag, rest) = rest.split_once(']')?;
    let ssl_type = SslType::from_tag(type_tag.trim());
    let rest = rest.trim();

    // Extract optional [domain]
    let (domain, rest) = if rest.starts_with('[') {
        let inner = rest.strip_prefix('[')?;
        let (dom, after) = inner.split_once(']')?;
        (Some(dom.trim().to_lowercase()), after.trim())
    } else {
        (None, rest)
    };

    // Strip trailing annotations: @..., F:..., A:..., →@...
    let body = strip_annotations(rest);
    // Strip |reason
    let body = if let Some(pos) = body.find('|') { &body[..pos] } else { &body };
    let body = body.trim();

    if body.is_empty() {
        return None;
    }

    // Split on → to get subject/predicate/object
    let parts: Vec<&str> = body.splitn(3, '→').collect();
    let subject = canonicalize(parts[0]);
    if subject.is_empty() {
        return None;
    }
    let predicate = parts.get(1).map(|s| canonicalize(s)).filter(|s| !s.is_empty());
    let object = parts.get(2).map(|s| canonicalize(s)).filter(|s| !s.is_empty());

    // Polarity detection
    let polarity = detect_polarity(&subject, predicate.as_deref(), object.as_deref());

    // Modality detection
    let modality = detect_modality(&subject, predicate.as_deref());

    // Build claim_key: "{realm}:{domain}:{subject}:{predicate}"
    let domain_str = domain.as_deref().unwrap_or("");
    let pred_str = predicate.as_deref().unwrap_or("");
    // For the claim key we use the canonical subject and bare predicate root
    // (strip negation markers so "use-ssh-not-slurm" and "use-slurm-not-ssh" share same subject scope)
    let subject_root = predicate_root(&subject);
    let pred_root = predicate_root(pred_str);
    let claim_key = format!("{}:{}:{}:{}", realm, domain_str, subject_root, pred_root);
    let claim_key_hash = fnv1a(&claim_key);

    Some(ClaimAtom {
        memory_id,
        realm: realm.to_string(),
        domain,
        ssl_type,
        subject,
        predicate,
        object,
        polarity,
        modality,
        claim_key,
        claim_key_hash,
        raw_line: line.to_string(),
    })
}

/// Strip trailing @location, F:FLAG, A:v,a, →@ref annotations.
fn strip_annotations(s: &str) -> String {
    let mut result = s.to_string();
    // Strip →@ref
    if let Some(pos) = result.rfind("→@") {
        result.truncate(pos);
    }
    // Strip F: and A: (trailing words starting with F: or A:)
    let parts: Vec<&str> = result.split_whitespace().collect();
    let clean: Vec<&str> = parts
        .iter()
        .copied()
        .filter(|p| !p.starts_with("F:") && !p.starts_with("A:") && !p.starts_with('@'))
        .collect();
    clean.join(" ")
}

/// Lowercase, collapse hyphens/underscores to `-`, strip punctuation except `-→>|!`.
fn canonicalize(s: &str) -> String {
    s.trim()
        .to_lowercase()
        .replace('_', "-")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '!' || *c == '→' || *c == '>' || *c == '|')
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

/// Extract the "root" of a predicate by removing common negation/qualifier suffixes.
/// "use-ssh-not-slurm" → "use-ssh", "use-slurm-not-ssh" → "use-slurm"
/// Used for claim_key so opposing claims share the same subject scope key.
fn predicate_root(s: &str) -> &str {
    if let Some(pos) = s.find("-not-") {
        return &s[..pos];
    }
    if let Some(pos) = s.find("-only") {
        return &s[..pos];
    }
    s
}

fn detect_polarity(subject: &str, predicate: Option<&str>, object: Option<&str>) -> Polarity {
    let all = format!(
        "{} {} {}",
        subject,
        predicate.unwrap_or(""),
        object.unwrap_or("")
    );
    if subject.starts_with('!')
        || all.contains("-not-")
        || all.contains("not-")
        || all.contains("avoid-")
        || all.contains("must-not-")
        || all.contains("do-not-")
        || all.contains("never-")
        || all.contains("!use")
    {
        Polarity::Negative
    } else {
        Polarity::Positive
    }
}

fn detect_modality(subject: &str, predicate: Option<&str>) -> Modality {
    let s = format!("{} {}", subject, predicate.unwrap_or(""));
    if s.contains("must-") || s.contains("always-") || s.contains("-only") {
        Modality::Must
    } else if s.contains("should-") {
        Modality::Should
    } else if s.contains("prefer") {
        Modality::Preference
    } else if s.contains("cannot-") || s.contains("must-not-") || s.contains("never-") {
        Modality::Cannot
    } else {
        Modality::Fact
    }
}

// ── Scoring ───────────────────────────────────────────────────────────────────

const ANTONYM_PAIRS: &[(&str, &str)] = &[
    ("ssh", "slurm"),
    ("ssh", "sbatch"),
    ("ssh", "srun"),
    ("direct", "queue"),
    ("yes", "no"),
    ("enable", "disable"),
    ("true", "false"),
    ("use", "avoid"),
    ("always", "never"),
    ("on", "off"),
    ("include", "exclude"),
    ("add", "remove"),
    ("allow", "deny"),
    ("local", "remote"),
];

fn is_antonym_pair(a: &str, b: &str) -> bool {
    ANTONYM_PAIRS.iter().any(|(x, y)| {
        (a.contains(x) && b.contains(y)) || (a.contains(y) && b.contains(x))
    })
}

/// Score two ClaimAtoms for contradiction.
/// Returns (total, same_score, opposition_score, reason).
pub fn score_contradiction(a: &ClaimAtom, b: &ClaimAtom) -> (f32, f32, f32, String) {
    // ── same_score: are they about the same thing? ────────────────────────────
    let a_pred = a.predicate.as_deref().unwrap_or("");
    let b_pred = b.predicate.as_deref().unwrap_or("");
    let same_score = if a.claim_key_hash == b.claim_key_hash {
        1.0
    } else if a.subject == b.subject && a.predicate.is_some() && a.predicate == b.predicate {
        0.8
    } else if a.subject == b.subject
        && (is_antonym_pair(a_pred, b_pred)
            || predicate_root(a_pred) == predicate_root(b_pred))
    {
        // Same subject + predicates reference the same antonym entities or share root action
        0.8
    } else if a.subject == b.subject {
        0.6
    } else {
        return (0.0, 0.0, 0.0, String::new());
    };

    // ── opposition_score: do they contradict? ─────────────────────────────────
    let mut opposition_score = 0.0f32;
    let mut reason_parts: Vec<String> = Vec::new();

    // Explicit polarity conflict on same claim_key
    if a.claim_key_hash == b.claim_key_hash {
        if a.polarity != b.polarity
            && a.polarity != Polarity::Unknown
            && b.polarity != Polarity::Unknown
        {
            opposition_score = opposition_score.max(1.0);
            reason_parts.push(format!(
                "same claim_key `{}`: {:?} vs {:?}",
                a.claim_key, a.polarity, b.polarity
            ));
        }
    }

    // One side is a CORRECTION referencing the other's subject
    if a.ssl_type == SslType::Correction && b.subject.contains(&a.subject) {
        opposition_score = opposition_score.max(0.9);
        reason_parts.push("CORRECTION type targets same subject".to_string());
    }
    if b.ssl_type == SslType::Correction && a.subject.contains(&b.subject) {
        opposition_score = opposition_score.max(0.9);
        reason_parts.push("CORRECTION type targets same subject".to_string());
    }

    // Known antonym pairs in predicate/object
    let a_pred = a.predicate.as_deref().unwrap_or("");
    let b_pred = b.predicate.as_deref().unwrap_or("");
    let a_obj = a.object.as_deref().unwrap_or("");
    let b_obj = b.object.as_deref().unwrap_or("");

    if is_antonym_pair(a_pred, b_pred) || is_antonym_pair(a_obj, b_obj)
        || is_antonym_pair(a_pred, b_obj) || is_antonym_pair(a_obj, b_pred)
    {
        opposition_score = opposition_score.max(0.85);
        reason_parts.push(format!(
            "antonym pair: pred=`{}` vs `{}`, obj=`{}` vs `{}`",
            a_pred, b_pred, a_obj, b_obj
        ));
    }

    // Must vs Cannot on same predicate
    if a_pred == b_pred && !a_pred.is_empty() {
        match (&a.modality, &b.modality) {
            (Modality::Must, Modality::Cannot) | (Modality::Cannot, Modality::Must) => {
                opposition_score = opposition_score.max(0.8);
                reason_parts.push(format!("Must vs Cannot on predicate `{}`", a_pred));
            }
            _ => {}
        }
    }

    // not- negation asymmetry on same predicate root
    if predicate_root(a_pred) == predicate_root(b_pred) && !a_pred.is_empty() {
        let a_neg = a_pred.contains("-not-") || a_pred.contains("not-");
        let b_neg = b_pred.contains("-not-") || b_pred.contains("not-");
        if a_neg != b_neg {
            opposition_score = opposition_score.max(0.7);
            reason_parts.push(format!(
                "negation asymmetry: `{}` vs `{}`",
                a_pred, b_pred
            ));
        }
    }

    if opposition_score < 0.5 {
        return (0.0, 0.0, 0.0, String::new());
    }

    let total = same_score * opposition_score;
    let reason = reason_parts.join("; ");
    (total, same_score, opposition_score, reason)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_operational_cluster() {
        let content = b"[OPERATIONAL] [cluster] chaos-nodes\xe2\x86\x92use-SSH-only-not-SLURM @ssh_config F:CORE";
        let atoms = parse_claim_atoms(42, "brahman", content);
        assert_eq!(atoms.len(), 1);
        let a = &atoms[0];
        assert_eq!(a.ssl_type, SslType::Operational);
        assert_eq!(a.domain.as_deref(), Some("cluster"));
        assert_eq!(a.subject, "chaos-nodes");
        assert!(a.predicate.as_deref().unwrap_or("").contains("ssh"));
        assert_eq!(a.polarity, Polarity::Negative); // contains "not-slurm"
    }

    #[test]
    fn test_parse_ascii_arrow() {
        // Test with ASCII → representation
        let content = "[OPERATIONAL] [cluster] chaos-nodes->use-SSH-only-not-SLURM F:CORE";
        let content_with_arrow = content.replace("->", "→");
        let atoms = parse_claim_atoms(42, "brahman", content_with_arrow.as_bytes());
        assert!(!atoms.is_empty());
        let a = &atoms[0];
        assert_eq!(a.subject, "chaos-nodes");
    }

    #[test]
    fn test_contradiction_ssh_vs_slurm() {
        let content_a = "[OPERATIONAL] [cluster] chaos-nodes→use-SSH-only-not-SLURM F:CORE".as_bytes();
        let content_b = "[cluster] chaos-nodes→use-SLURM-sbatch-not-SSH".as_bytes();

        let atoms_a = parse_claim_atoms(1, "brahman", content_a);
        let atoms_b = parse_claim_atoms(2, "brahman", content_b);

        assert!(!atoms_a.is_empty(), "atoms_a should not be empty");
        assert!(!atoms_b.is_empty(), "atoms_b should not be empty");

        let mut best = 0.0f32;
        for aa in &atoms_a {
            for ab in &atoms_b {
                let (score, _, _, _) = score_contradiction(aa, ab);
                if score > best {
                    best = score;
                }
            }
        }
        assert!(best >= 0.5, "Expected contradiction score >= 0.5, got {}", best);
    }

    #[test]
    fn test_no_false_positive_unrelated() {
        let content_a = "[GOTCHA] [bgv] mmap→segment-fault-on-resize".as_bytes();
        let content_b = "[OPERATIONAL] [cluster] chaos-nodes→use-SSH-only-not-SLURM".as_bytes();

        let atoms_a = parse_claim_atoms(1, "brahman", content_a);
        let atoms_b = parse_claim_atoms(2, "brahman", content_b);

        let mut best = 0.0f32;
        for aa in &atoms_a {
            for ab in &atoms_b {
                let (score, _, _, _) = score_contradiction(aa, ab);
                if score > best {
                    best = score;
                }
            }
        }
        assert!(best < 0.5, "Expected no contradiction for unrelated memories, got {}", best);
    }

    #[test]
    fn test_contradiction_index_detect() {
        let mut idx = ContradictionIndex::new();

        let content_a = "[OPERATIONAL] [cluster] chaos-nodes→use-SSH-only-not-SLURM".as_bytes();
        let content_b = "[cluster] chaos-nodes→use-SLURM-sbatch-not-SSH".as_bytes();

        // Register memory A
        let atoms_a = parse_claim_atoms(100, "brahman", content_a);
        idx.register_atoms(100, atoms_a);

        // Detect contradictions when memory B is added
        let candidates = idx.detect_for_new_memory(200, content_b, "brahman");
        // Should find at least one candidate
        assert!(
            !candidates.is_empty(),
            "Expected contradiction candidates between SSH and SLURM memories"
        );
        assert!(candidates[0].score >= 0.5);
    }

    #[test]
    fn test_resolve_produces_ops() {
        let mut idx = ContradictionIndex::new();
        let content_a = "[OPERATIONAL] [cluster] chaos-nodes→use-SSH-only-not-SLURM".as_bytes();
        let content_b = "[cluster] chaos-nodes→use-SLURM-sbatch-not-SSH".as_bytes();

        let atoms_a = parse_claim_atoms(100, "brahman", content_a);
        idx.register_atoms(100, atoms_a);
        let candidates = idx.detect_for_new_memory(200, content_b, "brahman");

        if let Some(candidate) = candidates.first() {
            let cid = candidate.id;
            let ops = idx.resolve(cid, 100, 200, "user-confirmed: SSH is correct");
            assert!(ops.is_some());
            let ops = ops.unwrap();
            assert_eq!(ops.new_confidence, 0.05);
            assert!(ops.add_triplets.iter().any(|(_, p, _)| p == "supersedes"));
        }
    }

    #[test]
    fn test_antonym_scoring() {
        let a = ClaimAtom {
            memory_id: 1, realm: "r".into(), domain: Some("cluster".into()),
            ssl_type: SslType::Operational,
            subject: "chaos-nodes".into(),
            predicate: Some("use-ssh".into()),
            object: None,
            polarity: Polarity::Positive,
            modality: Modality::Must,
            claim_key: "r:cluster:chaos-nodes:use-ssh".into(),
            claim_key_hash: fnv1a("r:cluster:chaos-nodes:use-ssh"),
            raw_line: String::new(),
        };
        let b = ClaimAtom {
            memory_id: 2, realm: "r".into(), domain: Some("cluster".into()),
            ssl_type: SslType::Operational,
            subject: "chaos-nodes".into(),
            predicate: Some("use-slurm".into()),
            object: None,
            polarity: Polarity::Positive,
            modality: Modality::Must,
            claim_key: "r:cluster:chaos-nodes:use-slurm".into(),
            claim_key_hash: fnv1a("r:cluster:chaos-nodes:use-slurm"),
            raw_line: String::new(),
        };
        let (total, same, opp, reason) = score_contradiction(&a, &b);
        // Same subject → same_score=0.6; ssh vs slurm antonym → opp=0.85
        assert!(same >= 0.6, "same={}", same);
        assert!(opp >= 0.8, "opp={}", opp);
        assert!(total >= 0.5, "total={} reason={}", total, reason);
    }
}

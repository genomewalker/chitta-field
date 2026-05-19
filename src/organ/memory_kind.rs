/// CEC Phase 16 — Memory-Kind Lattice.
///
/// Typed memory classification enforced at the association-edge level.
/// Prevents rationalization chains (observation→thought as evidence)
/// by making illegal edges structurally detectable.
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryKind {
    /// Internal speculation, synthesis, internal reasoning. May not be cited as evidence.
    Thought,
    /// Recorded external fact: tool output, user statement, measured result.
    Observation,
    /// Chosen action with rationale. May cite observations and code facts, not thoughts.
    Decision,
    /// Symbol, callgraph entry, file reference, type annotation.
    CodeFact,
}

impl MemoryKind {
    /// Infer kind from the existing `kind` string and content markers.
    /// Called at read time — no new field added to MemoryPayload.
    pub fn infer(kind: &str, realm: &str, content_prefix: &str) -> Self {
        let k = kind.to_lowercase();
        let c = content_prefix;

        if matches!(k.as_str(),
            "code_context" | "read_symbol" | "find_symbol" | "code_fact" |
            "symbol" | "callgraph" | "search_symbols" | "read_function"
        ) { return Self::CodeFact; }

        if realm.contains("code") || realm.contains("symbol") {
            return Self::CodeFact;
        }

        if c.contains("[decision]") || c.contains("[decided]") || k.contains("decision") {
            return Self::Decision;
        }

        if c.contains("[thought]") || c.contains("[synthesis]") || k.contains("thought") {
            return Self::Thought;
        }

        Self::Observation
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Thought     => "thought",
            Self::Observation => "observation",
            Self::Decision    => "decision",
            Self::CodeFact    => "code_fact",
        }
    }
}

/// Static edge-legality matrix. Returns false for rationalization chains.
///
/// Illegal: observation→thought (observation must not cite speculation as evidence).
/// Illegal: decision→thought (decision based on pure speculation, not fact).
pub fn edge_legal(from: MemoryKind, to: MemoryKind) -> bool {
    use MemoryKind::*;
    !matches!((from, to), (Observation, Thought) | (Decision, Thought))
}

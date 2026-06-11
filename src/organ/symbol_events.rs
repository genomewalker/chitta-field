use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::ids::MemoryId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolEventKind {
    Edited,
    Created,
    Deleted,
    TestFailed,
    TestPassed,
    Committed,
}

impl SymbolEventKind {
    pub fn to_u8(self) -> u8 {
        match self {
            Self::Edited    => 0,
            Self::Created   => 1,
            Self::Deleted   => 2,
            Self::TestFailed => 3,
            Self::TestPassed => 4,
            Self::Committed => 5,
        }
    }
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Edited,
            1 => Self::Created,
            2 => Self::Deleted,
            3 => Self::TestFailed,
            4 => Self::TestPassed,
            5 => Self::Committed,
            _ => Self::Edited,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Edited    => "edited",
            Self::Created   => "created",
            Self::Deleted   => "deleted",
            Self::TestFailed => "test_failed",
            Self::TestPassed => "test_passed",
            Self::Committed => "committed",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolEvent {
    pub id: u64,
    pub symbol_name: String,
    pub file_path: String,
    pub symbol_id: Option<u64>,
    pub kind: SymbolEventKind,
    pub session_id: String,
    pub harness: String,
    pub memory_id: Option<MemoryId>,
    pub timestamp_ms: i64,
    pub notes: Option<String>,
}

#[derive(Debug, Default)]
pub struct SymbolEventLog {
    next_id: u64,
    events: Vec<SymbolEvent>,
    // index: symbol_name → indices into events
    by_symbol_name: HashMap<String, Vec<usize>>,
    // index: file_path → indices into events
    by_file: HashMap<String, Vec<usize>>,
}

impl SymbolEventLog {
    pub fn new() -> Self { Self::default() }

    pub fn log(&mut self, mut ev: SymbolEvent) -> u64 {
        if ev.id == 0 {
            self.next_id += 1;
            ev.id = self.next_id;
        } else if ev.id >= self.next_id {
            self.next_id = ev.id + 1;
        }
        let idx = self.events.len();
        self.by_symbol_name.entry(ev.symbol_name.clone()).or_default().push(idx);
        self.by_file.entry(ev.file_path.clone()).or_default().push(idx);
        self.events.push(ev);
        self.events[idx].id
    }

    /// Called during WAL replay — same as log() but preserves supplied id.
    pub fn replay(&mut self, ev: SymbolEvent) {
        self.log(ev);
    }

    pub fn query_by_symbol(&self, symbol_name: &str, limit: usize) -> Vec<&SymbolEvent> {
        let indices = match self.by_symbol_name.get(symbol_name) {
            Some(v) => v,
            None => return vec![],
        };
        indices.iter().rev().take(limit).map(|&i| &self.events[i]).collect()
    }

    pub fn query_by_file(&self, file_path: &str, limit: usize) -> Vec<&SymbolEvent> {
        let indices = match self.by_file.get(file_path) {
            Some(v) => v,
            None => return vec![],
        };
        indices.iter().rev().take(limit).map(|&i| &self.events[i]).collect()
    }

    pub fn query(&self, symbol_name: Option<&str>, file_path: Option<&str>, limit: usize) -> Vec<&SymbolEvent> {
        match (symbol_name, file_path) {
            (Some(s), None) => self.query_by_symbol(s, limit),
            (None, Some(f)) => self.query_by_file(f, limit),
            (Some(s), Some(f)) => {
                // intersection: symbols matching both
                self.query_by_symbol(s, limit * 2)
                    .into_iter()
                    .filter(|e| e.file_path == f)
                    .take(limit)
                    .collect()
            }
            (None, None) => {
                self.events.iter().rev().take(limit).collect()
            }
        }
    }

    pub fn timeline_for_symbol(&self, symbol_name: &str) -> Vec<&SymbolEvent> {
        let indices = match self.by_symbol_name.get(symbol_name) {
            Some(v) => v,
            None => return vec![],
        };
        let mut evs: Vec<&SymbolEvent> = indices.iter().map(|&i| &self.events[i]).collect();
        evs.sort_by_key(|e| e.timestamp_ms);
        evs
    }

    pub fn len(&self) -> usize { self.events.len() }
    pub fn is_empty(&self) -> bool { self.events.is_empty() }
}

impl crate::organ::OrganApply for SymbolEventLog {
    /// Organ-owned WAL replay (THEORY.md §8 Phase 2).
    fn apply(&mut self, op: crate::ops::Op) -> Option<crate::ops::Op> {
        use crate::ops::Op;
        match op {
            Op::SymbolEvent(e) => {
                self.replay(SymbolEvent {
                    id: e.id,
                    symbol_name: e.symbol_name,
                    file_path: e.file_path,
                    symbol_id: e.symbol_id,
                    kind: SymbolEventKind::from_u8(e.kind),
                    session_id: e.session_id,
                    harness: e.harness,
                    memory_id: e.memory_id,
                    timestamp_ms: e.timestamp_ms,
                    notes: e.notes,
                });
                    None
                }
            other => Some(other),
        }
    }
}

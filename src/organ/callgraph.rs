use super::symbol::SymbolId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallGraph {
    callees: HashMap<SymbolId, HashSet<SymbolId>>,
    callers: HashMap<SymbolId, HashSet<SymbolId>>,
}

impl CallGraph {
    pub fn new() -> Self {
        Self {
            callees: HashMap::new(),
            callers: HashMap::new(),
        }
    }

    pub fn add_edge(&mut self, caller: SymbolId, callee: SymbolId) {
        self.callees
            .entry(caller)
            .or_insert_with(HashSet::new)
            .insert(callee);
        self.callers
            .entry(callee)
            .or_insert_with(HashSet::new)
            .insert(caller);
    }

    pub fn remove_symbol(&mut self, id: SymbolId) {
        if let Some(callees) = self.callees.remove(&id) {
            for callee in callees {
                if let Some(callers) = self.callers.get_mut(&callee) {
                    callers.remove(&id);
                }
            }
        }
        if let Some(callers) = self.callers.remove(&id) {
            for caller in callers {
                if let Some(callees) = self.callees.get_mut(&caller) {
                    callees.remove(&id);
                }
            }
        }
    }

    pub fn get_callees(&self, id: SymbolId) -> Vec<SymbolId> {
        self.callees
            .get(&id)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default()
    }

    pub fn get_callers(&self, id: SymbolId) -> Vec<SymbolId> {
        self.callers
            .get(&id)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default()
    }

    pub fn has_edge(&self, caller: SymbolId, callee: SymbolId) -> bool {
        self.callees
            .get(&caller)
            .map(|s| s.contains(&callee))
            .unwrap_or(false)
    }
}

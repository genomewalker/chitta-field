use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Agent identity record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRecord {
    pub agent_id: String,
    pub display_name: String,
    pub description: String,
    pub status: AgentStatus,
    pub first_seen_ms: i64,
    pub last_seen_ms: i64,
    pub memory_count: u64,
    pub session_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentStatus {
    Active,
    Inactive,
    Revoked,
}

impl Default for AgentStatus {
    fn default() -> Self {
        Self::Active
    }
}

/// Per-chain agent identity registry.
#[derive(Debug, Default)]
pub struct AgentRegistry {
    agents: HashMap<String, AgentRecord>,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register or update an agent. Returns true if newly created.
    pub fn upsert(
        &mut self,
        agent_id: &str,
        display_name: &str,
        description: &str,
        ts_ms: i64,
    ) -> bool {
        if let Some(record) = self.agents.get_mut(agent_id) {
            if !display_name.is_empty() {
                record.display_name = display_name.to_string();
            }
            if !description.is_empty() {
                record.description = description.to_string();
            }
            record.last_seen_ms = ts_ms;
            false
        } else {
            self.agents.insert(
                agent_id.to_string(),
                AgentRecord {
                    agent_id: agent_id.to_string(),
                    display_name: display_name.to_string(),
                    description: description.to_string(),
                    status: AgentStatus::Active,
                    first_seen_ms: ts_ms,
                    last_seen_ms: ts_ms,
                    memory_count: 0,
                    session_count: 0,
                },
            );
            true
        }
    }

    /// Record activity: increment memory count and touch timestamp.
    pub fn record_activity(&mut self, agent_id: &str, ts_ms: i64) {
        if let Some(record) = self.agents.get_mut(agent_id) {
            record.memory_count += 1;
            record.last_seen_ms = ts_ms;
        }
    }

    /// Record a new session for an agent.
    pub fn record_session(&mut self, agent_id: &str, ts_ms: i64) {
        if let Some(record) = self.agents.get_mut(agent_id) {
            record.session_count += 1;
            record.last_seen_ms = ts_ms;
        }
    }

    /// Get a specific agent record.
    pub fn get(&self, agent_id: &str) -> Option<&AgentRecord> {
        self.agents.get(agent_id)
    }

    /// List all agents.
    pub fn list(&self) -> Vec<&AgentRecord> {
        self.agents.values().collect()
    }

    /// Disable an agent.
    pub fn disable(&mut self, agent_id: &str) -> bool {
        if let Some(record) = self.agents.get_mut(agent_id) {
            record.status = AgentStatus::Revoked;
            true
        } else {
            false
        }
    }

    /// Total agent count.
    pub fn count(&self) -> usize {
        self.agents.len()
    }
}

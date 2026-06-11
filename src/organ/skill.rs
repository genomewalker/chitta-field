use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single immutable skill version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillVersion {
    pub skill_id: String,
    pub version: u32,
    pub content: String,
    pub uploaded_by: String,
    pub tags: Vec<String>,
    pub created_at_ms: i64,
    pub deprecated: bool,
}

/// Append-only versioned skill registry.
/// Each skill_id maps to an ordered list of immutable versions.
#[derive(Debug, Default)]
pub struct SkillRegistry {
    skills: HashMap<String, Vec<SkillVersion>>,
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Upload a new skill version. Returns the assigned version number.
    pub fn upload(
        &mut self,
        skill_id: &str,
        content: &str,
        uploaded_by: &str,
        tags: &[String],
        ts_ms: i64,
    ) -> u32 {
        let versions = self.skills.entry(skill_id.to_string()).or_default();
        let version = versions.len() as u32 + 1;
        versions.push(SkillVersion {
            skill_id: skill_id.to_string(),
            version,
            content: content.to_string(),
            uploaded_by: uploaded_by.to_string(),
            tags: tags.to_vec(),
            created_at_ms: ts_ms,
            deprecated: false,
        });
        version
    }

    /// Read a specific version (0 = latest).
    pub fn read(&self, skill_id: &str, version: u32) -> Option<&SkillVersion> {
        let versions = self.skills.get(skill_id)?;
        if version == 0 {
            versions.last()
        } else {
            versions.get((version - 1) as usize)
        }
    }

    /// List all versions for a skill.
    pub fn versions(&self, skill_id: &str) -> Vec<&SkillVersion> {
        self.skills
            .get(skill_id)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// List all skill IDs with their latest version number.
    pub fn list(&self) -> Vec<(&str, u32)> {
        self.skills
            .iter()
            .map(|(id, versions)| (id.as_str(), versions.len() as u32))
            .collect()
    }

    /// Search skills by text match on id, tags, or content.
    pub fn search(&self, query: &str, limit: usize) -> Vec<&SkillVersion> {
        let q = query.to_lowercase();
        let mut results = Vec::new();
        for versions in self.skills.values() {
            if let Some(latest) = versions.last() {
                if latest.deprecated {
                    continue;
                }
                let matches = latest.skill_id.to_lowercase().contains(&q)
                    || latest.tags.iter().any(|t| t.to_lowercase().contains(&q))
                    || latest.content.to_lowercase().contains(&q);
                if matches {
                    results.push(latest);
                }
            }
        }
        results.truncate(limit);
        results
    }

    /// Deprecate a skill (marks latest version).
    pub fn deprecate(&mut self, skill_id: &str) -> bool {
        if let Some(versions) = self.skills.get_mut(skill_id) {
            if let Some(latest) = versions.last_mut() {
                latest.deprecated = true;
                return true;
            }
        }
        false
    }

    /// Total skill count.
    pub fn count(&self) -> usize {
        self.skills.len()
    }
}

impl crate::organ::OrganApply for SkillRegistry {
    /// Organ-owned WAL replay (THEORY.md §8 Phase 2).
    fn apply(&mut self, op: crate::ops::Op) -> Option<crate::ops::Op> {
        use crate::ops::Op;
        match op {
            Op::SkillUpload(s) => {
                self.upload(&s.skill_id, &s.content, &s.uploaded_by, &s.tags, s.ts_ms);
                    None
                }
            Op::SkillDeprecate(s) => {
                self.deprecate(&s.skill_id);
                    None
                }
            other => Some(other),
        }
    }
}

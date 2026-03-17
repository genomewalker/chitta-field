use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type CodeFileId = u64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFile {
    pub id: CodeFileId,
    pub path: String,
    pub project: String,
    pub mtime: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFileIndex {
    by_id: HashMap<CodeFileId, CodeFile>,
    by_path: HashMap<String, CodeFileId>,
}

impl CodeFileIndex {
    pub fn new() -> Self {
        Self {
            by_id: HashMap::new(),
            by_path: HashMap::new(),
        }
    }

    /// Upsert a file. Returns (id, was_updated).
    pub fn upsert(
        &mut self,
        path: &str,
        project: &str,
        mtime: i64,
        next_id: impl FnOnce() -> CodeFileId,
    ) -> (CodeFileId, bool) {
        if let Some(&existing_id) = self.by_path.get(path) {
            let entry = self.by_id.get_mut(&existing_id).unwrap();
            let changed = entry.mtime != mtime || entry.project != project;
            entry.mtime = mtime;
            entry.project = project.to_string();
            return (existing_id, changed);
        }

        let id = next_id();
        let file = CodeFile {
            id,
            path: path.to_string(),
            project: project.to_string(),
            mtime,
        };
        self.by_path.insert(path.to_string(), id);
        self.by_id.insert(id, file);
        (id, true)
    }

    pub fn get_by_path(&self, path: &str) -> Option<&CodeFile> {
        let &id = self.by_path.get(path)?;
        self.by_id.get(&id)
    }

    pub fn get_by_project(&self, project: &str) -> Vec<&CodeFile> {
        self.by_id
            .values()
            .filter(|f| f.project == project)
            .collect()
    }

    pub fn remove(&mut self, id: CodeFileId) {
        if let Some(file) = self.by_id.remove(&id) {
            self.by_path.remove(&file.path);
        }
    }

    pub fn count(&self) -> usize {
        self.by_id.len()
    }

    /// Return the maximum code file ID present, or None if empty.
    pub fn max_id(&self) -> Option<u64> {
        self.by_id.keys().copied().max()
    }

    /// Iterate all code files.
    pub fn iter(&self) -> impl Iterator<Item = &CodeFile> {
        self.by_id.values()
    }

    /// Remove all code files belonging to a project. Returns removed file paths.
    pub fn remove_by_project(&mut self, project: &str) -> Vec<String> {
        let ids_to_remove: Vec<CodeFileId> = self
            .by_id
            .iter()
            .filter(|(_, f)| f.project == project)
            .map(|(&id, _)| id)
            .collect();
        let mut paths = Vec::new();
        for id in ids_to_remove {
            if let Some(file) = self.by_id.remove(&id) {
                self.by_path.remove(&file.path);
                paths.push(file.path);
            }
        }
        paths
    }
}

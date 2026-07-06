use crate::ids::MemoryId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type SymbolId = u64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolEntry {
    pub id: SymbolId,
    pub kind: String,
    pub name: String,
    pub signature: String,
    pub file_path: String,
    pub line_start: u32,
    pub line_end: u32,
    pub repo_id: u64,
    pub embedding: Vec<f32>,
    pub description: Option<String>,
    pub memory_id: Option<MemoryId>,
}

/// GC candidate ids by category plus top-directory histogram (see
/// `collect_gc_candidates`).
#[derive(Debug, Default)]
pub struct GcCandidates {
    pub dup: Vec<SymbolId>,
    pub excluded: Vec<SymbolId>,
    pub dead: Vec<SymbolId>,
    pub top_dirs: Vec<(String, usize)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolIndex {
    by_id: HashMap<SymbolId, SymbolEntry>,
    by_name: HashMap<String, Vec<SymbolId>>,
    dedup: HashMap<(String, String, String, u32), SymbolId>,
}

impl SymbolIndex {
    pub fn new() -> Self {
        Self {
            by_id: HashMap::new(),
            by_name: HashMap::new(),
            dedup: HashMap::new(),
        }
    }

    /// Dedup key: (kind, name, file_path). The u32 slot is always 0 — it used
    /// to be line_start, which made every moved function a "new" symbol and
    /// bloated the index with stale-line duplicates. The 4-tuple shape is kept
    /// so bincode snapshot layout is unchanged.
    fn dedup_key(entry: &SymbolEntry) -> (String, String, String, u32) {
        (
            entry.kind.clone(),
            entry.name.clone(),
            entry.file_path.clone(),
            0,
        )
    }

    /// Upsert a symbol. Deduplicates by (kind, name, file_path) — line numbers
    /// update in place. Returns the SymbolId (existing or new).
    pub fn upsert(&mut self, entry: SymbolEntry) -> SymbolId {
        let key = Self::dedup_key(&entry);

        if let Some(&existing_id) = self.dedup.get(&key) {
            // Update existing entry in place.
            self.by_id.insert(
                existing_id,
                SymbolEntry {
                    id: existing_id,
                    ..entry
                },
            );
            return existing_id;
        }

        let id = entry.id;
        self.dedup.insert(key, id);
        self.by_name
            .entry(entry.name.clone())
            .or_insert_with(Vec::new)
            .push(id);
        self.by_id.insert(id, entry);
        id
    }

    pub fn remove(&mut self, id: SymbolId) {
        if let Some(entry) = self.by_id.remove(&id) {
            let key = Self::dedup_key(&entry);
            // Only drop the dedup slot if this id owns it (a duplicate loser
            // being GC'd must not evict the winner's mapping).
            if self.dedup.get(&key) == Some(&id) {
                self.dedup.remove(&key);
            }
            if let Some(ids) = self.by_name.get_mut(&entry.name) {
                ids.retain(|&x| x != id);
                if ids.is_empty() {
                    self.by_name.remove(&entry.name);
                }
            }
        }
    }

    /// Rebuild by_name and dedup from by_id (the source of truth).
    /// Non-destructive: no entries are removed. Heals historical corruption
    /// (stale ids in by_name buckets, line-keyed dedup entries) on snapshot
    /// load. For duplicate (kind, name, file_path) groups the highest id —
    /// the most recently indexed — wins the dedup slot, so subsequent upserts
    /// update the newest copy. Returns the number of duplicate entries found.
    pub fn rebuild_derived(&mut self) -> usize {
        self.by_name.clear();
        self.dedup.clear();
        let mut ids: Vec<SymbolId> = self.by_id.keys().copied().collect();
        ids.sort_unstable();
        let mut duplicates = 0;
        for id in ids {
            let entry = &self.by_id[&id];
            let key = Self::dedup_key(entry);
            self.by_name
                .entry(entry.name.clone())
                .or_insert_with(Vec::new)
                .push(id);
            if self.dedup.insert(key, id).is_some() {
                duplicates += 1;
            }
        }
        duplicates
    }

    pub fn get(&self, id: SymbolId) -> Option<&SymbolEntry> {
        self.by_id.get(&id)
    }

    /// Search by name: exact match first, then prefix match, up to limit results.
    pub fn search_by_name(&self, query: &str, limit: usize) -> Vec<&SymbolEntry> {
        self.search_by_name_scoped(query, limit, None)
    }

    /// Search by name, optionally restricted to entries whose file_path
    /// contains `path_filter` (repo/directory scoping).
    pub fn search_by_name_scoped(
        &self,
        query: &str,
        limit: usize,
        path_filter: Option<&str>,
    ) -> Vec<&SymbolEntry> {
        let in_scope = |e: &SymbolEntry| match path_filter {
            Some(p) => e.file_path.contains(p),
            None => true,
        };
        let mut results: Vec<&SymbolEntry> = Vec::new();

        // Exact matches first.
        if let Some(ids) = self.by_name.get(query) {
            for &id in ids {
                if let Some(e) = self.by_id.get(&id) {
                    if in_scope(e) {
                        results.push(e);
                    }
                }
            }
        }

        // Prefix matches (excluding exact matches already added).
        if results.len() < limit {
            for (name, ids) in &self.by_name {
                if name != query && name.starts_with(query) {
                    for &id in ids {
                        if let Some(e) = self.by_id.get(&id) {
                            if !in_scope(e) {
                                continue;
                            }
                            results.push(e);
                            if results.len() >= limit {
                                return results;
                            }
                        }
                    }
                }
            }
        }

        // Collapse stale-line duplicates: keep the highest-id (newest) entry
        // per (kind, name, file_path). rebuild_derived heals the dedup map, but
        // by_name still lists every historical id for a moved symbol, so the
        // raw scan above can surface many stale lines of the same def. Dedup at
        // read time in first-seen order (exact before prefix).
        let mut winner: HashMap<(String, String, String, u32), usize> = HashMap::new();
        let mut order: Vec<usize> = Vec::new();
        for (i, e) in results.iter().enumerate() {
            let key = Self::dedup_key(e);
            match winner.get(&key) {
                Some(&j) => {
                    if e.id > results[j].id {
                        winner.insert(key, i);
                    }
                }
                None => {
                    winner.insert(key.clone(), i);
                    order.push(i);
                }
            }
        }
        let mut deduped: Vec<&SymbolEntry> = order
            .iter()
            .map(|&slot| results[winner[&Self::dedup_key(results[slot])]])
            .collect();
        deduped.truncate(limit);
        deduped
    }

    /// Semantic search: find k nearest by cosine similarity to query embedding.
    pub fn search_semantic(&self, query: &[f32], k: usize) -> Vec<(SymbolId, f32)> {
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_norm == 0.0 {
            return Vec::new();
        }

        let mut scored: Vec<(SymbolId, f32)> = self
            .by_id
            .values()
            .filter_map(|e| {
                if e.embedding.len() != query.len() {
                    return None;
                }
                let dot: f32 = e
                    .embedding
                    .iter()
                    .zip(query.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let emb_norm: f32 = e.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                if emb_norm == 0.0 {
                    return None;
                }
                Some((e.id, dot / (query_norm * emb_norm)))
            })
            .collect();

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Get all symbols in a file path.
    pub fn by_file(&self, file_path: &str) -> Vec<&SymbolEntry> {
        self.by_id
            .values()
            .filter(|e| e.file_path == file_path)
            .collect()
    }

    pub fn count(&self) -> usize {
        self.by_id.len()
    }

    /// Return the maximum symbol ID present, or None if empty.
    pub fn max_id(&self) -> Option<u64> {
        self.by_id.keys().copied().max()
    }

    /// Get mutable reference to a symbol by ID.
    pub fn get_mut(&mut self, id: SymbolId) -> Option<&mut SymbolEntry> {
        self.by_id.get_mut(&id)
    }

    /// Collect GC candidates without removing anything. Categories:
    /// - dup: entries that lost the (kind, name, file_path) dedup slot to a
    ///   higher id (stale-line copies from the old line-keyed dedup era)
    /// - excluded: file_path contains one of `path_excludes` (plugin caches,
    ///   build deps, …)
    /// - dead: file no longer exists on disk (only when `check_fs`)
    /// Each id is counted once, in priority order excluded > dead > dup.
    /// Also returns the top directories by symbol count for dry-run triage.
    pub fn collect_gc_candidates(
        &self,
        check_fs: bool,
        path_excludes: &[String],
    ) -> GcCandidates {
        // Winner per dedup key = highest id (matches rebuild_derived).
        let mut winners: HashMap<(String, String, String, u32), SymbolId> = HashMap::new();
        for (&id, e) in &self.by_id {
            let key = Self::dedup_key(e);
            let w = winners.entry(key).or_insert(id);
            if id > *w {
                *w = id;
            }
        }

        let mut fs_cache: HashMap<&str, bool> = HashMap::new();
        let mut out = GcCandidates::default();
        let mut dir_counts: HashMap<String, usize> = HashMap::new();

        for (&id, e) in &self.by_id {
            let dir = std::path::Path::new(&e.file_path)
                .parent()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default();
            *dir_counts.entry(dir).or_insert(0) += 1;

            if path_excludes.iter().any(|p| e.file_path.contains(p.as_str())) {
                out.excluded.push(id);
                continue;
            }
            if check_fs {
                let alive = *fs_cache
                    .entry(e.file_path.as_str())
                    .or_insert_with(|| std::path::Path::new(&e.file_path).exists());
                if !alive {
                    out.dead.push(id);
                    continue;
                }
            }
            if winners.get(&Self::dedup_key(e)) != Some(&id) {
                out.dup.push(id);
            }
        }

        let mut dirs: Vec<(String, usize)> = dir_counts.into_iter().collect();
        dirs.sort_by(|a, b| b.1.cmp(&a.1));
        dirs.truncate(15);
        out.top_dirs = dirs;
        out
    }

    /// Remove all symbols whose file_path matches any of the given paths.
    /// Returns the IDs that were removed (caller must clean up call_graph).
    pub fn remove_by_file_paths(&mut self, paths: &[String]) -> Vec<SymbolId> {
        if paths.is_empty() {
            return Vec::new();
        }
        let path_set: std::collections::HashSet<&str> = paths.iter().map(|s| s.as_str()).collect();
        let ids_to_remove: Vec<SymbolId> = self
            .by_id
            .iter()
            .filter(|(_, e)| path_set.contains(e.file_path.as_str()))
            .map(|(&id, _)| id)
            .collect();
        for id in &ids_to_remove {
            self.remove(*id);
        }
        ids_to_remove
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: SymbolId, name: &str, path: &str, line: u32) -> SymbolEntry {
        SymbolEntry {
            id,
            kind: "function".into(),
            name: name.into(),
            signature: format!("{}()", name),
            file_path: path.into(),
            line_start: line,
            line_end: line + 10,
            repo_id: 0,
            embedding: vec![],
            description: None,
            memory_id: None,
        }
    }

    #[test]
    fn upsert_dedups_across_line_moves() {
        let mut idx = SymbolIndex::new();
        let id1 = idx.upsert(entry(1, "foo", "/a.rs", 10));
        // Same symbol moved to line 42 — must UPDATE, not insert.
        let id2 = idx.upsert(entry(2, "foo", "/a.rs", 42));
        assert_eq!(id1, id2);
        assert_eq!(idx.count(), 1);
        assert_eq!(idx.get(id1).unwrap().line_start, 42);
    }

    #[test]
    fn rebuild_derived_heals_corrupt_by_name() {
        let mut idx = SymbolIndex::new();
        idx.upsert(entry(1, "foo", "/a.rs", 10));
        idx.upsert(entry(2, "bar", "/b.rs", 5));
        // Simulate historical duplicates: force a second "foo" copy into by_id
        // (as the old line-keyed dedup allowed).
        idx.by_id.insert(3, entry(3, "foo", "/a.rs", 99));
        idx.by_name.get_mut("bar").unwrap().push(3); // corrupt bucket
        let dups = idx.rebuild_derived();
        assert_eq!(dups, 1);
        // "bar" bucket only holds bar's id again.
        let bar_hits = idx.search_by_name("bar", 10);
        assert!(bar_hits.iter().all(|e| e.name == "bar"));
        // Newest "foo" (id 3) owns the dedup slot: next upsert updates id 3.
        let id = idx.upsert(entry(4, "foo", "/a.rs", 120));
        assert_eq!(id, 3);
    }

    #[test]
    fn search_dedups_stale_line_copies() {
        // by_name carries many historical ids for one moved def (the 135k-index
        // failure mode). Query-time dedup must collapse them to the newest.
        let mut idx = SymbolIndex::new();
        for (id, line) in [(1u64, 10u32), (2, 42), (3, 99), (4, 120)] {
            idx.by_id.insert(id, entry(id, "foo", "/a.rs", line));
            idx.by_name.entry("foo".into()).or_default().push(id);
        }
        // A genuinely distinct def in another file must survive.
        idx.by_id.insert(5, entry(5, "foo", "/vendor/a.rs", 7));
        idx.by_name.entry("foo".into()).or_default().push(5);
        let hits = idx.search_by_name("foo", 50);
        assert_eq!(hits.len(), 2, "one per (kind,name,file)");
        let a = hits.iter().find(|e| e.file_path == "/a.rs").unwrap();
        assert_eq!(a.id, 4, "newest id wins");
        assert_eq!(a.line_start, 120);
        assert!(hits.iter().any(|e| e.file_path == "/vendor/a.rs"));
    }

    #[test]
    fn gc_candidates_categorize() {
        let mut idx = SymbolIndex::new();
        idx.by_id.insert(1, entry(1, "foo", "/gone/x.rs", 1));
        idx.by_id.insert(2, entry(2, "foo", "/gone/x.rs", 9)); // dup of 1 (loser: lower id)
        idx.by_id.insert(3, entry(3, "baz", "/repo/plugins/cache/y.rs", 1));
        idx.rebuild_derived();
        let cand = idx.collect_gc_candidates(true, &["plugins/cache".into()]);
        assert_eq!(cand.excluded, vec![3]);
        assert_eq!(cand.dead.len(), 2); // both /gone entries: dead beats dup
        assert!(cand.dup.is_empty());
        let no_fs = idx.collect_gc_candidates(false, &["plugins/cache".into()]);
        assert_eq!(no_fs.dup, vec![1]); // without fs check, loser counted as dup
    }

    #[test]
    fn scoped_search_filters_paths() {
        let mut idx = SymbolIndex::new();
        idx.upsert(entry(1, "foo", "/repo_a/x.rs", 1));
        idx.upsert(entry(2, "foo", "/repo_b/y.rs", 1));
        let hits = idx.search_by_name_scoped("foo", 10, Some("/repo_a/"));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].file_path, "/repo_a/x.rs");
    }
}

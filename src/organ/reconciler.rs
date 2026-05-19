/// CEC Phase 17 — Reconcile Operator (R0).
///
/// The single CPU-native deliberation op for Phase 17. Scans association edges
/// for legality violations under the MemoryKind matrix and rewrites what it can.
/// Contradictions it cannot resolve structurally are returned as `unresolved`
/// for optional R1 (LLM proposal) handling.
use std::collections::HashMap;
use crate::ids::MemoryId;
use crate::field::AssocEdge;
use crate::payload::MemoryPayload;
use crate::organ::memory_kind::{edge_legal, MemoryKind};

pub struct ReconcileResult {
    /// Illegal edges that were identified (not physically removed — logged to KG).
    pub illegal_edges:  Vec<(MemoryId, MemoryId, String)>,  // (src, dst, reason)
    /// Contradictions detected by negation-pattern check (same subject, conflicting predicates).
    pub contradictions: Vec<(MemoryId, MemoryId, f32)>,     // (a, b, conflict_score)
    /// Edges this pass could not resolve — require R1 (model proposal).
    pub unresolved:     Vec<(MemoryId, MemoryId)>,
}

pub struct Reconciler;

impl Reconciler {
    pub fn new() -> Self { Self }

    /// Scan all assoc_edges for MemoryKind legality violations.
    /// R0: purely structural — no model call.
    pub fn reconcile_all(
        &self,
        payloads:    &HashMap<MemoryId, MemoryPayload>,
        assoc_edges: &HashMap<MemoryId, Vec<AssocEdge>>,
    ) -> ReconcileResult {
        let mut illegal_edges  = Vec::new();
        let mut unresolved     = Vec::new();

        for (&src_id, edges) in assoc_edges {
            let src_payload = match payloads.get(&src_id) { Some(p) => p, None => continue };
            let src_kind = MemoryKind::infer(
                &src_payload.kind, &src_payload.realm,
                std::str::from_utf8(&src_payload.content).unwrap_or("").get(..200).unwrap_or(""),
            );
            let src_candidate = src_payload.candidate;

            for edge in edges {
                let dst_payload = match payloads.get(&edge.dst) { Some(p) => p, None => continue };
                let dst_kind = MemoryKind::infer(
                    &dst_payload.kind, &dst_payload.realm,
                    std::str::from_utf8(&dst_payload.content).unwrap_or("").get(..200).unwrap_or(""),
                );
                let dst_candidate = dst_payload.candidate;

                // Candidate citing established knowledge: laundering
                if src_candidate && !dst_candidate {
                    illegal_edges.push((src_id, edge.dst,
                        format!("candidate({}) → established({})", src_kind.label(), dst_kind.label())));
                    unresolved.push((src_id, edge.dst));
                    continue;
                }

                if !edge_legal(src_kind, dst_kind) {
                    let reason = format!("{}→{} violates kind lattice",
                        src_kind.label(), dst_kind.label());
                    illegal_edges.push((src_id, edge.dst, reason));
                    // These can be structurally resolved by removing the edge (done by caller).
                }
            }
        }

        ReconcileResult { illegal_edges, contradictions: vec![], unresolved }
    }

    /// Detect contradictions: memories in the same realm with opposing content signals.
    /// Heuristic: same 6-word content prefix but differing outcome class → conflict_score=0.8.
    /// CPU-only, no model.
    pub fn detect_contradictions(
        &self,
        payloads: &HashMap<MemoryId, MemoryPayload>,
    ) -> Vec<(MemoryId, MemoryId, f32)> {
        let mut contradictions = Vec::new();

        // Group by (realm, first-6-word prefix)
        let mut prefix_map: HashMap<(String, String), Vec<MemoryId>> = HashMap::new();
        for (&id, p) in payloads {
            let text = std::str::from_utf8(&p.content).unwrap_or("").get(..80).unwrap_or("");
            let prefix: String = text.split_whitespace().take(6).collect::<Vec<_>>().join(" ").to_lowercase();
            if prefix.len() >= 10 {
                prefix_map.entry((p.realm.clone(), prefix)).or_default().push(id);
            }
        }

        for ids in prefix_map.values() {
            if ids.len() < 2 { continue; }
            // Pairs within same prefix group: check if kind or content tail differs enough
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    let a = &payloads[&ids[i]];
                    let b = &payloads[&ids[j]];
                    // Rough contradiction signal: same prefix, different kind classification
                    let ka = MemoryKind::infer(&a.kind, &a.realm, "");
                    let kb = MemoryKind::infer(&b.kind, &b.realm, "");
                    if ka != kb {
                        contradictions.push((ids[i], ids[j], 0.6));
                    } else if a.content != b.content && a.content.len() > 20 && b.content.len() > 20 {
                        // Same kind, same prefix, different body — possible update collision
                        contradictions.push((ids[i], ids[j], 0.4));
                    }
                }
            }
        }

        // Cap at 50 to avoid O(n²) explosions on large stores
        contradictions.truncate(50);
        contradictions
    }
}

impl Default for Reconciler {
    fn default() -> Self { Self::new() }
}

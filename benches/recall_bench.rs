//! Manual benchmark for chitta-field recall performance.
//!
//! Run with:
//!   ./build.sh run --bin bench --release
//!
//! Tests recall_semantic, recall_keyword, recall_temporal, and hybrid
//! (semantic + keyword merge) at corpus sizes 100, 1_000, 5_000, 10_000.
//!
//! Each method is exercised with 100 queries and the mean latency is reported.
//! A bottleneck analysis section explains the dominant cost at each scale.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use chitta_field::field::ChittaField;
use chitta_field::ops::EMBED_DIM;

fn tmp_field(tag: &str) -> (ChittaField, PathBuf) {
    let base = std::env::temp_dir().join(format!("chitta-bench-{tag}"));
    let data = base.join("data");
    // Start fresh
    let _ = std::fs::remove_dir_all(&base);
    let field = ChittaField::open(data).expect("open");
    (field, base)
}

/// A deterministic unit embedding for index i.
/// Two non-zero dimensions ensure varied cosine similarity across queries.
fn make_embedding(i: usize) -> Vec<f32> {
    let mut e = vec![0.0f32; EMBED_DIM];
    e[i % EMBED_DIM] = 1.0;
    e[(i.wrapping_mul(7)) % EMBED_DIM] += 0.5;
    // normalise
    let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut e { *x /= norm; }
    }
    e
}

struct BenchResult {
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

fn bench<F: FnMut() -> ()>(mut f: F, iters: usize) -> BenchResult {
    // Warmup
    for _ in 0..5 { f(); }

    let mut durations: Vec<Duration> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        durations.push(t.elapsed());
    }

    let total: Duration = durations.iter().sum();
    let mean_ms = total.as_secs_f64() * 1000.0 / iters as f64;
    let min_ms  = durations.iter().map(|d| d.as_secs_f64() * 1000.0).fold(f64::INFINITY, f64::min);
    let max_ms  = durations.iter().map(|d| d.as_secs_f64() * 1000.0).fold(0.0f64, f64::max);

    BenchResult { mean_ms, min_ms, max_ms }
}

fn print_result(label: &str, r: &BenchResult, iters: usize) {
    println!(
        "    {:<22} mean={:6.2}ms  min={:5.2}ms  max={:6.2}ms  ({} iters)",
        label, r.mean_ms, r.min_ms, r.max_ms, iters
    );
}

fn main() {
    let corpus_sizes: &[usize] = &[100, 1_000, 5_000, 10_000];
    let query_iters = 100usize;

    println!("chitta-field recall benchmark");
    println!("EMBED_DIM={EMBED_DIM}, query_iters={query_iters}");
    println!();

    for &n in corpus_sizes {
        println!("=== Corpus: {n} memories ===");

        let (field, base) = tmp_field(&n.to_string());

        // --- Insert phase ---
        let t_insert = Instant::now();
        for i in 0..n {
            let emb = make_embedding(i);
            let topic = i % 20;
            let project = i % 5;
            let content = format!(
                "memory {i} discusses topic {topic} in project {project} \
                 with context about system design and architecture patterns"
            );
            field.put_memory(
                "wisdom", "bench",
                content.as_bytes(), &emb,
                0.9, 0.001, i as i64 * 1000,
                vec![], None, None,
            ).unwrap();
        }
        let insert_ms = t_insert.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  Insert {n}: {insert_ms:.0}ms  ({:.0} rec/s)",
            n as f64 / (insert_ms / 1000.0).max(1e-6)
        );

        // Query embedding: aligned with topic 0 memories
        let query_emb = make_embedding(0);

        // --- recall_semantic ---
        let r = bench(|| {
            let hits = field.recall_semantic(&query_emb, 10, Some("bench")).unwrap();
            assert!(!hits.is_empty());
        }, query_iters);
        print_result("recall_semantic(k=10)", &r, query_iters);

        // --- recall_semantic without realm filter (no allowed-set build cost) ---
        let r = bench(|| {
            let hits = field.recall_semantic(&query_emb, 10, None).unwrap();
            assert!(!hits.is_empty());
        }, query_iters);
        print_result("recall_semantic(no realm)", &r, query_iters);

        // --- recall_keyword ---
        let r = bench(|| {
            let hits = field.recall_keyword("topic project memory system", 10).unwrap();
            let _ = hits;
        }, query_iters);
        print_result("recall_keyword(k=10)", &r, query_iters);

        // --- recall_temporal ---
        let mid_ms = (n as i64 / 2) * 1000;
        let r = bench(|| {
            let hits = field.recall_temporal(0, mid_ms, Some("bench"), 10).unwrap();
            let _ = hits;
        }, query_iters);
        print_result("recall_temporal(k=10)", &r, query_iters);

        // --- hybrid: semantic + keyword, deduplicated union ---
        let r = bench(|| {
            let sem = field.recall_semantic(&query_emb, 20, Some("bench")).unwrap();
            let kw  = field.recall_keyword("topic project design", 20).unwrap();
            let mut seen = HashSet::new();
            let combined: Vec<_> = sem.into_iter()
                .chain(kw)
                .filter(|h| seen.insert(h.memory_id))
                .take(10)
                .collect();
            assert!(!combined.is_empty());
        }, query_iters);
        print_result("hybrid(sem+kw, k=10)", &r, query_iters);

        // --- expand_associations (no edges → O(1) but measures overhead) ---
        let r = bench(|| {
            let hits = field.expand_associations(&[1], 2, 10).unwrap();
            let _ = hits;
        }, query_iters);
        print_result("expand_associations", &r, query_iters);

        println!();

        // Cleanup temp dir
        let _ = std::fs::remove_dir_all(&base);
    }

    // --- Scaling summary ---
    println!("=== Scaling analysis ===");
    println!();
    println!("recall_semantic:");
    println!("  Cost: O(n × {EMBED_DIM}) dot products + O(n log n) sort.");
    println!("  With realm filter: additionally O(n) HashSet build over payloads.");
    println!("  Dominant cost at >10k memories. Replace SemanticIndex with HNSW");
    println!("  (instant-distance crate) for O(log n) approximate recall.");
    println!();
    println!("recall_keyword (BM25):");
    println!("  Cost: O(|postings| per query term) — sublinear in n.");
    println!("  Scales well; bottleneck is postings list iteration, not corpus size.");
    println!();
    println!("recall_temporal:");
    println!("  Cost: O(log n) BTreeMap range scan + O(result) filter.");
    println!("  Essentially free; no scaling concern up to millions of memories.");
    println!();
    println!("hybrid (sem+kw):");
    println!("  Cost: recall_semantic + recall_keyword + O(k) dedup.");
    println!("  Dominated by semantic scan at scale.");
    println!();
    println!("expand_associations:");
    println!("  Cost: O(hops × fanout) BFS over assoc_edges HashMap.");
    println!("  Fast when edge density is low; fanout capped at 16 per hop.");
}

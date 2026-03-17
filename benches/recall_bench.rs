//! Manual large-scale benchmark for chitta-field.
//!
//! Examples:
//!   ./build.sh run --bin bench --release
//!   ./build.sh run --bin bench --release -- --sizes 100000,1000000 --queries 50 --flush

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use chitta_field::field::ChittaField;
use chitta_field::ops::EMBED_DIM;

struct Config {
    sizes: Vec<usize>,
    query_iters: usize,
    top_k: usize,
    flush_after_queries: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sizes: vec![10_000, 100_000],
            query_iters: 50,
            top_k: 10,
            flush_after_queries: true,
        }
    }
}

impl Config {
    fn from_args() -> Self {
        let mut cfg = Self::default();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--sizes" => {
                    if let Some(v) = args.next() {
                        cfg.sizes = parse_sizes(&v);
                    }
                }
                "--queries" => {
                    if let Some(v) = args.next() {
                        cfg.query_iters = v.parse().unwrap_or(cfg.query_iters);
                    }
                }
                "--top-k" => {
                    if let Some(v) = args.next() {
                        cfg.top_k = v.parse().unwrap_or(cfg.top_k);
                    }
                }
                "--flush" => cfg.flush_after_queries = true,
                "--no-flush" => cfg.flush_after_queries = false,
                _ => {}
            }
        }
        cfg
    }
}

fn parse_sizes(v: &str) -> Vec<usize> {
    let mut sizes = Vec::new();
    for raw in v.split(',') {
        if let Ok(n) = raw.trim().parse() {
            sizes.push(n);
        }
    }
    if sizes.is_empty() {
        Config::default().sizes
    } else {
        sizes
    }
}

fn tmp_field(tag: &str) -> (ChittaField, PathBuf) {
    let base = std::env::temp_dir().join(format!("chitta-bench-{tag}"));
    let data = base.join("data");
    let _ = std::fs::remove_dir_all(&base);
    let field = ChittaField::open(data).expect("open");
    (field, base)
}

fn make_embedding(i: usize) -> Vec<f32> {
    let mut e = vec![0.0f32; EMBED_DIM];
    e[i % EMBED_DIM] = 1.0;
    e[(i.wrapping_mul(7)) % EMBED_DIM] += 0.5;
    e[(i.wrapping_mul(13)) % EMBED_DIM] += 0.25;
    let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut e {
            *x /= norm;
        }
    }
    e
}

struct BenchResult {
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

fn bench<F: FnMut()>(mut f: F, iters: usize) -> BenchResult {
    for _ in 0..5.min(iters) {
        f();
    }

    let mut durations: Vec<Duration> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        durations.push(t.elapsed());
    }

    let total: Duration = durations.iter().sum();
    let mean_ms = total.as_secs_f64() * 1000.0 / iters as f64;
    let min_ms = durations
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .fold(f64::INFINITY, f64::min);
    let max_ms = durations
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .fold(0.0f64, f64::max);

    BenchResult {
        mean_ms,
        min_ms,
        max_ms,
    }
}

fn print_result(label: &str, r: &BenchResult, iters: usize) {
    println!(
        "    {:<26} mean={:8.2}ms  min={:7.2}ms  max={:8.2}ms  ({} iters)",
        label, r.mean_ms, r.min_ms, r.max_ms, iters
    );
}

fn insert_corpus(field: &ChittaField, n: usize) {
    let t0 = Instant::now();
    for i in 0..n {
        let emb = make_embedding(i);
        let topic = i % 256;
        let project = i % 32;
        let shard = i % 16;
        let realm = if i % 2 == 0 { "bench-a" } else { "bench-b" };
        let content = format!(
            "memory{i} topic{topic} project{project} shard{shard} architecture retrieval benchmark synthetic corpus"
        );
        field
            .put_memory(
                "wisdom",
                realm,
                content.as_bytes(),
                &emb,
                0.9,
                0.001,
                i as i64 * 1000,
                vec![],
                None,
                None,
            )
            .unwrap();
        if (i + 1) % 100_000 == 0 {
            let elapsed = t0.elapsed().as_secs_f64().max(1e-6);
            println!(
                "    inserted {:>8} / {:>8} ({:>8.0} rec/s)",
                i + 1,
                n,
                (i + 1) as f64 / elapsed
            );
        }
    }
}

fn main() {
    let cfg = Config::from_args();

    println!("chitta-field scalable benchmark");
    println!(
        "EMBED_DIM={} sizes={:?} query_iters={} top_k={} flush_after_queries={}",
        EMBED_DIM, cfg.sizes, cfg.query_iters, cfg.top_k, cfg.flush_after_queries
    );
    println!();

    for &n in &cfg.sizes {
        println!("=== Corpus: {n} memories ===");
        let (field, base) = tmp_field(&n.to_string());

        let t_insert = Instant::now();
        insert_corpus(&field, n);
        let insert_ms = t_insert.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  Insert {n}: {:.0}ms ({:.0} rec/s)",
            insert_ms,
            n as f64 / (insert_ms / 1000.0).max(1e-6)
        );

        let query_emb = make_embedding(0);
        let keyword_query = "topic0 project0 shard0";

        let r = bench(
            || {
                let hits = field
                    .recall_semantic(&query_emb, cfg.top_k, Some("bench-a"))
                    .unwrap();
                assert!(!hits.is_empty());
            },
            cfg.query_iters,
        );
        print_result("recall_semantic(realm)", &r, cfg.query_iters);

        let r = bench(
            || {
                let hits = field.recall_semantic(&query_emb, cfg.top_k, None).unwrap();
                assert!(!hits.is_empty());
            },
            cfg.query_iters,
        );
        print_result("recall_semantic(global)", &r, cfg.query_iters);

        let r = bench(
            || {
                let hits = field.recall_keyword(keyword_query, cfg.top_k).unwrap();
                assert!(!hits.is_empty());
            },
            cfg.query_iters,
        );
        print_result("recall_keyword", &r, cfg.query_iters);

        let mid_ms = (n as i64 / 2) * 1000;
        let r = bench(
            || {
                let hits = field
                    .recall_temporal(0, mid_ms, Some("bench-a"), cfg.top_k)
                    .unwrap();
                let _ = hits;
            },
            cfg.query_iters,
        );
        print_result("recall_temporal", &r, cfg.query_iters);

        let r = bench(
            || {
                let sem = field
                    .recall_semantic(&query_emb, cfg.top_k * 2, Some("bench-a"))
                    .unwrap();
                let kw = field.recall_keyword(keyword_query, cfg.top_k * 2).unwrap();
                let mut seen = HashSet::new();
                let combined: Vec<_> = sem
                    .into_iter()
                    .chain(kw)
                    .filter(|h| seen.insert(h.memory_id))
                    .take(cfg.top_k)
                    .collect();
                assert!(!combined.is_empty());
            },
            cfg.query_iters,
        );
        print_result("hybrid(sem+kw)", &r, cfg.query_iters);

        if cfg.flush_after_queries {
            let r = bench(
                || {
                    field.flush().unwrap();
                },
                10,
            );
            print_result("flush(deferred effects)", &r, 10);
        }

        println!("  memory_count={}", field.memory_count());
        println!();

        let _ = std::fs::remove_dir_all(&base);
    }

    println!("Notes:");
    println!("  semantic recall uses the ANN index directly");
    println!("  ANN candidate generation uses LSH first, then coarse centroid buckets");
    println!("  recall-side reconsolidation is deferred until flush/snapshot");
    println!("  use --sizes 1000000 to drive a 1M synthetic run");
}

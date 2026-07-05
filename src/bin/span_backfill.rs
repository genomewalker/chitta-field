//! Standalone Span Lane backfill: extract verbatim atoms from every transcript
//! and write the `spans.bin` sidecar. Runs offline (no daemon) so it can be
//! measured; the daemon loads the same sidecar at startup.
//!
//! Usage: span_backfill [PROJECTS_DIR] [DATA_DIR]
//!   PROJECTS_DIR default ~/.claude/projects
//!   DATA_DIR     default ~/.claude/mind   (where spans.bin is written)

use chitta_field::organ::span_store::SpanStore;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let home = std::env::var("HOME").unwrap_or_default();
    let args: Vec<String> = std::env::args().collect();
    let projects = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| format!("{home}/.claude/projects"));
    let data_dir = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| format!("{home}/.claude/mind"));

    let projects = PathBuf::from(projects);
    let data_dir = PathBuf::from(data_dir);
    eprintln!("[span_backfill] projects={projects:?} data_dir={data_dir:?}");

    if args.get(1).map(|a| a == "--query").unwrap_or(false) {
        // span_backfill --query <data_dir> <realm|-> <query...>
        let realm = args.get(3).map(|s| s.as_str()).filter(|s| *s != "-");
        let q = args.get(4..).map(|s| s.join(" ")).unwrap_or_default();
        let mut store = SpanStore::load(&data_dir);
        let names = ["PATH", "URL", "UUID", "HEX", "ISSUE", "FILELINE", "BASH", "ERROR"];
        let hits = store.query(&q, realm, 10);
        println!("query={q:?} realm={realm:?} -> {} hits", hits.len());
        for h in hits {
            println!(
                "  [{:8}] score={:.3} count={} realm={} :: {}",
                names.get(h.class as usize).unwrap_or(&"?"),
                h.score,
                h.count,
                h.realm,
                h.text
            );
        }
        return;
    }

    if args.get(1).map(|a| a == "--analyze").unwrap_or(false) {
        let store = SpanStore::load(&data_dir);
        let names = ["PATH", "URL", "UUID", "HEX", "ISSUE", "FILELINE", "BASH", "ERROR"];
        println!("class     total    singletons  text_MB");
        for (c, total, singles, bytes) in store.histogram() {
            println!(
                "{:9} {:>8} {:>10} {:>8.1}",
                names.get(c as usize).unwrap_or(&"?"),
                total,
                singles,
                bytes as f64 / 1e6
            );
        }
        println!("unique total: {}  disk: {:.1} MB", store.len(), store.on_disk_bytes() as f64 / 1e6);
        return;
    }

    let t0 = Instant::now();
    let mut store = SpanStore::load(&data_dir);
    let before = store.len();
    let stats = store.ingest_dir(&projects);
    let wall = t0.elapsed();

    let disk = store.on_disk_bytes();
    println!("── span backfill complete ──");
    println!("wall:            {:.1}s", wall.as_secs_f64());
    println!("lines parsed:    {}", stats.lines);
    println!("raw atoms:       {}", stats.raw_spans);
    println!("new unique:      {}", stats.new_spans);
    println!("unique total:    {} (was {})", store.len(), before);
    println!("redacted/dropped:{}", stats.redacted);
    println!("skipped inject:  {}", stats.skipped_injection);
    println!("spans.bin bytes: {} ({:.1} MB)", disk, disk as f64 / 1e6);
}

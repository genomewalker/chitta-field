//! Encode all unindexed memories into the cortical sparse index.
//!
//! Usage:
//!   ./build.sh run --bin encode --release -- --field-dir ~/.claude/mind/chitta-field [--encode-pq] [--save-snapshot]

use std::path::PathBuf;
use std::time::Instant;
use chitta_field::field::ChittaField;

fn expand_home(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        let home = std::env::var("HOME").unwrap_or_default();
        PathBuf::from(home).join(rest)
    } else {
        PathBuf::from(p)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let field_dir = args.windows(2)
        .find(|w| w[0] == "--field-dir")
        .map(|w| expand_home(&w[1]))
        .unwrap_or_else(|| {
            eprintln!("Usage: encode --field-dir <path> [--encode-pq] [--save-snapshot]");
            std::process::exit(1);
        });
    let save_snapshot = args.iter().any(|a| a == "--save-snapshot");
    let save_full_snapshot = args.iter().any(|a| a == "--save-full-snapshot");
    let encode_pq = args.iter().any(|a| a == "--encode-pq");

    eprintln!("Opening chitta-field at {:?}", field_dir);
    let field = ChittaField::open(field_dir).expect("failed to open chitta-field");

    let total = field.memory_count();
    let before = field.cortical_count();
    eprintln!("Total memories: {}  Already indexed: {}  Unindexed: {}", total, before, total.saturating_sub(before));

    let t0 = Instant::now();
    let encoded = field.encode_all_unindexed().expect("encode_all failed");
    let elapsed = t0.elapsed();

    let after = field.cortical_count();
    let protos = field.prototype_count();
    eprintln!("Encoded {} memories in {:.1}s  ({:.0}/s)  Cortical index: {}  Prototypes: {}",
        encoded, elapsed.as_secs_f64(), encoded as f64 / elapsed.as_secs_f64().max(0.001), after, protos);

    if encode_pq {
        eprintln!("Running residual PQ encoding...");
        let t1 = Instant::now();
        match field.encode_all_pq() {
            Ok(pq_count) => {
                let pq_elapsed = t1.elapsed();
                let total_pq = field.pq_count();
                eprintln!("PQ encoded {} memories in {:.1}s  Total PQ indexed: {}",
                    pq_count, pq_elapsed.as_secs_f64(), total_pq);
            }
            Err(e) => {
                eprintln!("PQ encoding failed: {}  (need at least 256 encoded memories)", e);
            }
        }
    }

    if save_snapshot {
        eprintln!("Saving cortical snapshot...");
        field.save_snapshot().expect("save_snapshot failed");
        eprintln!("Cortical snapshot saved.");
    }

    if save_full_snapshot {
        eprintln!("Saving full state snapshot...");
        field.save_full_snapshot().expect("save_full_snapshot failed");
        eprintln!("Full snapshot saved.");
    }
}

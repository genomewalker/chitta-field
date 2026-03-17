//! Ingest JSONL from migrate_from_duckdb.py into a ChittaField store.
//!
//! Usage:
//!   ./build.sh run --bin migrate --release -- \
//!       --memories /tmp/chitta_migration/memories.jsonl \
//!       --triplets /tmp/chitta_migration/triplets.jsonl \
//!       --field-dir ~/.claude/mind/chitta-field
//!
//! Memories without embeddings (embedding: null in JSON) receive a zero-vector
//! placeholder.  Re-embed them afterwards with chitta-field's ONNX pipeline.
//! Pinned memories get a high initial confidence (1.0) and decay_rate = 0.0.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

use chitta_field::field::ChittaField;
use chitta_field::ops::EMBED_DIM;

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

    let mut memories_path: Option<PathBuf> = None;
    let mut triplets_path: Option<PathBuf> = None;
    let mut field_dir = expand_home("~/.claude/mind/chitta-field");
    let mut batch_report = 500usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--memories" => {
                i += 1;
                memories_path = Some(expand_home(&args[i]));
            }
            "--triplets" => {
                i += 1;
                triplets_path = Some(expand_home(&args[i]));
            }
            "--field-dir" => {
                i += 1;
                field_dir = expand_home(&args[i]);
            }
            "--lock-dir" => {
                i += 1;
            } // ignored — Upanishads model needs no locks
            "--batch" => {
                i += 1;
                batch_report = args[i].parse().unwrap_or(500);
            }
            _ => {}
        }
        i += 1;
    }

    let memories_path = memories_path.expect("--memories <path> required");

    println!("chitta-field migration");
    println!("  field-dir : {}", field_dir.display());
    println!("  memories  : {}", memories_path.display());
    if let Some(ref tp) = triplets_path {
        println!("  triplets  : {}", tp.display());
    }
    println!();

    let field = ChittaField::open(field_dir).expect("Failed to open ChittaField");

    // --- Ingest memories ---
    let t0 = Instant::now();
    let file = File::open(&memories_path)
        .unwrap_or_else(|e| panic!("Cannot open {}: {}", memories_path.display(), e));
    let reader = BufReader::new(file);

    let mut total = 0usize;
    let mut ok = 0usize;
    let mut skipped = 0usize;
    // Track original_id → new MemoryId for triplet source linking (if needed later)
    let mut _id_map: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();

    let zero_embedding = vec![0.0f32; EMBED_DIM];

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("IO error: {e}");
                continue;
            }
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        total += 1;

        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                skipped += 1;
                if skipped <= 5 {
                    eprintln!("  Parse error line {total}: {e}");
                }
                continue;
            }
        };

        let content = v["content"].as_str().unwrap_or("").as_bytes().to_vec();
        let kind = v["kind"].as_str().unwrap_or("unknown");
        let realm = v["realm"].as_str().unwrap_or("brahman");
        let confidence = v["confidence"].as_f64().unwrap_or(1.0) as f32;
        let decay_rate = v["decay_rate"].as_f64().unwrap_or(0.001) as f32;
        let created_ms = v["created_at_ms"].as_i64().unwrap_or(0);
        let pinned = v["pinned"].as_bool().unwrap_or(false);
        let original_id = v["original_id"].as_u64().unwrap_or(0);

        // Use embedding from JSON if present and correct length; otherwise zero.
        let embedding: Vec<f32> = match v["embedding"].as_array() {
            Some(arr) if arr.len() == EMBED_DIM => arr
                .iter()
                .filter_map(|x| x.as_f64().map(|f| f as f32))
                .collect(),
            _ => zero_embedding.clone(),
        };

        // Pinned memories: force confidence=1.0, decay_rate=0.0
        let (confidence, decay_rate) = if pinned {
            (1.0f32, 0.0f32)
        } else {
            (confidence, decay_rate)
        };

        match field.put_memory(
            kind,
            realm,
            &content,
            &embedding,
            confidence,
            decay_rate,
            created_ms,
            vec![],
            None,
            None,
        ) {
            Ok((new_id, _)) => {
                _id_map.insert(original_id, new_id.into());
                // Restore pin state if needed
                if pinned {
                    let _ = field.update_state(new_id, None, None, None, false, Some(true));
                }
                ok += 1;
            }
            Err(e) => {
                skipped += 1;
                if skipped <= 5 {
                    eprintln!("  Error on record {total}: {e}");
                }
            }
        }

        if total % batch_report == 0 {
            let rate = total as f64 / t0.elapsed().as_secs_f64();
            println!(
                "  memories: {total} processed ({ok} ok, {skipped} skipped) — {rate:.0} rec/s"
            );
        }
    }

    let mem_elapsed = t0.elapsed().as_secs_f64();
    println!(
        "\nMemories complete: {ok}/{total} ingested, {skipped} skipped ({mem_elapsed:.1}s, {:.0} rec/s)",
        ok as f64 / mem_elapsed.max(0.001)
    );

    // --- Ingest triplets ---
    if let Some(tp) = triplets_path {
        let t1 = Instant::now();
        let file =
            File::open(&tp).unwrap_or_else(|e| panic!("Cannot open {}: {}", tp.display(), e));
        let reader = BufReader::new(file);

        let mut trip_total = 0usize;
        let mut trip_ok = 0usize;
        let mut trip_skipped = 0usize;
        let mut seen: HashSet<(String, String, String)> = HashSet::new();

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            trip_total += 1;

            let v: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => {
                    trip_skipped += 1;
                    continue;
                }
            };

            let subject = v["subject"].as_str().unwrap_or("").to_string();
            let predicate = v["predicate"].as_str().unwrap_or("").to_string();
            let object = v["object"].as_str().unwrap_or("").to_string();
            let weight = v["weight"].as_f64().unwrap_or(1.0) as f32;
            let source = v["source_file"].as_str().map(|s| s.to_string());

            if subject.is_empty() || predicate.is_empty() || object.is_empty() {
                trip_skipped += 1;
                continue;
            }

            // Deduplicate: skip exact (subject, predicate, object) duplicates
            let key = (subject.clone(), predicate.clone(), object.clone());
            if !seen.insert(key) {
                trip_skipped += 1;
                continue;
            }

            match field.add_triplet(subject, predicate, object, weight, None, source) {
                Ok(_) => trip_ok += 1,
                Err(e) => {
                    trip_skipped += 1;
                    if trip_skipped <= 5 {
                        eprintln!("  Triplet error {trip_total}: {e}");
                    }
                }
            }

            if trip_total % batch_report == 0 {
                let rate = trip_total as f64 / t1.elapsed().as_secs_f64();
                println!(
                    "  triplets: {trip_total} processed ({trip_ok} ok, {trip_skipped} skipped) — {rate:.0}/s"
                );
            }
        }

        let trip_elapsed = t1.elapsed().as_secs_f64();
        println!(
            "\nTriplets complete: {trip_ok}/{trip_total} ingested, {trip_skipped} skipped \
             ({trip_elapsed:.1}s, {:.0}/s)",
            trip_ok as f64 / trip_elapsed.max(0.001)
        );
    }

    println!("\nFinal field state:");
    println!("  Total live memories: {}", field.memory_count());
}

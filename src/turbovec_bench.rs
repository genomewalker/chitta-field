//! Offline TurboQuant A/B against the production .emb sidecar (read-only).
//! Gold = exact f32 cosine. Compares latency, recall@10, and memory vs the
//! current SemanticIndex search path. Run:
//!   CHITTA_EMB_PATH=<...>.emb cargo test --release \
//!     turbovec_ab -- --ignored --nocapture

use crate::hnsw::SemanticIndex;
use std::time::Instant;

fn newest_emb_path() -> Option<std::path::PathBuf> {
    let dir = dirs_path();
    let mut best: Option<(std::time::SystemTime, std::path::PathBuf)> = None;
    for e in std::fs::read_dir(&dir).ok()?.flatten() {
        let p = e.path();
        if p.extension().map(|x| x == "emb").unwrap_or(false) {
            let m = e.metadata().ok()?.modified().ok()?;
            if best.as_ref().map(|(t, _)| m > *t).unwrap_or(true) {
                best = Some((m, p));
            }
        }
    }
    best.map(|(_, p)| p)
}

fn dirs_path() -> std::path::PathBuf {
    std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_default()
        .join(".claude/mind/chitta-field")
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 { v.iter().map(|x| x / n).collect() } else { v.to_vec() }
}

#[test]
#[ignore = "offline benchmark against the production .emb sidecar"]
fn turbovec_ab() {
    let emb_path = std::env::var_os("CHITTA_EMB_PATH")
        .map(std::path::PathBuf::from)
        .or_else(newest_emb_path)
        .expect("no .emb sidecar found; set CHITTA_EMB_PATH");
    eprintln!("emb: {:?}", emb_path);

    let mut idx = SemanticIndex::new();
    assert!(idx.load_embeddings_sidecar(&emb_path), "failed to load .emb");
    let ids: Vec<u64> = idx.all_ids().collect();
    let n = ids.len();
    let dim = crate::ops::EMBED_DIM;
    eprintln!("loaded {} embeddings, dim {}", n, dim);

    // Flattened normalized matrix + id map (insertion order = turbovec index).
    let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
    for id in &ids {
        flat.extend(normalize(idx.get_embedding(*id).unwrap()));
    }

    // Deterministic query sample.
    const Q: usize = 100;
    const K: usize = 10;
    let stride = n / Q;
    let q_rows: Vec<usize> = (0..Q).map(|i| i * stride).collect();

    // ── Gold: exact cosine top-K (normalized dot) ──
    let t0 = Instant::now();
    let mut gold: Vec<Vec<u64>> = Vec::with_capacity(Q);
    for &qr in &q_rows {
        let q = &flat[qr * dim..(qr + 1) * dim];
        let mut scored: Vec<(f32, usize)> = (0..n)
            .map(|r| {
                let v = &flat[r * dim..(r + 1) * dim];
                (q.iter().zip(v).map(|(a, b)| a * b).sum::<f32>(), r)
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        gold.push(scored[..K].iter().map(|(_, r)| ids[*r]).collect());
    }
    let gold_ms = t0.elapsed().as_millis() as f64 / Q as f64;
    eprintln!("gold exact scan: {:.1} ms/query", gold_ms);

    let recall = |got: &[Vec<u64>]| -> f64 {
        let mut hit = 0usize;
        for (g, w) in gold.iter().zip(got) {
            let gs: std::collections::HashSet<_> = g.iter().collect();
            hit += w.iter().filter(|id| gs.contains(id)).count();
        }
        hit as f64 / (Q * K) as f64
    };

    // ── Current engine (HNSW path) ──
    let t0 = Instant::now();
    let cur: Vec<Vec<u64>> = q_rows
        .iter()
        .map(|&qr| {
            idx.search(&flat[qr * dim..(qr + 1) * dim], K, None, None)
                .into_iter()
                .map(|h| h.memory_id)
                .collect()
        })
        .collect();
    let cur_us = t0.elapsed().as_micros() as f64 / Q as f64;
    eprintln!(
        "current engine (BENCH-CRIPPLED — .emb only, no .hnsw/.bin/.mu loaded; \
         production recall is LOCOMO-validated parity): {:.0} µs/query, recall@{K} {:.3}",
        cur_us,
        recall(&cur)
    );

    // ── TurboQuant at 4 and 2 bits ──
    for bits in [4usize, 2usize] {
        let t0 = Instant::now();
        let mut tq = turbovec::TurboQuantIndex::new(dim, bits).expect("construct");
        tq.add(&flat);
        tq.prepare();
        let build_s = t0.elapsed().as_secs_f64();

        // Batched (the library's design center) AND single-query timings.
        let mut q_flat: Vec<f32> = Vec::with_capacity(Q * dim);
        for &qr in &q_rows {
            q_flat.extend_from_slice(&flat[qr * dim..(qr + 1) * dim]);
        }
        let t0 = Instant::now();
        let res = tq.search(&q_flat, K);
        let batched_us = t0.elapsed().as_micros() as f64 / Q as f64;
        let got: Vec<Vec<u64>> = (0..Q)
            .map(|qi| {
                res.indices_for_query(qi)
                    .iter()
                    .map(|&r| ids[r as usize])
                    .collect()
            })
            .collect();
        let t0 = Instant::now();
        for &qr in &q_rows {
            let _ = tq.search(&flat[qr * dim..(qr + 1) * dim], K);
        }
        let us = t0.elapsed().as_micros() as f64 / Q as f64;
        eprintln!("turbovec {bits}-bit batched: {:.0} µs/query", batched_us);
        let mem_mb = (n * dim) as f64 * (bits as f64 / 8.0) / 1e6;
        eprintln!(
            "turbovec {bits}-bit: build {:.1}s, {:.0} µs/query, recall@{K} {:.3}, ~{:.0} MB codes (f32 = {:.0} MB)",
            build_s,
            us,
            recall(&got),
            mem_mb,
            (n * dim * 4) as f64 / 1e6
        );
    }
}

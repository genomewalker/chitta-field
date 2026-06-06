use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::PathBuf;

use chitta_field::field::ChittaField;
use chitta_field::ops::EMBED_DIM;

fn make_embedding(i: usize) -> Vec<f32> {
    let mut e = vec![0.0f32; EMBED_DIM];
    e[i % EMBED_DIM] = 1.0;
    let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut e {
        *x /= norm;
    }
    e
}

fn populate_store(path: &PathBuf, n: usize) {
    let field = ChittaField::open(path.clone()).expect("open");
    for i in 0..n {
        let emb = make_embedding(i);
        let content = format!("memory-{i} wal replay benchmark synthetic");
        field
            .put_memory(
                "wisdom",
                "bench",
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
    }
    field.flush().unwrap();
}

fn bench_wal_replay(c: &mut Criterion) {
    let ns: &[usize] = &[100, 1000, 10000];

    let mut group = c.benchmark_group("wal_replay");

    for &n in ns {
        let base = std::env::temp_dir().join(format!("chitta-wal-replay-{n}"));
        let data = base.join("data");
        let _ = std::fs::remove_dir_all(&base);
        populate_store(&data, n);

        group.bench_with_input(BenchmarkId::new("reopen", n), &n, |b, _| {
            b.iter(|| {
                let field = ChittaField::open(data.clone()).expect("open");
                assert!(field.memory_count() > 0);
            });
        });

        let _ = std::fs::remove_dir_all(&base);
    }

    group.finish();
}

criterion_group!(benches, bench_wal_replay);
criterion_main!(benches);

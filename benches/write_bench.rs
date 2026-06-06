use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::PathBuf;

use chitta_field::field::ChittaField;
use chitta_field::ops::EMBED_DIM;

fn tmp_field(tag: &str) -> (ChittaField, PathBuf) {
    let base = std::env::temp_dir().join(format!("chitta-write-bench-{tag}"));
    let data = base.join("data");
    let _ = std::fs::remove_dir_all(&base);
    let field = ChittaField::open(data).expect("open");
    (field, base)
}

fn make_embedding(i: usize) -> Vec<f32> {
    let mut e = vec![0.0f32; EMBED_DIM];
    e[i % EMBED_DIM] = 1.0;
    let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut e {
        *x /= norm;
    }
    e
}

fn bench_write(c: &mut Criterion) {
    let sizes: &[(&str, usize)] = &[("small_128B", 128), ("medium_1KB", 1024), ("large_8KB", 8192)];

    let mut group = c.benchmark_group("remember");
    group.sample_size(100);

    for &(label, payload_size) in sizes {
        let (field, base) = tmp_field(label);
        let payload = vec![b'x'; payload_size];
        let mut counter = 0usize;

        group.bench_with_input(BenchmarkId::new("put_memory", label), &payload_size, |b, _| {
            b.iter(|| {
                let emb = make_embedding(counter);
                counter += 1;
                field
                    .put_memory(
                        "wisdom",
                        "bench",
                        &payload,
                        &emb,
                        0.9,
                        0.001,
                        counter as i64 * 1000,
                        vec![],
                        None,
                        None,
                    )
                    .unwrap();
            });
        });

        let _ = std::fs::remove_dir_all(&base);
    }

    group.finish();
}

criterion_group!(benches, bench_write);
criterion_main!(benches);

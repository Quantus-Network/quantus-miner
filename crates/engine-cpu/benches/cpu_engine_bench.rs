use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engine_cpu::{FastCpuEngine, MinerEngine, Range};
use pow_core::{distance_from_y, JobContext};
use primitive_types::U512;
use rand::RngCore;
use std::sync::atomic::AtomicBool;

fn bench_cpu_fast_engine(c: &mut Criterion) {
    // Create the engine
    let engine = FastCpuEngine::new();
    let cancel_flag = AtomicBool::new(false);

    let large_range = Range {
        start: U512::from(1000u64),
        end: U512::from(101000u64), // 100,000 nonces
    };

    c.bench_function("cpu_fast_large_range", |b| {
        b.iter(|| {
            let mut header = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut header);
            let threshold = U512::one() << 500;
            let ctx = JobContext::new(header, threshold);

            let result = engine.search_range(
                black_box(&ctx),
                black_box(large_range.clone()),
                black_box(&cancel_flag),
            );
            black_box(result)
        })
    });
}

fn bench_distance_from_y(c: &mut Criterion) {
    // Create a test job context
    let mut header = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut header);
    let threshold = U512::one() << 500;
    let ctx = JobContext::new(header, threshold);

    // Create some test y values
    let test_y_values: Vec<U512> = (0..100).map(|i| U512::from(1000u64 + i)).collect();

    c.bench_function("distance_from_y_single", |b| {
        let mut i = 0;
        b.iter(|| {
            let y = test_y_values[i % test_y_values.len()];
            i += 1;
            let distance = distance_from_y(black_box(&ctx), black_box(y));
            black_box(distance)
        })
    });

}
criterion_group!(benches, bench_cpu_fast_engine, bench_distance_from_y);
criterion_main!(benches);



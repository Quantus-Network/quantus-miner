use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engine_cpu::{FastCpuEngine, MinerEngine, Range};
use pow_core::{hash_from_nonce, JobContext};
use primitive_types::U512;
use rand::RngCore;
use std::sync::atomic::AtomicBool;

fn bench_cpu_fast_engine(c: &mut Criterion) {
    // Create the engine
    let engine = FastCpuEngine::new();
    let cancel_flag = AtomicBool::new(false);

    let large_range = Range {
        start: U512::from(0u64),
        end: U512::from(100000u64), // 100,000 nonces
    };

    c.bench_function("cpu_fast_large_range", |b| {
        b.iter(|| {
            let mut header = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut header);
            let difficulty = U512::from(1000000u64);
            let ctx = JobContext::new(header, difficulty);

            let result = engine.search_range(
                black_box(&ctx),
                black_box(large_range.clone()),
                black_box(&cancel_flag),
            );
            black_box(result)
        })
    });
}

fn bench_hash_from_nonce(c: &mut Criterion) {
    // Create a test job context
    let mut header = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut header);
    let difficulty = U512::from(1000u64);
    let ctx = JobContext::new(header, difficulty);

    // Create some test nonce values
    let test_nonce_values: Vec<U512> = (0..100).map(|i| U512::from(1000u64 + i)).collect();

    c.bench_function("hash_from_nonce_single", |b| {
        let mut i = 0;
        b.iter(|| {
            let nonce = test_nonce_values[i % test_nonce_values.len()];
            i += 1;
            let hash = hash_from_nonce(black_box(&ctx), black_box(nonce));
            black_box(hash)
        })
    });
}
criterion_group!(benches, bench_cpu_fast_engine, bench_hash_from_nonce);
criterion_main!(benches);

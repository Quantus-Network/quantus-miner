use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engine_cpu::{AtomicBoolCancelCheck, FastCpuEngine, MinerEngine, Range};
use pow_core::{hash_from_nonce, JobContext};
use primitive_types::U512;
use rand::RngCore;
use std::sync::atomic::AtomicBool;

const BENCHMARK_DIFFICULTY: U512 = U512::MAX;

fn benchmark_context() -> JobContext {
    let mut header = [0u8; 32];
    rand::rng().fill_bytes(&mut header);
    JobContext::new(header, BENCHMARK_DIFFICULTY)
}

fn bench_cpu_fast_engine(c: &mut Criterion) {
    let engine = FastCpuEngine::new(10_000);
    let cancel_flag = AtomicBool::new(false);
    let cancel_check = AtomicBoolCancelCheck(&cancel_flag);
    let ctx = benchmark_context();

    let large_range = Range {
        start: U512::from(0u64),
        end: U512::from(99_999u64),
    };

    c.bench_function("cpu_fast_large_range", |b| {
        b.iter(|| {
            let result = engine.search_range(
                black_box(&ctx),
                black_box(large_range.clone()),
                black_box(&cancel_check),
            );
            black_box(result)
        })
    });
}

fn bench_hash_from_nonce(c: &mut Criterion) {
    let ctx = benchmark_context();

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

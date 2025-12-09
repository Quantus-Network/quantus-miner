use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engine_cpu::{MinerEngine, Range};
use engine_gpu::GpuEngine;
use pow_core::{hash_from_nonce, JobContext};
use primitive_types::U512;
use rand::RngCore;
use std::sync::atomic::AtomicBool;

fn bench_gpu_engine(c: &mut Criterion) {
    // Create the engine
    let engine = GpuEngine::new();
    let cancel_flag = AtomicBool::new(false);

    // Small range for benchmarking - should complete quickly
    let small_range = Range {
        start: U512::from(0u64),
        end: U512::from(100_000u64), // 100K nonces
    };

    // Medium range for testing batch efficiency
    let medium_range = Range {
        start: U512::from(0u64),
        end: U512::from(1_000_000u64), // 1M nonces
    };

    // Large range for stress testing
    let large_range = Range {
        start: U512::from(0u64),
        end: U512::from(10_000_000u64), // 10M nonces
    };

    c.bench_function("gpu_small_range_100k", |b| {
        b.iter(|| {
            let mut header = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut header);
            let difficulty = U512::from(10_000_000u64); // Hard enough to not find solution
            let ctx = JobContext::new(header, difficulty);

            let result = engine.search_range(
                black_box(&ctx),
                black_box(small_range.clone()),
                black_box(&cancel_flag),
            );
            black_box(result)
        })
    });

    c.bench_function("gpu_medium_range_1m", |b| {
        b.iter(|| {
            let mut header = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut header);
            let difficulty = U512::from(50_000_000u64);
            let ctx = JobContext::new(header, difficulty);

            let result = engine.search_range(
                black_box(&ctx),
                black_box(medium_range.clone()),
                black_box(&cancel_flag),
            );
            black_box(result)
        })
    });

    c.bench_function("gpu_large_range_10m", |b| {
        b.iter(|| {
            let mut header = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut header);
            let difficulty = U512::from(100_000_000u64);
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

fn bench_gpu_solution_finding(c: &mut Criterion) {
    let engine = GpuEngine::new();
    let cancel_flag = AtomicBool::new(false);

    // Small range with easy difficulty to find solutions quickly
    let solution_range = Range {
        start: U512::from(0u64),
        end: U512::from(1_000_000u64), // 1M nonces should contain solution
    };

    c.bench_function("gpu_find_solution", |b| {
        b.iter(|| {
            let mut header = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut header);
            let difficulty = U512::from(1000u64); // Easy difficulty
            let ctx = JobContext::new(header, difficulty);

            let result = engine.search_range(
                black_box(&ctx),
                black_box(solution_range.clone()),
                black_box(&cancel_flag),
            );
            black_box(result)
        })
    });
}

fn bench_gpu_vs_hash_from_nonce(c: &mut Criterion) {
    // Compare GPU batch processing vs individual hash computations
    let mut header = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut header);
    let difficulty = U512::from(1000u64);
    let ctx = JobContext::new(header, difficulty);

    // Create test nonce values
    let test_nonce_values: Vec<U512> = (0..1000).map(|i| U512::from(1000u64 + i)).collect();

    c.bench_function("hash_from_nonce_1000", |b| {
        b.iter(|| {
            let mut results = Vec::new();
            for nonce in &test_nonce_values {
                let hash = hash_from_nonce(black_box(&ctx), black_box(*nonce));
                results.push(hash);
            }
            black_box(results)
        })
    });

    // GPU equivalent - process same 1000 nonces
    let engine = GpuEngine::new();
    let cancel_flag = AtomicBool::new(false);
    let nonce_range = Range {
        start: U512::from(1000u64),
        end: U512::from(2000u64), // Same 1000 nonces
    };

    c.bench_function("gpu_process_1000", |b| {
        b.iter(|| {
            let result = engine.search_range(
                black_box(&ctx),
                black_box(nonce_range.clone()),
                black_box(&cancel_flag),
            );
            black_box(result)
        })
    });
}

fn bench_gpu_cancellation_responsiveness(c: &mut Criterion) {
    let engine = GpuEngine::new();

    c.bench_function("gpu_immediate_cancel", |b| {
        b.iter(|| {
            let mut header = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut header);
            let difficulty = U512::from(1000000u64);
            let ctx = JobContext::new(header, difficulty);

            // Pre-cancelled flag - should return immediately
            let cancel_flag = AtomicBool::new(true);
            let range = Range {
                start: U512::from(0u64),
                end: U512::MAX, // Huge range
            };

            let result =
                engine.search_range(black_box(&ctx), black_box(range), black_box(&cancel_flag));
            black_box(result)
        })
    });
}

criterion_group!(
    benches,
    bench_gpu_engine,
    bench_gpu_solution_finding,
    bench_gpu_vs_hash_from_nonce,
    bench_gpu_cancellation_responsiveness
);
criterion_main!(benches);

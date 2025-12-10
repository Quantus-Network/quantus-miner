use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engine_cpu::{FastCpuEngine, MinerEngine, Range};
use engine_gpu::GpuEngine;
use pow_core::JobContext;
use primitive_types::U512;
use rand::RngCore;
use std::sync::atomic::AtomicBool;

fn bench_cpu_vs_gpu_small(c: &mut Criterion) {
    let cpu_engine = FastCpuEngine::new();
    let gpu_engine = GpuEngine::new();
    let cancel_flag = AtomicBool::new(false);

    // Small range: 10K nonces - reasonable for benchmarking
    let small_range = Range {
        start: U512::from(0u64),
        end: U512::from(10_000u64),
    };

    let mut group = c.benchmark_group("small_range_10k");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(3));

    group.bench_function("cpu", |b| {
        b.iter(|| {
            let mut header = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut header);
            let difficulty = U512::from(u64::MAX); // High difficulty - no solutions expected
            let ctx = JobContext::new(header, difficulty);

            let result = cpu_engine.search_range(
                black_box(&ctx),
                black_box(small_range.clone()),
                black_box(&cancel_flag),
            );
            black_box(result)
        })
    });

    group.bench_function("gpu", |b| {
        b.iter(|| {
            let mut header = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut header);
            let difficulty = U512::from(u64::MAX); // High difficulty - no solutions expected
            let ctx = JobContext::new(header, difficulty);

            let result = gpu_engine.search_range(
                black_box(&ctx),
                black_box(small_range.clone()),
                black_box(&cancel_flag),
            );
            black_box(result)
        })
    });

    group.finish();
}

fn bench_cpu_vs_gpu_medium(c: &mut Criterion) {
    let cpu_engine = FastCpuEngine::new();
    let gpu_engine = GpuEngine::new();
    let cancel_flag = AtomicBool::new(false);

    // Medium range: 100K nonces
    let medium_range = Range {
        start: U512::from(0u64),
        end: U512::from(100_000u64),
    };

    let mut group = c.benchmark_group("medium_range_100k");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(3));

    group.bench_function("cpu", |b| {
        b.iter(|| {
            let mut header = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut header);
            let difficulty = U512::from(u64::MAX); // High difficulty - no solutions expected
            let ctx = JobContext::new(header, difficulty);

            let result = cpu_engine.search_range(
                black_box(&ctx),
                black_box(medium_range.clone()),
                black_box(&cancel_flag),
            );
            black_box(result)
        })
    });

    group.bench_function("gpu", |b| {
        b.iter(|| {
            let mut header = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut header);
            let difficulty = U512::from(u64::MAX); // High difficulty - no solutions expected
            let ctx = JobContext::new(header, difficulty);

            let result = gpu_engine.search_range(
                black_box(&ctx),
                black_box(medium_range.clone()),
                black_box(&cancel_flag),
            );
            black_box(result)
        })
    });

    group.finish();
}

fn bench_cpu_vs_gpu_large(c: &mut Criterion) {
    let cpu_engine = FastCpuEngine::new();
    let gpu_engine = GpuEngine::new();
    let cancel_flag = AtomicBool::new(false);

    // Large range: 1M nonces - where GPU should really shine
    let large_range = Range {
        start: U512::from(0u64),
        end: U512::from(1_000_000u64),
    };

    let mut group = c.benchmark_group("large_range_1m");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(5));

    group.bench_function("cpu", |b| {
        b.iter(|| {
            let mut header = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut header);
            let difficulty = U512::from(u64::MAX); // High difficulty - no solutions expected
            let ctx = JobContext::new(header, difficulty);

            let result = cpu_engine.search_range(
                black_box(&ctx),
                black_box(large_range.clone()),
                black_box(&cancel_flag),
            );
            black_box(result)
        })
    });

    group.bench_function("gpu", |b| {
        b.iter(|| {
            let mut header = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut header);
            let difficulty = U512::from(u64::MAX); // High difficulty - no solutions expected
            let ctx = JobContext::new(header, difficulty);

            let result = gpu_engine.search_range(
                black_box(&ctx),
                black_box(large_range.clone()),
                black_box(&cancel_flag),
            );
            black_box(result)
        })
    });

    group.finish();
}

fn bench_solution_finding(c: &mut Criterion) {
    let cpu_engine = FastCpuEngine::new();
    let gpu_engine = GpuEngine::new();
    let cancel_flag = AtomicBool::new(false);

    // Range where we expect to find solutions quickly
    let solution_range = Range {
        start: U512::from(0u64),
        end: U512::from(50_000u64),
    };

    let mut group = c.benchmark_group("solution_finding");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(3));

    group.bench_function("cpu_find_solution", |b| {
        b.iter(|| {
            let mut header = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut header);
            let difficulty = U512::from(10_000u64); // Easy difficulty - should find solution
            let ctx = JobContext::new(header, difficulty);

            let result = cpu_engine.search_range(
                black_box(&ctx),
                black_box(solution_range.clone()),
                black_box(&cancel_flag),
            );
            black_box(result)
        })
    });

    group.bench_function("gpu_find_solution", |b| {
        b.iter(|| {
            let mut header = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut header);
            let difficulty = U512::from(10_000u64); // Easy difficulty - should find solution
            let ctx = JobContext::new(header, difficulty);

            let result = gpu_engine.search_range(
                black_box(&ctx),
                black_box(solution_range.clone()),
                black_box(&cancel_flag),
            );
            black_box(result)
        })
    });

    group.finish();
}

fn bench_throughput_per_second(c: &mut Criterion) {
    let cpu_engine = FastCpuEngine::new();
    let gpu_engine = GpuEngine::new();
    let cancel_flag = AtomicBool::new(false);

    // Fixed time benchmark - see how many hashes we can do in 1 second
    let throughput_range = Range {
        start: U512::from(0u64),
        end: U512::from(10_000_000u64), // 10M nonce range
    };

    let mut group = c.benchmark_group("throughput_comparison");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(5));

    group.bench_function("cpu_throughput", |b| {
        b.iter(|| {
            let mut header = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut header);
            let difficulty = U512::from(u64::MAX); // High difficulty - no solutions expected
            let ctx = JobContext::new(header, difficulty);

            let result = cpu_engine.search_range(
                black_box(&ctx),
                black_box(throughput_range.clone()),
                black_box(&cancel_flag),
            );
            black_box(result)
        })
    });

    group.bench_function("gpu_throughput", |b| {
        b.iter(|| {
            let mut header = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut header);
            let difficulty = U512::from(u64::MAX); // High difficulty - no solutions expected
            let ctx = JobContext::new(header, difficulty);

            let result = gpu_engine.search_range(
                black_box(&ctx),
                black_box(throughput_range.clone()),
                black_box(&cancel_flag),
            );
            black_box(result)
        })
    });

    group.finish();
}

fn bench_gpu_batch_efficiency(c: &mut Criterion) {
    let gpu_engine = GpuEngine::new();
    let cancel_flag = AtomicBool::new(false);

    let mut group = c.benchmark_group("gpu_batch_sizes");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(3));

    // Test different batch sizes to see GPU efficiency
    let small_batch = Range {
        start: U512::from(0u64),
        end: U512::from(1_000u64), // 1K nonces - very small for GPU
    };

    let medium_batch = Range {
        start: U512::from(0u64),
        end: U512::from(50_000u64), // 50K nonces - medium
    };

    let large_batch = Range {
        start: U512::from(0u64),
        end: U512::from(500_000u64), // 500K nonces - large
    };

    group.bench_function("gpu_1k_batch", |b| {
        b.iter(|| {
            let mut header = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut header);
            let difficulty = U512::from(u64::MAX);
            let ctx = JobContext::new(header, difficulty);

            let result = gpu_engine.search_range(
                black_box(&ctx),
                black_box(small_batch.clone()),
                black_box(&cancel_flag),
            );
            black_box(result)
        })
    });

    group.bench_function("gpu_50k_batch", |b| {
        b.iter(|| {
            let mut header = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut header);
            let difficulty = U512::from(u64::MAX);
            let ctx = JobContext::new(header, difficulty);

            let result = gpu_engine.search_range(
                black_box(&ctx),
                black_box(medium_batch.clone()),
                black_box(&cancel_flag),
            );
            black_box(result)
        })
    });

    group.bench_function("gpu_500k_batch", |b| {
        b.iter(|| {
            let mut header = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut header);
            let difficulty = U512::from(u64::MAX);
            let ctx = JobContext::new(header, difficulty);

            let result = gpu_engine.search_range(
                black_box(&ctx),
                black_box(large_batch.clone()),
                black_box(&cancel_flag),
            );
            black_box(result)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_cpu_vs_gpu_small,
    bench_cpu_vs_gpu_medium,
    bench_cpu_vs_gpu_large,
    bench_solution_finding,
    bench_throughput_per_second,
    bench_gpu_batch_efficiency
);
criterion_main!(benches);

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engine_cpu::{AtomicBoolCancelCheck, FastCpuEngine, MinerEngine, Range};
use engine_gpu::GpuEngine;
use pow_core::JobContext;
use primitive_types::U512;
use rand::RngCore;
use std::sync::atomic::AtomicBool;

const BENCHMARK_DIFFICULTY: U512 = U512::MAX;

fn benchmark_context() -> JobContext {
    let mut header = [0u8; 32];
    rand::rng().fill_bytes(&mut header);
    JobContext::new(header, BENCHMARK_DIFFICULTY)
}

/// Drop thread-local wgpu buffers before `GpuEngine` is dropped. Criterion
/// creates a fresh engine per group; without this, TLS buffers outlive the
/// device and the next group panics (`Buffer[…] does not exist`).
fn teardown_gpu(engine: GpuEngine) {
    GpuEngine::clear_worker_resources();
    drop(engine);
}

fn bench_cpu_vs_gpu_small(c: &mut Criterion) {
    let cpu_engine = FastCpuEngine::new(10_000);
    let gpu_engine = GpuEngine::try_new(10_000_000, 0, false).expect("Failed to init GPU");
    let cancel_flag = AtomicBool::new(false);
    let cancel_check = AtomicBoolCancelCheck(&cancel_flag);
    let ctx = benchmark_context();

    let small_range = Range {
        start: U512::from(0u64),
        end: U512::from(9_999u64),
    };

    let mut group = c.benchmark_group("small_range_10k");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(3));

    group.bench_function("cpu", |b| {
        b.iter(|| {
            let result = cpu_engine.search_range(
                black_box(&ctx),
                black_box(small_range.clone()),
                black_box(&cancel_check),
            );
            black_box(result)
        })
    });

    group.bench_function("gpu", |b| {
        b.iter(|| {
            let result = gpu_engine.search_range(
                black_box(&ctx),
                black_box(small_range.clone()),
                black_box(&cancel_check),
            );
            black_box(result)
        })
    });

    group.finish();
    teardown_gpu(gpu_engine);
}

fn bench_cpu_vs_gpu_medium(c: &mut Criterion) {
    let cpu_engine = FastCpuEngine::new(10_000);
    let gpu_engine = GpuEngine::try_new(10_000_000, 0, false).expect("Failed to init GPU");
    let cancel_flag = AtomicBool::new(false);
    let cancel_check = AtomicBoolCancelCheck(&cancel_flag);
    let ctx = benchmark_context();

    let medium_range = Range {
        start: U512::from(0u64),
        end: U512::from(99_999u64),
    };

    let mut group = c.benchmark_group("medium_range_100k");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(3));

    group.bench_function("cpu", |b| {
        b.iter(|| {
            let result = cpu_engine.search_range(
                black_box(&ctx),
                black_box(medium_range.clone()),
                black_box(&cancel_check),
            );
            black_box(result)
        })
    });

    group.bench_function("gpu", |b| {
        b.iter(|| {
            let result = gpu_engine.search_range(
                black_box(&ctx),
                black_box(medium_range.clone()),
                black_box(&cancel_check),
            );
            black_box(result)
        })
    });

    group.finish();
    teardown_gpu(gpu_engine);
}

fn bench_cpu_vs_gpu_large(c: &mut Criterion) {
    let cpu_engine = FastCpuEngine::new(10_000);
    let gpu_engine = GpuEngine::try_new(10_000_000, 0, false).expect("Failed to init GPU");
    let cancel_flag = AtomicBool::new(false);
    let cancel_check = AtomicBoolCancelCheck(&cancel_flag);
    let ctx = benchmark_context();

    let large_range = Range {
        start: U512::from(0u64),
        end: U512::from(999_999u64),
    };

    let mut group = c.benchmark_group("large_range_1m");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(5));

    group.bench_function("cpu", |b| {
        b.iter(|| {
            let result = cpu_engine.search_range(
                black_box(&ctx),
                black_box(large_range.clone()),
                black_box(&cancel_check),
            );
            black_box(result)
        })
    });

    group.bench_function("gpu", |b| {
        b.iter(|| {
            let result = gpu_engine.search_range(
                black_box(&ctx),
                black_box(large_range.clone()),
                black_box(&cancel_check),
            );
            black_box(result)
        })
    });

    group.finish();
    teardown_gpu(gpu_engine);
}

fn bench_throughput_per_second(c: &mut Criterion) {
    let cpu_engine = FastCpuEngine::new(10_000);
    let gpu_engine = GpuEngine::try_new(10_000_000, 0, false).expect("Failed to init GPU");
    let cancel_flag = AtomicBool::new(false);
    let cancel_check = AtomicBoolCancelCheck(&cancel_flag);
    let ctx = benchmark_context();

    let throughput_range = Range {
        start: U512::from(0u64),
        end: U512::from(9_999_999u64),
    };

    let mut group = c.benchmark_group("throughput_comparison");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(5));

    group.bench_function("cpu_throughput", |b| {
        b.iter(|| {
            let result = cpu_engine.search_range(
                black_box(&ctx),
                black_box(throughput_range.clone()),
                black_box(&cancel_check),
            );
            black_box(result)
        })
    });

    group.bench_function("gpu_throughput", |b| {
        b.iter(|| {
            let result = gpu_engine.search_range(
                black_box(&ctx),
                black_box(throughput_range.clone()),
                black_box(&cancel_check),
            );
            black_box(result)
        })
    });

    group.finish();
    teardown_gpu(gpu_engine);
}

fn bench_gpu_batch_efficiency(c: &mut Criterion) {
    let gpu_engine = GpuEngine::try_new(10_000_000, 0, false).expect("Failed to init GPU");
    let cancel_flag = AtomicBool::new(false);
    let cancel_check = AtomicBoolCancelCheck(&cancel_flag);
    let ctx = benchmark_context();

    let mut group = c.benchmark_group("gpu_batch_sizes");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(3));

    let small_batch = Range {
        start: U512::from(0u64),
        end: U512::from(999u64),
    };

    let medium_batch = Range {
        start: U512::from(0u64),
        end: U512::from(49_999u64),
    };

    let large_batch = Range {
        start: U512::from(0u64),
        end: U512::from(499_999u64),
    };

    group.bench_function("gpu_1k_batch", |b| {
        b.iter(|| {
            let result = gpu_engine.search_range(
                black_box(&ctx),
                black_box(small_batch.clone()),
                black_box(&cancel_check),
            );
            black_box(result)
        })
    });

    group.bench_function("gpu_50k_batch", |b| {
        b.iter(|| {
            let result = gpu_engine.search_range(
                black_box(&ctx),
                black_box(medium_batch.clone()),
                black_box(&cancel_check),
            );
            black_box(result)
        })
    });

    group.bench_function("gpu_500k_batch", |b| {
        b.iter(|| {
            let result = gpu_engine.search_range(
                black_box(&ctx),
                black_box(large_batch.clone()),
                black_box(&cancel_check),
            );
            black_box(result)
        })
    });

    group.finish();
    teardown_gpu(gpu_engine);
}

criterion_group!(
    benches,
    bench_cpu_vs_gpu_small,
    bench_cpu_vs_gpu_medium,
    bench_cpu_vs_gpu_large,
    bench_throughput_per_second,
    bench_gpu_batch_efficiency
);
criterion_main!(benches);

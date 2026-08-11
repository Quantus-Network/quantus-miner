use engine_cpu::{AtomicBoolCancelCheck, MinerEngine, Range};
use engine_gpu::GpuEngine;
use primitive_types::U512;
use std::sync::atomic::AtomicBool;

fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    let total: u64 = args
        .get(1)
        .map(|s| s.parse().expect("total nonces"))
        .unwrap_or(4_000_000);
    let batch: u32 = args
        .get(2)
        .map(|s| s.parse().expect("batch size"))
        .unwrap_or(1_000_000);

    let engine = GpuEngine::try_new(batch, 0, false).expect("GPU init failed");
    let cancel_flag = AtomicBool::new(false);
    let cancel = AtomicBoolCancelCheck(&cancel_flag);

    let header = [42u8; 32];
    let ctx = engine.prepare_context(header, U512::from(u64::MAX));

    let range = Range {
        start: U512::from(1u64) << 200,
        end: (U512::from(1u64) << 200) + U512::from(total - 1),
    };

    // Warmup
    let warm = Range {
        start: range.start,
        end: range.start + U512::from(200_000u64),
    };
    engine.search_range(&ctx, warm, &cancel);

    let start = std::time::Instant::now();
    let status = engine.search_range(&ctx, range, &cancel);
    let elapsed = start.elapsed().as_secs_f64();

    let hashes = match status {
        engine_cpu::EngineStatus::Exhausted { hash_count } => hash_count,
        engine_cpu::EngineStatus::Found { hash_count, .. } => hash_count,
        other => panic!("unexpected status: {other:?}"),
    };
    println!(
        "{} hashes in {:.3}s = {:.3} MH/s",
        hashes,
        elapsed,
        hashes as f64 / elapsed / 1e6
    );
}

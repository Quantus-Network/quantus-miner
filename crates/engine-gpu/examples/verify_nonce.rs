use engine_cpu::{BaselineCpuEngine, EngineStatus, MinerEngine, Range};
use engine_gpu::GpuEngine;
use primitive_types::U512;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

fn main() {
    // Initialize logging
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();

    log::info!("Starting verify_nonce example");

    // 1. Setup Context
    // Use a fixed header and easy difficulty (1) so any nonce is valid
    let header = [1u8; 32];
    let difficulty = U512::from(u64::MAX); // High difficulty - no solutions expected
    let cpu_engine = BaselineCpuEngine::new();
    let ctx = cpu_engine.prepare_context(header, difficulty);

    log::info!("Context prepared. Difficulty: {}", difficulty);

    let cancel = AtomicBool::new(false);

    // 3. Verify with GPU engine
    log::info!("Initializing GPU engine...");
    let gpu_engine = GpuEngine::new(Duration::from_millis(3000));

    // Search a small range around the valid nonce
    let gpu_range = Range {
        start: U512::from(0u64),
        end: U512::from(1_000_000u64), // Search 1,000,000 nonces
    };

    log::info!(
        "Searching for nonce with GPU engine in range {} - {}",
        gpu_range.start,
        gpu_range.end
    );
    let start = std::time::Instant::now();
    let gpu_result = gpu_engine.search_range(&ctx, gpu_range, &cancel);
    let elapsed = start.elapsed();

    log::info!("GPU search took {:?}", elapsed);

    match gpu_result {
        EngineStatus::Found { candidate, .. } => {
            log::info!("GPU found nonce: {}", candidate.nonce);
            log::info!("GPU hash: {:x}", candidate.hash);
        }
        EngineStatus::Exhausted { .. } => {
            log::info!("GPU exhausted range (expected)");
        }
        EngineStatus::Cancelled { .. } => {
            log::error!("FAILURE: GPU search cancelled!");
        }
        EngineStatus::Running { .. } => {
            log::error!("FAILURE: GPU returned Running status!");
        }
    }
}

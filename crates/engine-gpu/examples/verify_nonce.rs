use engine_cpu::{BaselineCpuEngine, EngineStatus, MinerEngine, Range};
use engine_gpu::GpuEngine;
use pow_core::JobContext;
use primitive_types::U512;
use std::sync::atomic::AtomicBool;

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
    let difficulty = U512::from(1u64);
    let cpu_engine = BaselineCpuEngine::new();
    let ctx = cpu_engine.prepare_context(header, difficulty);

    log::info!("Context prepared. Difficulty: {}", difficulty);

    // 2. Find a valid nonce using CPU engine
    // Since difficulty is 1, the first nonce (0) should be valid.
    // But let's search a small range to be sure.
    let range = Range {
        start: U512::from(0u64),
        end: U512::from(100u64),
    };
    let cancel = AtomicBool::new(false);

    log::info!("Searching for nonce with CPU engine...");
    let cpu_result = cpu_engine.search_range(&ctx, range.clone(), &cancel);

    let valid_nonce = match cpu_result {
        EngineStatus::Found { candidate, .. } => {
            log::info!("CPU found nonce: {}", candidate.nonce);
            log::info!("CPU hash: {:x}", candidate.hash);
            candidate.nonce
        }
        _ => {
            log::error!("CPU failed to find a nonce in range!");
            return;
        }
    };

    // 3. Verify with GPU engine
    log::info!("Initializing GPU engine...");
    let gpu_engine = GpuEngine::new();

    // Search a small range around the valid nonce
    let gpu_range = Range {
        start: valid_nonce,
        end: valid_nonce + U512::from(10u64), // Search 10 nonces starting from the valid one
    };

    log::info!(
        "Searching for nonce with GPU engine in range {} - {}",
        gpu_range.start,
        gpu_range.end
    );
    let gpu_result = gpu_engine.search_range(&ctx, gpu_range, &cancel);

    match gpu_result {
        EngineStatus::Found { candidate, .. } => {
            log::info!("GPU found nonce: {}", candidate.nonce);
            log::info!("GPU hash: {:x}", candidate.hash);

            if candidate.nonce == valid_nonce {
                log::info!("SUCCESS: GPU found the same nonce as CPU!");
            } else {
                log::warn!(
                    "GPU found a DIFFERENT nonce: {} (Expected: {})",
                    candidate.nonce,
                    valid_nonce
                );
            }
        }
        EngineStatus::Exhausted { .. } => {
            log::error!("FAILURE: GPU exhausted range without finding the nonce!");
        }
        EngineStatus::Cancelled { .. } => {
            log::error!("FAILURE: GPU search cancelled!");
        }
        EngineStatus::Running { .. } => {
            log::error!("FAILURE: GPU returned Running status!");
        }
    }
}

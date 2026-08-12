use engine_cpu::{AtomicBoolCancelCheck, EngineStatus, MinerEngine, Range};
use engine_gpu::GpuEngine;
use primitive_types::U512;
use rand::RngCore;
use std::sync::atomic::AtomicBool;

fn main() {
    env_logger::init();
    let jobs: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("job count"))
        .unwrap_or(25);

    let engine = GpuEngine::try_new(1_000_000, 0, false).expect("GPU init failed");
    let cancel_flag = AtomicBool::new(false);
    let cancel = AtomicBoolCancelCheck(&cancel_flag);
    let mut rng = rand::rng();

    let mut found = 0usize;
    for job in 0..jobs {
        let mut header = [0u8; 32];
        rng.fill_bytes(&mut header);
        let difficulty = U512::from(100_000u64);
        let ctx = engine.prepare_context(header, difficulty);

        let start = if job == 0 {
            // Cross a 2^256 boundary: the high nonce half changes mid-range,
            // exercising the midstate batch clamp.
            (U512::from(3u64) << 256) - U512::from(1_000u64)
        } else {
            let mut start_bytes = [0u8; 64];
            rng.fill_bytes(&mut start_bytes);
            // Keep clear of the very top so range arithmetic cannot wrap
            start_bytes[0] = 0;
            U512::from_big_endian(&start_bytes)
        };
        let range = Range {
            start,
            end: start + U512::from(10_000_000u64),
        };

        match engine.search_range(&ctx, range.clone(), &cancel) {
            EngineStatus::Found { candidate, .. } => {
                let cpu_hash = pow_core::hash_from_nonce(&ctx, candidate.nonce);
                assert_eq!(
                    cpu_hash, candidate.hash,
                    "job {job}: GPU hash != CPU hash for nonce {}",
                    candidate.nonce
                );
                assert!(cpu_hash < ctx.target, "job {job}: hash not below target");
                assert!(
                    candidate.nonce >= range.start && candidate.nonce <= range.end,
                    "job {job}: nonce outside range"
                );
                found += 1;
            }
            EngineStatus::Exhausted { .. } => {}
            other => panic!("job {job}: unexpected status {other:?}"),
        }
    }
    assert!(found > 0, "no solutions found across {jobs} jobs");
    println!("PARITY OK: {found}/{jobs} jobs found solutions, all verified against CPU");
}

//! Offline end-to-end CPU range comparison against the independent full hash.
use engine_cpu::{AtomicBoolCancelCheck, EngineStatus, FastCpuEngine, MinerEngine, Range};
use pow_core::{hash_from_nonce, JobContext};
use primitive_types::U512;
use std::{hint::black_box, sync::atomic::AtomicBool, time::Instant};

fn main() {
    let engine = FastCpuEngine::new(10_000);
    let flag = AtomicBool::new(false);
    let cancel = AtomicBoolCancelCheck(&flag);
    let count = 20_000u64;
    let mut totals = [0.0; 2];
    // Warm both paths, then ABBA with the same count and header per pair.
    for round in 0..13u64 {
        let ctx = JobContext::new([round as u8; 32], U512::MAX);
        let start = U512::one() << 200;
        for index in [0, 1, 1, 0] {
            let timer = Instant::now();
            if index == 0 {
                let mut hashes = 0u64;
                for offset in 0..count {
                    let hash = hash_from_nonce(black_box(&ctx), start + U512::from(offset));
                    assert!(black_box(hash) >= ctx.target);
                    hashes += 1;
                }
                assert_eq!(hashes, count);
            } else {
                let result = engine.search_range(
                    black_box(&ctx),
                    Range {
                        start,
                        end: start + U512::from(count - 1),
                    },
                    &cancel,
                );
                assert!(
                    matches!(result, EngineStatus::Exhausted { hash_count } if hash_count == count)
                );
            }
            if round > 0 {
                totals[index] += timer.elapsed().as_secs_f64();
            }
        }
    }
    let hashes = (count * 24) as f64;
    println!("reference: {:.4} MH/s", hashes / totals[0] / 1e6);
    println!("CPU engine: {:.4} MH/s", hashes / totals[1] / 1e6);
    println!("speedup: {:.3}%", (totals[0] / totals[1] - 1.0) * 100.0);
}

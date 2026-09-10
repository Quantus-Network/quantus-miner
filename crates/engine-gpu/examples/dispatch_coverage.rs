//! Check tail batches and high-half nonce carry against an exhaustive CPU search.
use engine_cpu::{AtomicBoolCancelCheck, EngineStatus, MinerEngine, Range};
use engine_gpu::GpuEngine;
use primitive_types::U512;
use std::sync::atomic::AtomicBool;

fn main() {
    let flag = AtomicBool::new(false);
    let cancel = AtomicBoolCancelCheck(&flag);
    for batch in [1, 31, 65, 257, 1000] {
        let engine = GpuEngine::try_new(batch, 0, false).expect("GPU required");
        for header in [0u8, 17, 255] {
            let mut ctx = engine.prepare_context([header; 32], U512::one());
            let start = (U512::one() << 256) - U512::from(100u32);
            let end = start + U512::from(512u32);
            let (hash, nonce) = (0..513u32)
                .map(|offset| {
                    let nonce = start + U512::from(offset);
                    (pow_core::hash_from_nonce(&ctx, nonce), nonce)
                })
                .min()
                .unwrap();
            // Only the CPU's minimum hash qualifies. Missing it exposes gaps
            // in dispatch coverage, unlike accepting any easy random solution.
            ctx.target = hash + U512::one();
            match engine.search_range(&ctx, Range { start, end }, &cancel) {
                EngineStatus::Found { candidate, .. } => {
                    assert_eq!(candidate.nonce, nonce);
                    assert_eq!(candidate.hash, hash);
                }
                other => panic!("batch {batch}: expected CPU minimum, got {other:?}"),
            }
            ctx.target = U512::zero();
            match engine.search_range(&ctx, Range { start, end }, &cancel) {
                EngineStatus::Exhausted { hash_count } => assert_eq!(hash_count, 513),
                other => panic!("batch {batch}: expected exhausted range, got {other:?}"),
            }
        }
        GpuEngine::clear_worker_resources();
    }
    println!("COVERAGE OK: 15 minimum-hash cases and 15 exhausted ranges");
}

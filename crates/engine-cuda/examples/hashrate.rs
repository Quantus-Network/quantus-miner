//! Offline CUDA benchmark. Fails if CUDA is absent; never connects to a node.
use engine_cpu::{AtomicBoolCancelCheck, EngineStatus, MinerEngine, Range};
use engine_cuda::CudaEngine;
use primitive_types::U512;
use std::{sync::atomic::AtomicBool, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let count: u32 = args
        .get(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(32_000_000);
    let runs: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(7);
    if count == 0 || runs == 0 {
        return Err("count and runs must be positive".into());
    }
    let engine = CudaEngine::try_new(count, 0)?;
    let header: [u8; 32] = hex::decode(pow_core::NONCE_HASH_KVS[0].header)?
        .try_into()
        .map_err(|_| "invalid golden header")?;
    let mut ctx = engine.prepare_context(header, U512::one());
    // Force an exhaustive search. CPU verification still rejects any prefix-equal candidate.
    ctx.target = U512::zero();
    let cancel = AtomicBool::new(false);
    let mut rates = Vec::new();
    const WARMUPS: usize = 64;
    for run in 0..runs + WARMUPS {
        let start = U512::from(0x1234_0000_0000_0000u64) + U512::from(run) * U512::from(count);
        let before = Instant::now();
        let status = engine.search_range(
            &ctx,
            Range {
                start,
                end: start + U512::from(count - 1),
            },
            &AtomicBoolCancelCheck(&cancel),
        );
        let elapsed = before.elapsed().as_secs_f64();
        match status {
            EngineStatus::Exhausted { hash_count } if hash_count == count as u64 => {}
            other => {
                return Err(
                    format!("expected exhaustive {count}-nonce search, got {other:?}").into(),
                )
            }
        }
        if run >= WARMUPS {
            let rate = count as f64 / elapsed / 1e6;
            println!(
                "run={},seconds={elapsed:.6},mh_s={rate:.6}",
                run - WARMUPS + 1
            );
            rates.push(rate);
        }
    }
    rates.sort_by(f64::total_cmp);
    println!(
        "shared_square={},median_mh_s={:.6}",
        cfg!(feature = "experimental-shared-square"),
        rates[rates.len() / 2]
    );
    CudaEngine::clear_worker_resources();
    Ok(())
}

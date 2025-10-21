#![deny(rust_2018_idioms)]
#![forbid(unsafe_code)]

//! CPU mining engine scaffolding and trait definition.
//!
//! This crate defines a minimal `MinerEngine` trait so the service layer can
//! orchestrate mining without knowing about specific math/device details,
//! plus a baseline CPU implementation that uses the reference path in `pow-core`.
//!
//! The baseline engine performs a straightforward linear scan across an inclusive
//! nonce range, computing distance per nonce using the context (header-derived)
//! constants. It is intended as a correctness reference and may be replaced at
//! runtime by faster engines (e.g., incremental CPU path, CUDA/OpenCL engines).

use core::cmp::Ordering;

use pow_core::{is_valid_nonce, is_valid_nonce_for_context, JobContext};
use primitive_types::U512;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::time::Duration;

/// An inclusive nonce range to search.
#[derive(Clone, Debug)]
pub struct Range {
    pub start: U512,
    pub end: U512, // inclusive
}

/// A winning candidate produced by an engine.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub nonce: U512,
    pub work: [u8; 64], // big-endian representation of nonce
    pub hash: U512,     // output hash for this nonce
}

/// Origin of a found candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FoundOrigin {
    Cpu,
    GpuG1,
    GpuG2,
    Unknown,
}

/// Status from an engine search.
///
/// For synchronous `search_range` calls, the final outcome will be one of:
/// - `Found`
/// - `Exhausted`
/// - `Cancelled`
///
/// `Running` is included for potential future async/streaming engines.
#[derive(Clone, Debug)]
pub enum EngineStatus {
    Running {
        hash_count: u64,
    },
    Found {
        candidate: Candidate,
        hash_count: u64,
        origin: FoundOrigin,
    },
    Exhausted {
        hash_count: u64,
    },
    Cancelled {
        hash_count: u64,
    },
}

/// Abstract mining engine interface.
///
/// The service layer depends only on this trait to manage jobs.
/// Different engines (baseline CPU, optimized CPU, CUDA, OpenCL) can implement
/// this trait and be selected at runtime via configuration.
pub trait MinerEngine: Send + Sync {
    /// Human-readable engine name (for logs/metrics).
    fn name(&self) -> &'static str;

    /// Prepare a precomputed context for a job (header + difficulty).
    fn prepare_context(&self, header_hash: [u8; 32], difficulty: U512) -> JobContext;

    /// Search an inclusive nonce range with cancellation support.
    ///
    /// Implementations should:
    /// - Respect `cancel` promptly to minimize wasted work after a solution is found.
    /// - Return `Found` with a `Candidate` on success.
    /// - Return `Exhausted` if the range is fully searched without a solution.
    /// - Return `Cancelled` if `cancel` was observed during the search.
    fn search_range(&self, ctx: &JobContext, range: Range, cancel: &AtomicBool) -> EngineStatus;
}

/// Baseline CPU engine.
///
/// This implementation uses the reference path from `pow-core` for distance
/// computation and scans the range linearly, one nonce at a time.
#[derive(Default)]
pub struct BaselineCpuEngine;

impl BaselineCpuEngine {
    pub fn new() -> Self {
        Self
    }
}

impl MinerEngine for BaselineCpuEngine {
    fn name(&self) -> &'static str {
        "cpu-baseline"
    }

    fn prepare_context(&self, header_hash: [u8; 32], difficulty: U512) -> JobContext {
        JobContext::new(header_hash, difficulty)
    }

    fn search_range(&self, ctx: &JobContext, range: Range, cancel: &AtomicBool) -> EngineStatus {
        // Ensure start <= end (inclusive range). If not, treat as exhausted.
        if range.start > range.end {
            return EngineStatus::Exhausted { hash_count: 0 };
        }

        let mut current = range.start;
        let mut hash_count: u64 = 0;

        loop {
            // Cancellation check
            if cancel.load(AtomicOrdering::Relaxed) {
                return EngineStatus::Cancelled { hash_count };
            }

            // Compute hash for this nonce using Bitcoin-style double Poseidon2
            let (is_valid, hash) = is_valid_nonce_for_context(ctx, current);
            hash_count = hash_count.saturating_add(1);

            // Check if it meets difficulty target
            if is_valid {
                let work = current.to_big_endian();
                let candidate = Candidate {
                    nonce: current,
                    work,
                    hash,
                };
                return EngineStatus::Found {
                    candidate,
                    hash_count,
                    origin: FoundOrigin::Cpu,
                };
            }

            // Advance or finish
            match current.cmp(&range.end) {
                Ordering::Less => {
                    current = current.saturating_add(U512::one());
                }
                _ => {
                    // End of inclusive range reached
                    break EngineStatus::Exhausted { hash_count };
                }
            }
        }
    }
}

// Re-export commonly used items for convenience by consumers.

// Fast CPU engine using incremental pow-core helpers (init_worker_y0 + step_mul)
#[derive(Default)]
pub struct FastCpuEngine;

impl FastCpuEngine {
    pub fn new() -> Self {
        Self
    }
}

impl MinerEngine for FastCpuEngine {
    fn name(&self) -> &'static str {
        "cpu-fast"
    }

    fn prepare_context(&self, header_hash: [u8; 32], difficulty: U512) -> JobContext {
        JobContext::new(header_hash, difficulty)
    }

    fn search_range(&self, ctx: &JobContext, range: Range, cancel: &AtomicBool) -> EngineStatus {
        use pow_core::{hash_from_nonce, is_valid_hash, step_nonce};

        // Ensure start <= end (inclusive range). If not, treat as exhausted.
        if range.start > range.end {
            return EngineStatus::Exhausted { hash_count: 0 };
        }

        let mut current = range.start;
        let mut hash_count: u64 = 0;

        loop {
            // Cancellation check
            if cancel.load(AtomicOrdering::Relaxed) {
                return EngineStatus::Cancelled { hash_count };
            }

            // Compute hash using Bitcoin-style double Poseidon2
            let hash = hash_from_nonce(ctx, current);
            hash_count = hash_count.saturating_add(1);

            if is_valid_hash(ctx, hash) {
                let work = current.to_big_endian();
                return EngineStatus::Found {
                    candidate: Candidate {
                        nonce: current,
                        work,
                        hash,
                    },
                    hash_count,
                    origin: FoundOrigin::Cpu,
                };
            }

            if current == range.end {
                break;
            }

            // Advance to next nonce
            current = step_nonce(current);
        }

        EngineStatus::Exhausted { hash_count }
    }
}

#[derive(Default)]
pub struct ChainManipulatorEngine {
    /// Base sleep per batch in nanoseconds; actual sleep increases linearly with job count.
    pub base_delay_ns: u64,
    /// Number of nonce attempts between sleeps.
    pub step_batch: u64,
    /// Optional cap for solved-block throttling (sleep index will not exceed this).
    pub throttle_cap: Option<u64>,
    /// Monotonically increasing solved-block counter used to scale throttling.
    /// Public so the service can initialize from CLI (pick up where we left off).
    pub job_index: AtomicU64,
}

impl ChainManipulatorEngine {
    pub fn new() -> Self {
        // Start fast: first block has no throttle (job_index = 0 -> 0ns sleep),
        // then 0.5ms per batch at block 1, 1.0ms at block 2, etc.
        Self {
            base_delay_ns: 500_000, // 0.5 ms
            step_batch: 10_000,     // sleep every 10k nonce checks
            throttle_cap: None,     // unlimited by default
            job_index: AtomicU64::new(0),
        }
    }
}

impl MinerEngine for ChainManipulatorEngine {
    fn name(&self) -> &'static str {
        "cpu-chain-manipulator"
    }

    fn prepare_context(&self, header_hash: [u8; 32], difficulty: U512) -> JobContext {
        // Per-block throttling: do NOT increment here. We increment on Found (i.e., when a block is solved).
        let ctx = JobContext::new(header_hash, difficulty);
        // Debug: log current throttle state at job start
        log::debug!(
            target: "miner",
            "manipulator throttle start: solved_blocks={}, sleep_ns_per_batch={}, step_batch={}",
            self.job_index.load(AtomicOrdering::Relaxed),
            self.base_delay_ns.saturating_mul(self.job_index.load(AtomicOrdering::Relaxed)),
            self.step_batch
        );
        ctx
    }

    fn search_range(&self, ctx: &JobContext, range: Range, cancel: &AtomicBool) -> EngineStatus {
        use pow_core::{hash_from_nonce, is_valid_hash, step_nonce};

        if range.start > range.end {
            return EngineStatus::Exhausted { hash_count: 0 };
        }

        // Bitcoin-style hashing path
        let mut current = range.start;
        let mut hash_count: u64 = 0;
        let mut batch_counter: u64 = 0;

        // Per-block throttling: derived from blocks solved so far (apply cap if configured).
        let mut solved_blocks = self.job_index.load(AtomicOrdering::Relaxed);
        if let Some(cap) = self.throttle_cap {
            if solved_blocks > cap {
                solved_blocks = cap;
            }
        }
        let sleep_ns = self.base_delay_ns.saturating_mul(solved_blocks);
        let do_sleep = sleep_ns > 0;

        loop {
            if cancel.load(AtomicOrdering::Relaxed) {
                return EngineStatus::Cancelled { hash_count };
            }

            // Compute hash using Bitcoin-style double Poseidon2
            let hash = hash_from_nonce(ctx, current);
            hash_count = hash_count.saturating_add(1);
            batch_counter = batch_counter.saturating_add(1);

            // Optional detailed debug of throttle progression within a job
            #[allow(unused_variables)]
            let _dbg_batch = batch_counter;

            if is_valid_hash(ctx, hash) {
                let work = current.to_big_endian();
                // Increment solved-block counter so the NEXT block throttles more.
                let _new_idx = self.job_index.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                {
                    let capped = if let Some(cap) = self.throttle_cap {
                        std::cmp::min(_new_idx, cap)
                    } else {
                        _new_idx
                    };
                    log::debug!(
                        target: "miner",
                        "manipulator throttle increment: solved_blocks={} (next sleep_ns_per_batch={}, cap={:?})",
                        _new_idx,
                        self.base_delay_ns.saturating_mul(capped),
                        self.throttle_cap
                    );
                }
                return EngineStatus::Found {
                    candidate: Candidate {
                        nonce: current,
                        work,
                        hash,
                    },
                    hash_count,
                    origin: FoundOrigin::Cpu,
                };
            }

            // Throttle after each batch to artificially slow down based on blocks solved so far.
            if do_sleep && batch_counter >= self.step_batch {
                std::thread::sleep(Duration::from_nanos(sleep_ns));
                batch_counter = 0;
            }

            // Advance
            if current < range.end {
                current = step_nonce(current);
            } else {
                break EngineStatus::Exhausted { hash_count };
            }
        }
    }
}

pub use {
    BaselineCpuEngine as DefaultEngine, Candidate as EngineCandidate,
    ChainManipulatorEngine as ChainEngine, FastCpuEngine as FastEngine, Range as EngineRange,
};

#[cfg(test)]
mod tests {
    use super::*;
    use primitive_types::U512;
    use std::sync::atomic::AtomicBool;

    fn make_ctx() -> JobContext {
        let header = [1u8; 32];
        let difficulty = U512::from(1u64); // easy difficulty for "found" parity test
        JobContext::new(header, difficulty)
    }

    #[test]
    fn baseline_and_fast_engines_find_same_candidate_on_small_range() {
        let ctx = make_ctx();

        let range = Range {
            start: U512::from(0u64),
            end: U512::from(100u64),
        };

        let cancel = AtomicBool::new(false);

        let baseline = BaselineCpuEngine::new();
        let fast = FastCpuEngine::new();

        let b_status = baseline.search_range(&ctx, range.clone(), &cancel);
        let f_status = fast.search_range(&ctx, range.clone(), &cancel);

        match (b_status, f_status) {
            (
                EngineStatus::Found {
                    candidate: b_cand,
                    hash_count: b_hashes,
                    origin: _,
                },
                EngineStatus::Found {
                    candidate: f_cand,
                    hash_count: f_hashes,
                    origin: _,
                },
            ) => {
                assert_eq!(
                    b_cand.nonce, f_cand.nonce,
                    "engines disagreed on winning nonce"
                );
                assert_eq!(
                    b_cand.distance, f_cand.distance,
                    "engines disagreed on distance"
                );
                assert_eq!(b_hashes, f_hashes, "engines disagreed on hash_count");
            }
            (b, f) => panic!("expected Found/Found, got baseline={b:?}, fast={f:?}"),
        }
    }

    #[test]
    fn engine_returns_exhausted_when_no_solution_in_range() {
        // Use a very hard difficulty to make solutions effectively impossible in a tiny range.
        let header = [2u8; 32];
        let difficulty = U512::MAX;
        let ctx = JobContext::new(header, difficulty);

        let range = Range {
            start: U512::from(1u64),
            end: U512::from(1000u64), // small range; probability of accidental match is negligible
        };

        let cancel = AtomicBool::new(false);
        let baseline = BaselineCpuEngine::new();

        let status = baseline.search_range(&ctx, range.clone(), &cancel);
        match status {
            EngineStatus::Exhausted { hash_count } => {
                // Inclusive range length = end - start + 1
                let expected = (range.end - range.start + U512::one()).as_u64();
                assert_eq!(hash_count, expected, "hash_count should equal range length");
            }
            other => panic!("expected Exhausted, got {other:?}"),
        }
    }

    #[test]
    fn engine_respects_immediate_cancellation() {
        let ctx = make_ctx();
        let range = Range {
            start: U512::from(0u64),
            end: U512::from(1_000_000u64),
        };
        let cancel = AtomicBool::new(true); // cancelled before starting
        let baseline = BaselineCpuEngine::new();

        let status = baseline.search_range(&ctx, range, &cancel);
        match status {
            EngineStatus::Cancelled { hash_count } => {
                // Cancellation was pre-set; allow zero or near-zero work depending on timing.
                assert_eq!(hash_count, 0, "expected no work when cancelled immediately");
            }
            other => panic!("expected Cancelled, got {other:?}"),
        }
    }
}

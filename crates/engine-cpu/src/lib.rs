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

use pow_core::{distance_for_nonce, is_valid_distance, JobContext};
use primitive_types::U512;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

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
    pub distance: U512, // achieved distance for this nonce
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
    Running { hash_count: u64 },
    Found(Candidate),
    Exhausted { hash_count: u64 },
    Cancelled { hash_count: u64 },
}

/// Abstract mining engine interface.
///
/// The service layer depends only on this trait to manage jobs.
/// Different engines (baseline CPU, optimized CPU, CUDA, OpenCL) can implement
/// this trait and be selected at runtime via configuration.
pub trait MinerEngine: Send + Sync {
    /// Human-readable engine name (for logs/metrics).
    fn name(&self) -> &'static str;

    /// Prepare a precomputed context for a job (header + threshold).
    fn prepare_context(&self, header_hash: [u8; 32], threshold: U512) -> JobContext;

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

    fn prepare_context(&self, header_hash: [u8; 32], threshold: U512) -> JobContext {
        JobContext::new(header_hash, threshold)
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

            // Compute distance for this nonce using the context.
            let distance = distance_for_nonce(ctx, current);
            hash_count = hash_count.saturating_add(1);

            // Check if it's valid under threshold.
            if is_valid_distance(ctx, distance) {
                let work = current.to_big_endian();
                let candidate = Candidate {
                    nonce: current,
                    work,
                    distance,
                };
                return EngineStatus::Found(candidate);
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

    fn prepare_context(&self, header_hash: [u8; 32], threshold: U512) -> JobContext {
        JobContext::new(header_hash, threshold)
    }

    fn search_range(&self, ctx: &JobContext, range: Range, cancel: &AtomicBool) -> EngineStatus {
        use pow_core::{distance_from_y, init_worker_y0, is_valid_distance, step_mul};

        // Ensure start <= end (inclusive range). If not, treat as exhausted.
        if range.start > range.end {
            return EngineStatus::Exhausted { hash_count: 0 };
        }

        let mut current = range.start;
        let mut y = init_worker_y0(ctx, current);
        let mut hash_count: u64 = 0;

        loop {
            // Cancellation check
            if cancel.load(AtomicOrdering::Relaxed) {
                return EngineStatus::Cancelled { hash_count };
            }

            // Compute distance from current accumulator
            let distance = distance_from_y(ctx, y);
            hash_count = hash_count.saturating_add(1);

            if is_valid_distance(ctx, distance) {
                let work = current.to_big_endian();
                return EngineStatus::Found(Candidate {
                    nonce: current,
                    work,
                    distance,
                });
            }

            if current == range.end {
                break EngineStatus::Exhausted { hash_count };
            }

            // Advance to next nonce: y <- y * m (mod n), current <- current + 1
            y = step_mul(ctx, y);
            current = current.saturating_add(U512::one());
        }
    }
}

pub use {
    BaselineCpuEngine as DefaultEngine, Candidate as EngineCandidate, FastCpuEngine as FastEngine,
    Range as EngineRange,
};

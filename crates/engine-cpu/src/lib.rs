#![deny(rust_2018_idioms)]
#![forbid(unsafe_code)]

//! CPU mining engine for Quantus.
//!
//! This crate defines the `MinerEngine` trait for mining orchestration,
//! plus a fast CPU implementation using optimized paths from `pow-core`.

use pow_core::JobContext;
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
/// Different engines (CPU, CUDA, OpenCL) can implement this trait.
pub trait MinerEngine: Send + Sync {
    /// Human-readable engine name (for logs).
    fn name(&self) -> &'static str;

    /// Prepare a precomputed context for a job (header + difficulty).
    fn prepare_context(&self, header_hash: [u8; 32], difficulty: U512) -> JobContext;

    /// Search an inclusive nonce range with cancellation support.
    fn search_range(&self, ctx: &JobContext, range: Range, cancel: &AtomicBool) -> EngineStatus;

    /// Enable downcasting to concrete engine types.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Fast CPU engine using optimized pow-core helpers.
#[derive(Default)]
pub struct FastCpuEngine;

impl FastCpuEngine {
    pub fn new() -> Self {
        Self
    }
}

impl MinerEngine for FastCpuEngine {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn prepare_context(&self, header_hash: [u8; 32], difficulty: U512) -> JobContext {
        JobContext::new(header_hash, difficulty)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn search_range(&self, ctx: &JobContext, range: Range, cancel: &AtomicBool) -> EngineStatus {
        use pow_core::{hash_from_nonce, is_valid_hash, step_nonce};

        if range.start > range.end {
            return EngineStatus::Exhausted { hash_count: 0 };
        }

        let mut current = range.start;
        let mut hash_count: u64 = 0;

        loop {
            if cancel.load(AtomicOrdering::Relaxed) {
                return EngineStatus::Cancelled { hash_count };
            }

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

            current = step_nonce(current);
        }

        EngineStatus::Exhausted { hash_count }
    }
}

// Re-exports for convenience
pub use {Candidate as EngineCandidate, Range as EngineRange};

#[cfg(test)]
mod tests {
    use super::*;
    use primitive_types::U512;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn engine_returns_exhausted_when_no_solution_in_range() {
        let header = [2u8; 32];
        let difficulty = U512::MAX; // impossible difficulty
        let ctx = JobContext::new(header, difficulty);

        let range = Range {
            start: U512::from(1u64),
            end: U512::from(1000u64),
        };

        let cancel = AtomicBool::new(false);
        let engine = FastCpuEngine::new();

        let status = engine.search_range(&ctx, range.clone(), &cancel);
        match status {
            EngineStatus::Exhausted { hash_count } => {
                let expected = (range.end - range.start + U512::one()).as_u64();
                assert_eq!(hash_count, expected);
            }
            other => panic!("expected Exhausted, got {other:?}"),
        }
    }

    #[test]
    fn engine_respects_immediate_cancellation() {
        let header = [1u8; 32];
        let difficulty = U512::from(1u64);
        let ctx = JobContext::new(header, difficulty);

        let range = Range {
            start: U512::from(0u64),
            end: U512::from(1_000_000u64),
        };

        let cancel = AtomicBool::new(true); // pre-cancelled
        let engine = FastCpuEngine::new();

        let status = engine.search_range(&ctx, range, &cancel);
        match status {
            EngineStatus::Cancelled { hash_count } => {
                assert_eq!(hash_count, 0);
            }
            other => panic!("expected Cancelled, got {other:?}"),
        }
    }
}

#![deny(rust_2018_idioms)]
#![forbid(unsafe_code)]

//! CUDA-based GPU mining engine (placeholder)
//!
//! This crate is a scaffold for a future CUDA backend that will implement the
//! mining engine interface used by the service layer. It currently provides:
//! - A `CudaEngine` type with a constructor and basic helpers.
//! - Documentation of the intended integration points.
//!
//! Planned responsibilities (non-exhaustive):
//! - Accept a prepared `JobContext` (from `pow-core`) per job.
//! - Partition nonce ranges into GPU work assignments.
//! - Run a CUDA kernel that performs, per nonce in the range:
//!     - y <- y * m (mod n) using Montgomery multiplication (in Montgomery domain)
//!     - nonce_element <- SHA3_512(y) in the normal domain
//!     - distance <- target XOR nonce_element
//!     - if distance <= threshold: report solution and signal early-cancel
//! - Coordinate early-exit via device-global flags and host polling.
//!
//! Notes:
//! - This crate deliberately does NOT implement the `MinerEngine` trait yet,
//!   because the engine trait currently lives in `engine-cpu`. Once the trait
//!   is promoted to a shared crate (or re-exported for engines), this crate
//!   will implement it and become selectable at runtime via the service config.
//! - CUDA bindings (e.g., via `cust`/`rustacuda`) and kernels will be added
//!   behind feature flags (e.g., `cuda`). For now, we only offer placeholders
//!   so the workspace compiles cleanly and the integration points are clear.

use pow_core::JobContext;
use primitive_types::U512;

/// Placeholder type for the CUDA engine.
///
/// When fully implemented, this engine will manage CUDA device/context
/// initialization, kernel module loading, memory transfers, and kernel launches.
/// It will expose the same search range semantics as the CPU engine(s) but
/// backed by the GPU.
#[derive(Default, Debug)]
pub struct CudaEngine {
    // Future fields (examples):
    // device_id: usize,
    // context: cust::context::Context,
    // module: cust::module::Module,
    // stream: cust::stream::Stream,
}

impl CudaEngine {
    /// Construct a new CUDA engine placeholder.
    ///
    /// Future versions may accept configuration (e.g., device index, module path).
    pub fn new() -> Self {
        Self::default()
    }

    /// Human-readable name for logs/metrics.
    pub fn name(&self) -> &'static str {
        "gpu-cuda (placeholder)"
    }

    /// Prepare a precomputed job context for a given header and threshold.
    ///
    /// This defers to `pow-core` to derive (m, n) and `target` from the header.
    /// In a full CUDA implementation, this context will be uploaded to device
    /// constant memory or passed as kernel parameters.
    pub fn prepare_context(&self, header_hash: [u8; 32], threshold: U512) -> JobContext {
        JobContext::new(header_hash, threshold)
    }

    /// Returns whether this build has CUDA support compiled in.
    ///
    /// When actual CUDA integration is added behind a feature flag, this will
    /// return true only if that feature is enabled.
    pub fn cuda_available(&self) -> bool {
        // Adjust once actual CUDA integration is implemented behind a feature:
        // cfg!(feature = "cuda")
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use primitive_types::U512;

    #[test]
    fn placeholder_engine_basics() {
        let eng = CudaEngine::new();
        assert_eq!(eng.name(), "gpu-cuda (placeholder)");

        // Ensure context creation works and is deterministic in shape.
        let header = [1u8; 32];
        let threshold = U512::from(12345u64);
        let ctx = eng.prepare_context(header, threshold);

        assert_eq!(ctx.header, header);
        assert_eq!(ctx.threshold, threshold);
    }
}

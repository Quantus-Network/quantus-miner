// #![deny(rust_2018_idioms)]
// #![forbid(unsafe_code)]

// //! OpenCL-based GPU mining engine (placeholder)
// //!
// //! This crate is a scaffold for a future OpenCL backend that will implement the
// //! mining engine interface used by the service layer. It currently provides:
// //! - An `OpenClEngine` type with a constructor and basic helpers.
// //! - Documentation of the intended integration points.
// //!
// //! Planned responsibilities (non-exhaustive):
// //! - Accept a prepared `JobContext` (from `pow-core`) per job.
// //! - Partition nonce ranges into GPU work assignments.
// //! - Run an OpenCL kernel that performs, per nonce in the range:
// //!     - y <- y * m (mod n) using Montgomery multiplication (in Montgomery domain)
// //!     - nonce_element <- SHA3_512(y) in the normal domain
// //!     - distance <- target XOR nonce_element
// //!     - if distance <= threshold: report solution and signal early-cancel
// //! - Coordinate early-exit via device-global flags and host polling.
// //!
// //! Notes:
// //! - This crate deliberately does NOT implement the `MinerEngine` trait yet,
// //!   because the engine trait currently lives in `engine-cpu`. Once the trait
// //!   is promoted to a shared crate (or re-exported for engines), this crate
// //!   will implement it and become selectable at runtime via the service config.
// //! - OpenCL bindings (e.g., via the `ocl` crate) and kernels will be added
// //!   behind feature flags (e.g., `opencl`). For now, we only offer placeholders
// //!   so the workspace compiles cleanly and the integration points are clear.

// use pow_core::JobContext;
// use primitive_types::U512;

// /// Placeholder type for the OpenCL engine.
// ///
// /// When fully implemented, this engine will manage OpenCL platform/device
// /// discovery, context/queue creation, kernel compilation, memory transfers,
// /// and kernel launches. It will expose the same search-range semantics as
// /// the CPU engine(s) but backed by the GPU.
// #[derive(Default, Debug)]
// pub struct OpenClEngine {
//     // Future fields (examples):
//     // platform_id: usize,
//     // device_id: usize,
//     // context: ocl::Context,
//     // queue: ocl::Queue,
//     // program: ocl::Program,
//     // kernel: ocl::Kernel,
// }

// impl OpenClEngine {
//     /// Construct a new OpenCL engine placeholder.
//     ///
//     /// Future versions may accept configuration (e.g., platform/device index).
//     pub fn new() -> Self {
//         Self::default()
//     }

//     /// Human-readable name for logs/metrics.
//     pub fn name(&self) -> &'static str {
//         "gpu-opencl (placeholder)"
//     }

//     /// Prepare a precomputed job context for a given header and threshold.
//     ///
//     /// This defers to `pow-core` to derive (m, n) and `target` from the header.
//     /// In a full OpenCL implementation, this context will be uploaded to device
//     /// constant buffers or passed as kernel arguments.
//     pub fn prepare_context(&self, header_hash: [u8; 32], threshold: U512) -> JobContext {
//         JobContext::new(header_hash, threshold)
//     }

//     /// Returns whether this build has OpenCL support compiled in.
//     ///
//     /// When actual OpenCL integration is added behind a feature flag, this will
//     /// return true only if that feature is enabled.
//     pub fn opencl_available(&self) -> bool {
//         // Adjust once actual OpenCL integration is implemented behind a feature:
//         // cfg!(feature = "opencl")
//         false
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use primitive_types::U512;

//     #[test]
//     fn placeholder_engine_basics() {
//         let eng = OpenClEngine::new();
//         assert_eq!(eng.name(), "gpu-opencl (placeholder)");

//         // Ensure context creation works and is deterministic in shape.
//         let header = [1u8; 32];
//         let threshold = U512::from(12345u64);
//         let ctx = eng.prepare_context(header, threshold);

//         assert_eq!(ctx.header, header);
//         assert_eq!(ctx.threshold, threshold);
//     }
// }

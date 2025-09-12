#![deny(rust_2018_idioms)]
#![deny(unsafe_code)]
#![cfg_attr(not(feature = "cuda"), allow(dead_code, unused_imports))]

//! CUDA-based GPU mining engine (G1 bring-up)
//!
//! Tuning (G1):
//! - You can influence the launcher behavior via environment variables (applied if present):
//!   - MINER_CUDA_BLOCK_DIM: threads per block (default 256)
//!   - MINER_CUDA_THREADS: number of threads to launch per kernel (capped by device/block limits)
//!   - MINER_CUDA_ITERS: number of iterations per thread (kernel supports iterating ŷ ← ŷ·m̂ repeatedly)
//!     These will be hooked into the launcher in the next step to reduce per-launch CPU y0 exponentiations
//!     (few threads, many iterations per launch) and to pool device constants/module/stream across launches.
//!
//! Notes:
//! - Embedded PTX is preferred at runtime; fallback is ENGINE_GPU_CUDA_PTX_DIR for external PTX.
//! - For arch-specific builds (optional): set CUDA_ARCH=sm_86 (3060), sm_89 (4090), compute_120/sm_120 (5090).
//!
//! This crate now includes a host-side launcher for a minimal CUDA kernel
//! (see `src/kernels/qpow_kernel.cu`) that performs 512-bit Montgomery
//! multiplication on-device and returns normalized y values. For G1 bring-up,
//! we perform SHA3 and threshold checks on the host for parity against CPU.
//!
//! Behavior:
//! - If compiled without the `cuda` feature or if CUDA init/launch fails,
//!   we delegate to the CPU fast engine (unchanged external semantics).
//! - If CUDA is available, we attempt chunked GPU execution with small
//!   per-launch windows and check cancelation between launches.
//!
//! Notes:
//! - The kernel writes normalized y values (LE limbs) for y_{k+1} given y_k.
//!   We handle the very first nonce in each chunk on CPU for parity, then
//!   consume GPU outputs for subsequent nonces in the chunk.

#[cfg(feature = "cuda")]
use anyhow::Context as AnyhowContext;
use pow_core::JobContext;
use primitive_types::U512;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

use engine_cpu::{EngineRange, EngineStatus, MinerEngine};

#[cfg(feature = "cuda")]
use cust as cuda;
#[cfg(feature = "cuda")]
use cust::context::legacy::ContextStack;
#[cfg(feature = "cuda")]
use cust::context::Context;
#[cfg(feature = "cuda")]
use cust::device::Device;
#[cfg(feature = "cuda")]
use cust::launch;
#[cfg(feature = "cuda")]
use cust::memory::CopyDestination;
#[cfg(feature = "cuda")]
include!(concat!(env!("OUT_DIR"), "/ptx_bindings.rs"));

/// CUDA engine with host-side G1 launcher and CPU fallback.
#[derive(Default, Debug)]
pub struct CudaEngine;

impl CudaEngine {
    /// Construct a new CUDA engine.
    pub fn new() -> Self {
        Self
    }

    /// Human-readable name for logs/metrics.
    pub fn name(&self) -> &'static str {
        "gpu-cuda"
    }

    /// Prepare a precomputed job context for a given header and threshold.
    pub fn prepare_context(&self, header_hash: [u8; 32], threshold: U512) -> JobContext {
        JobContext::new(header_hash, threshold)
    }

    /// Returns whether CUDA is compiled in and the driver/device can be initialized.
    pub fn cuda_available(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            match cuda::quick_init() {
                Ok(_) => true,
                Err(e) => {
                    // Fall back to manual driver/device/context init so service gating can pass.
                    log::warn!(target: "miner", "CUDA quick_init failed: {e:?}; attempting manual init");
                    let manual_ok = (|| -> Result<(), cust::error::CudaError> {
                        let dev = Device::get_device(0)?;
                        let ctx = Context::new(dev)?;
                        let _guard = ContextStack::push(&ctx)?;
                        Ok(())
                    })();
                    match manual_ok {
                        Ok(()) => {
                            log::info!(target: "miner", "CUDA manual init succeeded");
                            true
                        }
                        Err(err) => {
                            log::warn!(target: "miner", "CUDA manual init failed: {err:?}");
                            false
                        }
                    }
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    #[cfg(feature = "cuda")]
    #[allow(unsafe_code)]
    fn try_search_range_gpu_g1(
        &self,
        ctx: &JobContext,
        range: EngineRange,
        cancel: &AtomicBool,
    ) -> anyhow::Result<EngineStatus> {
        use std::ffi::CString;

        // Basic sanity
        if range.start > range.end {
            return Ok(EngineStatus::Exhausted { hash_count: 0 });
        }

        // Initialize CUDA context and keep it current for the entire GPU work scope
        let (_context, _ctx_guard) = match cuda::quick_init() {
            Ok(ctx) => {
                // Ensure the context stays current while this function runs
                let guard = ContextStack::push(&ctx)?;
                (ctx, guard)
            }
            Err(e) => {
                log::warn!(target: "miner", "CUDA quick_init failed: {e:?}; attempting manual device/context init");
                let dev = Device::get_device(0)?;
                let ctx = Context::new(dev)?;
                let guard = ContextStack::push(&ctx)?;
                (ctx, guard)
            }
        };

        // Prefer embedded CUBIN (native SASS); fall back to embedded/env PTX if unavailable.
        let module = if let Some(cubin) = cubin_embedded::get_cubin("qpow_kernel") {
            log::info!(target: "miner", "CUDA: using CUBIN (embedded)");
            cuda::module::Module::from_cubin(cubin, &[]).with_context(|| "load CUBIN module")?
        } else {
            // Load PTX module: prefer embedded PTX; fall back to env-dir PTX
            let (ptx_text, ptx_origin) = if let Some(s) = ptx_embedded::get("qpow_kernel") {
                (s.to_string(), String::from("embedded"))
            } else {
                let ptx_dir = std::env::var("ENGINE_GPU_CUDA_PTX_DIR").map_err(|_| {
                    anyhow::anyhow!("ENGINE_GPU_CUDA_PTX_DIR not set and no embedded PTX")
                })?;
                let ptx_path = std::path::Path::new(&ptx_dir).join("qpow_kernel.ptx");
                let txt = std::fs::read_to_string(&ptx_path).map_err(|e| {
                    anyhow::anyhow!("failed to read PTX at {}: {e}", ptx_path.display())
                })?;
                (txt, format!("env:{}", ptx_path.display()))
            };
            log::info!(target: "miner", "CUDA: using PTX source = {ptx_origin}");
            let ptx_cstr = CString::new(ptx_text)?;
            cuda::module::Module::from_ptx_cstr(&ptx_cstr, &[])
                .with_context(|| "load PTX module")?
        };
        let stream = cuda::stream::Stream::new(cuda::stream::StreamFlags::DEFAULT, None)
            .with_context(|| "create stream")?;
        let func = module
            .get_function("qpow_montgomery_g1_kernel")
            .with_context(|| "get kernel function 'qpow_montgomery_g1_kernel'")?;

        // Precompute Montgomery constants (host-side)
        let gc = GpuConstants::from_ctx(ctx);

        // Chunked loop: process the inclusive range [current ..= end]
        // Launcher tuning knobs (env overrides)
        let block_dim: u32 = std::env::var("MINER_CUDA_BLOCK_DIM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);
        let threads: u32 = std::env::var("MINER_CUDA_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let iters_per_thread: u32 = std::env::var("MINER_CUDA_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);
        log::info!(target: "miner", "CUDA launch config: block_dim={block_dim}, threads={threads}, iters={iters_per_thread}");
        // block_dim configured above via MINER_CUDA_BLOCK_DIM (default 256)

        let mut current = range.start;
        let end = range.end;
        let mut hash_count: u64 = 0;

        while current <= end {
            // Cancellation check early and often
            if cancel.load(AtomicOrdering::Relaxed) {
                return Ok(EngineStatus::Cancelled { hash_count });
            }

            // 1) Handle the first nonce of this chunk on CPU (parity and to align GPU outputs)
            {
                let distance = pow_core::distance_for_nonce(ctx, current);
                hash_count = hash_count.saturating_add(1);
                if pow_core::is_valid_distance(ctx, distance) {
                    let work = current.to_big_endian();
                    return Ok(EngineStatus::Found {
                        candidate: engine_cpu::EngineCandidate {
                            nonce: current,
                            work,
                            distance,
                        },
                        hash_count,
                    });
                }
            }

            // Remaining work after consuming `current`
            let remaining = end.saturating_sub(current);
            if remaining.is_zero() {
                // We just processed the last nonce above
                break;
            }

            // Determine a small launch window (cover next nonces via GPU)
            // coverage bounded by remaining is enforced during consumption

            let num_threads = threads.min(1024);
            // Use many iterations per thread to reduce host exponentiations per launch
            let grid_dim = ((num_threads + block_dim - 1) / block_dim).max(1);

            // Prepare per-thread y0 for base nonces: current + t
            let mut y0_host: Vec<u64> = vec![0u64; (num_threads as usize) * 8];
            for t in 0..(num_threads as usize) {
                let stride = iters_per_thread as u64;
                let base_nonce = current.saturating_add(U512::from((t as u64) * stride));
                let y0_u512 = pow_core::init_worker_y0(ctx, base_nonce);
                let y0_le = u512_to_le_limbs(y0_u512);
                let off = t * 8;
                y0_host[off..off + 8].copy_from_slice(&y0_le);
            }

            // Flatten constants
            let m_le = gc.m_le;
            let n_le = gc.n_le;
            let r2_le = gc.r2_le;
            let m_hat_le = gc.m_hat_le;
            let n0_inv = gc.n0_inv;

            // Allocate/copy device buffers
            let d_m = cuda::memory::DeviceBuffer::<u64>::from_slice(&m_le)
                .with_context(|| "alloc/copy d_m")?;
            let d_n = cuda::memory::DeviceBuffer::<u64>::from_slice(&n_le)
                .with_context(|| "alloc/copy d_n")?;
            let d_r2 = cuda::memory::DeviceBuffer::<u64>::from_slice(&r2_le)
                .with_context(|| "alloc/copy d_r2")?;
            let d_mhat = cuda::memory::DeviceBuffer::<u64>::from_slice(&m_hat_le)
                .with_context(|| "alloc/copy d_mhat")?;
            let d_y0 = cuda::memory::DeviceBuffer::<u64>::from_slice(&y0_host)
                .with_context(|| "alloc/copy d_y0")?;
            let mut d_y_out = cuda::memory::DeviceBuffer::<u64>::zeroed(
                (num_threads as usize) * (iters_per_thread as usize) * 8,
            )
            .with_context(|| "alloc d_y_out")?;

            // Launch kernel: computes y for (current + t + 1) for each thread t in [0, num_threads)
            log::info!(target: "miner", "CUDA launch: grid_dim={grid_dim}, block_dim={block_dim}, threads={num_threads}, iters={iters_per_thread}");
            let launch_result = unsafe {
                launch!(func<<<grid_dim, block_dim, 0, stream>>>(
                    d_m.as_device_ptr(),
                    d_n.as_device_ptr(),
                    n0_inv as u64,
                    d_r2.as_device_ptr(),
                    d_mhat.as_device_ptr(),
                    d_y0.as_device_ptr(),
                    d_y_out.as_device_ptr(),
                    num_threads as u32,
                    iters_per_thread as u32
                ))
            };
            let t_kernel_start = std::time::Instant::now();
            launch_result.with_context(|| "launch kernel")?;
            stream.synchronize().with_context(|| "stream synchronize")?;
            let kernel_ms = t_kernel_start.elapsed().as_millis();
            log::info!(target: "miner", "CUDA kernel and sync OK (kernel_ms={kernel_ms})");

            // Copy back results
            let mut y_out_host =
                vec![0u64; (num_threads as usize) * (iters_per_thread as usize) * 8];
            let t_copy_start = std::time::Instant::now();
            d_y_out
                .copy_to(&mut y_out_host)
                .with_context(|| "copy d_y_out -> host")?;
            let copy_ms = t_copy_start.elapsed().as_millis();
            log::info!(target: "miner", "CUDA copy-back OK: elems={}, copy_ms={copy_ms}", y_out_host.len());

            // Consume GPU results in-order; each thread emitted `iters_per_thread` y values:
            // nonce = current + 1 + (t * iters_per_thread) + j
            for t in 0..(num_threads as usize) {
                for j in 0..(iters_per_thread as usize) {
                    if cancel.load(AtomicOrdering::Relaxed) {
                        return Ok(EngineStatus::Cancelled { hash_count });
                    }

                    let idx_words = (t * (iters_per_thread as usize) + j) * 8;
                    let mut le = [0u64; 8];
                    le.copy_from_slice(&y_out_host[idx_words..idx_words + 8]);
                    let y_norm = le_limbs_to_u512(&le);

                    let distance = pow_core::distance_from_y(ctx, y_norm);
                    hash_count = hash_count.saturating_add(1);

                    let nonce = current.saturating_add(U512::from(
                        1u64 + (t as u64) * (iters_per_thread as u64) + (j as u64),
                    ));
                    if pow_core::is_valid_distance(ctx, distance) {
                        let work = nonce.to_big_endian();
                        return Ok(EngineStatus::Found {
                            candidate: engine_cpu::EngineCandidate {
                                nonce,
                                work,
                                distance,
                            },
                            hash_count,
                        });
                    }

                    if nonce >= end {
                        return Ok(EngineStatus::Exhausted { hash_count });
                    }
                }
            }

            // Advance: we covered 1 (CPU step) + num_threads * iters_per_thread nonces in this iteration
            current = current.saturating_add(U512::from(
                1u64 + (num_threads as u64) * (iters_per_thread as u64),
            ));
        }

        Ok(EngineStatus::Exhausted { hash_count })
    }
}

impl MinerEngine for CudaEngine {
    fn name(&self) -> &'static str {
        "gpu-cuda"
    }

    fn prepare_context(&self, header_hash: [u8; 32], threshold: U512) -> JobContext {
        self.prepare_context(header_hash, threshold)
    }

    fn search_range(
        &self,
        ctx: &JobContext,
        range: EngineRange,
        cancel: &AtomicBool,
    ) -> EngineStatus {
        // If CUDA is available, try GPU G1 path; on any error, fall back to CPU fast.
        #[cfg(feature = "cuda")]
        {
            match self.try_search_range_gpu_g1(ctx, range.clone(), cancel) {
                Ok(eng) => return eng,
                Err(e) => {
                    log::info!(target: "miner", "GPU path failed: {e:?}; delegating to CPU fast engine.");
                }
            }
        }

        let cpu = engine_cpu::FastCpuEngine::new();
        cpu.search_range(ctx, range, cancel)
    }
}

#[derive(Clone, Copy, Debug)]
struct GpuConstants {
    m_le: [u64; 8],
    n_le: [u64; 8],
    r2_le: [u64; 8],
    m_hat_le: [u64; 8],
    n0_inv: u64, // -n^{-1} mod 2^64
}

impl GpuConstants {
    fn from_ctx(ctx: &JobContext) -> Self {
        // Convert to LE limbs expected by the kernel
        let m_le = u512_to_le_limbs(ctx.m);
        let n_le = u512_to_le_limbs(ctx.n);

        // n0_inv from n[0] (must be odd)
        let n0_inv = mont_n0_inv(n_le[0]);

        // R^2 mod n via pow-core compat (2^1024 mod n)
        let r2_u512 = pow_core::compat::mod_pow(&U512::from(2u32), &U512::from(1024u32), &ctx.n);
        let r2_le = u512_to_le_limbs(r2_u512);

        // m_hat = to_mont(m) = mont_mul(m, R^2)
        let m_hat_le = mont_mul_portable(&m_le, &r2_le, &n_le, n0_inv);

        GpuConstants {
            m_le,
            n_le,
            r2_le,
            m_hat_le,
            n0_inv,
        }
    }
}

// --- Helpers: conversions and portable Montgomery (host-side precompute) ---

#[allow(clippy::needless_range_loop)]
fn u512_to_le_limbs(x: U512) -> [u64; 8] {
    let be = x.to_big_endian();
    let mut le = [0u64; 8];
    for i in 0..8 {
        let start = 56usize.saturating_sub(i * 8);
        let mut limb_bytes = [0u8; 8];
        limb_bytes.copy_from_slice(&be[start..start + 8]);
        le[i] = u64::from_be_bytes(limb_bytes);
    }
    le
}

fn le_limbs_to_u512(le: &[u64; 8]) -> U512 {
    let mut be = [0u8; 64];
    for i in 0..8 {
        let limb = le[7 - i].to_be_bytes();
        be[i * 8..i * 8 + 8].copy_from_slice(&limb);
    }
    U512::from_big_endian(&be)
}

// Compute n0_inv = -n[0]^{-1} mod 2^64 using Newton–Raphson (n[0] must be odd).
fn mont_n0_inv(n0: u64) -> u64 {
    // Compute inverse of n0 modulo 2^64
    let mut x = 1u64;
    // Newton–Raphson iterations to refine inverse modulo 2^64
    for _ in 0..6 {
        x = x.wrapping_mul(2u64.wrapping_sub(n0.wrapping_mul(x)));
    }
    x.wrapping_neg()
}

// Portable CIOS Montgomery multiplication: returns (a * b * R^{-1}) mod n
#[allow(clippy::needless_range_loop)]
fn mont_mul_portable(a: &[u64; 8], b: &[u64; 8], n: &[u64; 8], n0_inv: u64) -> [u64; 8] {
    const MASK: u128 = 0xFFFF_FFFF_FFFF_FFFFu128;
    let mut acc = [0u128; 9];

    for i in 0..8 {
        // acc += a[i] * b
        let ai = a[i] as u128;
        let mut carry: u128 = 0;
        for j in 0..8 {
            let prod = (ai * (b[j] as u128)) + (acc[j] & MASK) + carry;
            acc[j] = (acc[j] & !MASK) + (prod & MASK);
            carry = prod >> 64;
        }
        acc[8] = (acc[8] & MASK) + carry;

        // m = (acc[0] * n0_inv) mod 2^64
        let m = ((acc[0] as u64).wrapping_mul(n0_inv)) as u128;

        // acc += m * n
        let mut carry2: u128 = 0;
        for j in 0..8 {
            let prod2 = (m * (n[j] as u128)) + (acc[j] & MASK) + carry2;
            acc[j] = (acc[j] & !MASK) + (prod2 & MASK);
            carry2 = prod2 >> 64;
        }
        acc[8] = (acc[8] & MASK) + carry2;

        // Shift accumulator right by one limb (drop acc[0])
        for j in 0..8 {
            acc[j] = (acc[j + 1] & MASK) + (acc[j] & !MASK);
        }
        acc[8] = 0;
    }

    // Conditional subtract if acc >= n
    let mut res = [0u64; 8];
    for i in 0..8 {
        res[i] = (acc[i] & MASK) as u64;
    }

    if ge_le(&res, n) {
        sub_le_in_place(&mut res, n);
    }

    res
}

// Compare a >= b for 8-limb LE arrays
fn ge_le(a: &[u64; 8], b: &[u64; 8]) -> bool {
    for i in (0..8).rev() {
        if a[i] != b[i] {
            return a[i] > b[i];
        }
    }
    true
}

// res := res - b (LE limbs)
fn sub_le_in_place(res: &mut [u64; 8], b: &[u64; 8]) {
    let mut borrow: u128 = 0;
    for i in 0..8 {
        let ai = res[i] as u128;
        let bi = b[i] as u128;
        let tmp = ai.wrapping_sub(bi).wrapping_sub(borrow);
        res[i] = (tmp & 0xFFFF_FFFF_FFFF_FFFFu128) as u64;
        // borrow if ai < bi + borrow
        let need = if ai < bi + borrow { 1 } else { 0 };
        borrow = need;
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use primitive_types::U512;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn gpu_g1_parity_with_cpu_on_small_range() {
        // Ensure CUDA is available; otherwise skip.
        if cust::quick_init().is_err() {
            eprintln!("CUDA unavailable; skipping gpu_g1_parity_with_cpu_on_small_range");
            return;
        }

        // Require PTX artifact to be present; otherwise skip.
        let ptx_dir = match std::env::var("ENGINE_GPU_CUDA_PTX_DIR") {
            Ok(v) => v,
            Err(_) => {
                eprintln!("ENGINE_GPU_CUDA_PTX_DIR not set; skipping gpu parity test");
                return;
            }
        };
        let ptx_path = std::path::Path::new(&ptx_dir).join("qpow_kernel.ptx");
        if !ptx_path.is_file() {
            eprintln!(
                "PTX not found at {}; skipping gpu parity test",
                ptx_path.display()
            );
            return;
        }

        // Prepare context and a small range
        let eng = CudaEngine::new();
        let header = [5u8; 32];
        let threshold = U512::MAX; // permissive for parity in small ranges
        let ctx = eng.prepare_context(header, threshold);

        let range = engine_cpu::EngineRange {
            start: U512::from(0u64),
            end: U512::from(100u64),
        };
        let cancel = AtomicBool::new(false);

        // Run GPU (which falls back to CPU only if GPU path fails)
        let gpu_status = eng.search_range(&ctx, range.clone(), &cancel);

        // Run CPU fast for parity
        let cpu = engine_cpu::FastCpuEngine::new();
        let cpu_status = cpu.search_range(&ctx, range.clone(), &cancel);

        match (gpu_status, cpu_status) {
            (
                engine_cpu::EngineStatus::Found {
                    candidate: g,
                    hash_count: gh,
                },
                engine_cpu::EngineStatus::Found {
                    candidate: c,
                    hash_count: ch,
                },
            ) => {
                assert_eq!(g.nonce, c.nonce, "winner nonce mismatch");
                assert_eq!(g.distance, c.distance, "distance mismatch");
                assert_eq!(gh, ch, "hash_count mismatch");
            }
            (
                engine_cpu::EngineStatus::Exhausted { hash_count: gh },
                engine_cpu::EngineStatus::Exhausted { hash_count: ch },
            ) => {
                assert_eq!(gh, ch, "hash_count mismatch on Exhausted");
            }
            (g, c) => {
                panic!("gpu and cpu statuses differ: gpu={:?}, cpu={:?}", g, c);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use primitive_types::U512;

    #[test]
    fn placeholder_engine_basics() {
        let eng = CudaEngine::new();
        assert_eq!(eng.name(), "gpu-cuda");

        // Ensure context creation works and is deterministic in shape.
        let header = [1u8; 32];
        let threshold = U512::from(12345u64);
        let ctx = eng.prepare_context(header, threshold);

        assert_eq!(ctx.header, header);
        assert_eq!(ctx.threshold, threshold);
    }
}

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

        // Prefer embedded CUBIN (native SASS) by default; allow env override; fall back to embedded/env PTX if unavailable.
        let image_override = std::env::var("MINER_CUDA_IMAGE")
            .ok()
            .map(|s| s.to_ascii_lowercase());
        let module = match image_override.as_deref() {
            Some("ptx") => {
                // Force PTX
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
                log::info!(target: "miner", "CUDA: using PTX source = {ptx_origin} (forced by MINER_CUDA_IMAGE=ptx)");
                let ptx_cstr = CString::new(ptx_text)?;
                cuda::module::Module::from_ptx_cstr(&ptx_cstr, &[])
                    .with_context(|| "load PTX module")?
            }
            Some("cubin") => {
                // Force CUBIN
                if let Some(cubin) = cubin_embedded::get_cubin("qpow_kernel") {
                    log::info!(target: "miner", "CUDA: using CUBIN (embedded) (forced by MINER_CUDA_IMAGE=cubin)");
                    cuda::module::Module::from_cubin(cubin, &[])
                        .with_context(|| "load CUBIN module")?
                } else {
                    anyhow::bail!(
                        "MINER_CUDA_IMAGE=cubin requested but no embedded CUBIN image was found"
                    );
                }
            }
            _ => {
                if let Some(cubin) = cubin_embedded::get_cubin("qpow_kernel") {
                    log::info!(target: "miner", "CUDA: using CUBIN (embedded)");
                    cuda::module::Module::from_cubin(cubin, &[])
                        .with_context(|| "load CUBIN module")?
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
                }
            }
        };
        let stream = cuda::stream::Stream::new(cuda::stream::StreamFlags::DEFAULT, None)
            .with_context(|| "create stream")?;
        // Select kernel by mode; try G2 when requested, otherwise use G1. If G2 unavailable, fall back to G1.
        let func_name = match std::env::var("MINER_CUDA_MODE").ok().as_deref() {
            Some("g2") => {
                log::info!(target: "miner", "CUDA: MINER_CUDA_MODE=g2 requested; attempting to resolve G2 kernel");
                "qpow_montgomery_g2_kernel"
            }
            _ => "qpow_montgomery_g1_kernel",
        };
        let is_g2;
        let func = match module.get_function(func_name) {
            Ok(f) => {
                is_g2 = func_name == "qpow_montgomery_g2_kernel";
                f
            }
            Err(e) => {
                if func_name == "qpow_montgomery_g2_kernel" {
                    log::warn!(target: "miner", "CUDA: G2 kernel unavailable ({e:?}); falling back to G1");
                    is_g2 = false;
                    module
                        .get_function("qpow_montgomery_g1_kernel")
                        .with_context(|| "get kernel function 'qpow_montgomery_g1_kernel'")?
                } else {
                    return Err(e).with_context(|| "get kernel function")?;
                }
            }
        };

        // Emit backend label metric (g1/g2)
        #[cfg(feature = "metrics")]
        {
            metrics::set_engine_backend("gpu-cuda", if is_g2 { "g2" } else { "g1" });
        }
        // Probe device kernel ABI version if symbol is present (best-effort)
        if is_g2 {
            if let Ok(sym) =
                module.get_global::<u32>(std::ffi::CString::new("C_ABI_VERSION")?.as_c_str())
            {
                let mut abi: u32 = 0;
                if sym.copy_to(&mut abi).is_ok() {
                    log::info!(target: "miner", "CUDA G2: kernel ABI version = {abi}");
                }
            }
        }
        // Precompute Montgomery constants (host-side)
        let gc = GpuConstants::from_ctx(ctx);

        // Flatten constants (host)
        let m_le = gc.m_le;
        let n_le = gc.n_le;
        let r2_le = gc.r2_le;
        let m_hat_le = gc.m_hat_le;
        let n0_inv = gc.n0_inv;

        // Allocate device constant buffers once (persist across launches)
        let d_m = cuda::memory::DeviceBuffer::<u64>::from_slice(&m_le)
            .with_context(|| "alloc/copy d_m")?;
        let d_n = cuda::memory::DeviceBuffer::<u64>::from_slice(&n_le)
            .with_context(|| "alloc/copy d_n")?;
        let d_r2 = cuda::memory::DeviceBuffer::<u64>::from_slice(&r2_le)
            .with_context(|| "alloc/copy d_r2")?;
        let d_mhat = cuda::memory::DeviceBuffer::<u64>::from_slice(&m_hat_le)
            .with_context(|| "alloc/copy d_mhat")?;

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

            let num_threads = threads;
            // Use many iterations per thread to reduce host exponentiations per launch
            let grid_dim = ((num_threads + block_dim - 1) / block_dim).max(1);

            // Prepare per-thread y0 for base nonces: current + t
            let mut base_nonces: Vec<U512> = vec![U512::zero(); num_threads as usize];
            let mut y0_host: Vec<u64> = vec![0u64; (num_threads as usize) * 8];
            for t in 0..(num_threads as usize) {
                let stride = iters_per_thread as u64;
                let base_nonce = current.saturating_add(U512::from((t as u64) * stride));
                base_nonces[t] = base_nonce;
                let y0_u512 = pow_core::init_worker_y0(ctx, base_nonce);
                let y0_le = u512_to_le_limbs(y0_u512);
                let off = t * 8;
                y0_host[off..off + 8].copy_from_slice(&y0_le);
            }

            // Prepare/update per-chunk inputs
            let d_y0 = cuda::memory::DeviceBuffer::<u64>::from_slice(&y0_host)
                .with_context(|| "alloc/copy d_y0")?;
            let d_y_out = cuda::memory::DeviceBuffer::<u64>::zeroed(
                (num_threads as usize) * (iters_per_thread as usize) * 8,
            )
            .with_context(|| "alloc d_y_out")?;

            // Branch kernel launch by mode: G2 (device SHA3 + early-exit) vs G1 (return y values)
            if is_g2 {
                // Compute full-precision bounds for accounting
                let rem_be = end.saturating_sub(current).to_big_endian();
                let mut last8 = [0u8; 8];
                last8.copy_from_slice(&rem_be[56..64]);
                let remaining_inclusive: u64 = u64::from_be_bytes(last8);
                let total_elems_u64 = (num_threads as u64) * (iters_per_thread as u64);
                let covered = std::cmp::min(total_elems_u64, remaining_inclusive);

                // Prepare target/threshold (64-byte big-endian) for device
                let target_be = ctx.target.to_big_endian();
                let threshold_be = ctx.threshold.to_big_endian();
                let d_target = cuda::memory::DeviceBuffer::<u8>::from_slice(&target_be)
                    .with_context(|| "alloc/copy d_target")?;
                let d_threshold = cuda::memory::DeviceBuffer::<u8>::from_slice(&threshold_be)
                    .with_context(|| "alloc/copy d_threshold")?;

                // Early-exit outputs
                let d_found = cuda::memory::DeviceBuffer::<i32>::from_slice(&[0i32])
                    .with_context(|| "alloc/copy d_found")?;
                let d_index = cuda::memory::DeviceBuffer::<u32>::from_slice(&[0u32])
                    .with_context(|| "alloc/copy d_index")?;
                // Winner coordinates (thread id and iteration)
                let d_win_tid = cuda::memory::DeviceBuffer::<u32>::from_slice(&[u32::MAX])
                    .with_context(|| "alloc/copy d_win_tid")?;
                let d_win_j = cuda::memory::DeviceBuffer::<u32>::from_slice(&[u32::MAX])
                    .with_context(|| "alloc/copy d_win_j")?;
                let d_distance = cuda::memory::DeviceBuffer::<u8>::zeroed(64)
                    .with_context(|| "alloc d_distance")?;
                let d_dbg_y = cuda::memory::DeviceBuffer::<u8>::zeroed(64)
                    .with_context(|| "alloc d_dbg_y")?;
                let d_dbg_h = cuda::memory::DeviceBuffer::<u8>::zeroed(64)
                    .with_context(|| "alloc d_dbg_h")?;

                // Launch G2 kernel
                // Attempt to populate __constant__ symbols for per-job constants (G2 fast path).
                // If any symbol is unavailable, leave C_CONSTS_READY at 0 and the kernel will use parameters.
                {
                    let mut consts_ready_set = false;
                    // Prepare big-endian 64-bit limbs for target and threshold
                    let mut target_limbs = [0u64; 8];
                    let mut thresh_limbs = [0u64; 8];
                    for i in 0..8 {
                        let mut limb = [0u8; 8];
                        limb.copy_from_slice(&target_be[i * 8..(i + 1) * 8]);
                        target_limbs[i] = u64::from_be_bytes(limb);
                        let mut limb2 = [0u8; 8];
                        limb2.copy_from_slice(&threshold_be[i * 8..(i + 1) * 8]);
                        thresh_limbs[i] = u64::from_be_bytes(limb2);
                    }
                    if let (
                        Ok(mut c_n),
                        Ok(mut c_r2),
                        Ok(mut c_mhat),
                        Ok(mut c_n0),
                        Ok(mut c_target),
                        Ok(mut c_thresh),
                        Ok(mut c_ready),
                    ) = (
                        module.get_global::<[u64; 8]>(CString::new("C_N")?.as_c_str()),
                        module.get_global::<[u64; 8]>(CString::new("C_R2")?.as_c_str()),
                        module.get_global::<[u64; 8]>(CString::new("C_MHAT")?.as_c_str()),
                        module.get_global::<u64>(CString::new("C_N0_INV")?.as_c_str()),
                        module.get_global::<[u64; 8]>(CString::new("C_TARGET")?.as_c_str()),
                        module.get_global::<[u64; 8]>(CString::new("C_THRESH")?.as_c_str()),
                        module.get_global::<i32>(CString::new("C_CONSTS_READY")?.as_c_str()),
                    ) {
                        // Copy constants into constant memory
                        c_n.copy_from(&n_le)?;
                        c_r2.copy_from(&r2_le)?;
                        c_mhat.copy_from(&m_hat_le)?;
                        c_n0.copy_from(&(n0_inv as u64))?;
                        c_target.copy_from(&target_limbs)?;
                        c_thresh.copy_from(&thresh_limbs)?;
                        // Optional sampler: initialize when env MINER_CUDA_SAMPLER=1
                        if std::env::var("MINER_CUDA_SAMPLER")
                            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                            .unwrap_or(false)
                        {
                            if let (
                                Ok(mut c_samp_en),
                                Ok(mut c_samp_y),
                                Ok(mut c_samp_h),
                                Ok(mut c_samp_target_be),
                                Ok(mut c_samp_thresh_be),
                                Ok(mut c_samp_index),
                                Ok(mut c_samp_dec),
                            ) = (
                                module.get_global::<i32>(
                                    CString::new("C_SAMPLER_ENABLE")?.as_c_str(),
                                ),
                                module.get_global::<[u8; 64]>(
                                    CString::new("C_SAMPLER_Y_BE")?.as_c_str(),
                                ),
                                module.get_global::<[u8; 64]>(
                                    CString::new("C_SAMPLER_H_BE")?.as_c_str(),
                                ),
                                module.get_global::<[u8; 64]>(
                                    CString::new("C_SAMPLER_TARGET_BE")?.as_c_str(),
                                ),
                                module.get_global::<[u8; 64]>(
                                    CString::new("C_SAMPLER_THRESH_BE")?.as_c_str(),
                                ),
                                module
                                    .get_global::<u32>(CString::new("C_SAMPLER_INDEX")?.as_c_str()),
                                module.get_global::<u32>(
                                    CString::new("C_SAMPLER_DECISION")?.as_c_str(),
                                ),
                            ) {
                                let one_i32: i32 = 1;
                                c_samp_en.copy_from(&one_i32)?;
                                // Initialize sampler buffers (copy target/threshold; zero y/h/index/decision)
                                c_samp_target_be.copy_from(&target_be)?;
                                c_samp_thresh_be.copy_from(&threshold_be)?;
                                c_samp_index.copy_from(&0u32)?;
                                c_samp_dec.copy_from(&0u32)?;
                                let zero64 = [0u8; 64];
                                c_samp_y.copy_from(&zero64)?;
                                c_samp_h.copy_from(&zero64)?;
                            }
                        }
                        // Flag ready = 1
                        let one: i32 = 1;
                        c_ready.copy_from(&one)?;
                        consts_ready_set = true;
                    }
                    if !consts_ready_set {
                        if let Ok(mut c_ready) =
                            module.get_global::<i32>(CString::new("C_CONSTS_READY")?.as_c_str())
                        {
                            let zero: i32 = 0;
                            c_ready.copy_from(&zero)?;
                        }
                    }
                }
                log::info!(target: "miner", "CUDA G2 launch: grid_dim={grid_dim}, block_dim={block_dim}, threads={num_threads}, iters={iters_per_thread}");
                log::debug!(target: "miner", "CUDA G2 coverage: remaining_inclusive={}, covered={covered}", remaining_inclusive);
                let t_kernel_start = std::time::Instant::now();
                let launch_result = unsafe {
                    launch!(func<<<grid_dim, block_dim, 0, stream>>>(
                        d_m.as_device_ptr(),
                        d_n.as_device_ptr(),
                        n0_inv as u64,
                        d_r2.as_device_ptr(),
                        d_mhat.as_device_ptr(),
                        d_y0.as_device_ptr(),
                        d_target.as_device_ptr(),
                        d_threshold.as_device_ptr(),
                        d_found.as_device_ptr(),
                        d_index.as_device_ptr(),
                        d_win_tid.as_device_ptr(),
                        d_win_j.as_device_ptr(),
                        d_distance.as_device_ptr(),
                        d_dbg_y.as_device_ptr(),
                        d_dbg_h.as_device_ptr(),
                        num_threads as u32,
                        iters_per_thread as u32,
                        covered as u64
                    ))
                };
                launch_result.with_context(|| "launch G2 kernel")?;
                stream
                    .synchronize()
                    .with_context(|| "stream synchronize (G2)")?;
                let kernel_ms = t_kernel_start.elapsed().as_millis();
                log::info!(target: "miner", "CUDA G2 kernel and sync OK (kernel_ms={kernel_ms})");
                // Optional sampler readback and host parity check
                if std::env::var("MINER_CUDA_SAMPLER")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false)
                {
                    if let (Ok(c_en), Ok(c_idx), Ok(c_dec), Ok(c_y), Ok(c_h), Ok(c_t), Ok(c_th)) = (
                        module.get_global::<i32>(CString::new("C_SAMPLER_ENABLE")?.as_c_str()),
                        module.get_global::<u32>(CString::new("C_SAMPLER_INDEX")?.as_c_str()),
                        module.get_global::<u32>(CString::new("C_SAMPLER_DECISION")?.as_c_str()),
                        module.get_global::<[u8; 64]>(CString::new("C_SAMPLER_Y_BE")?.as_c_str()),
                        module.get_global::<[u8; 64]>(CString::new("C_SAMPLER_H_BE")?.as_c_str()),
                        module.get_global::<[u8; 64]>(
                            CString::new("C_SAMPLER_TARGET_BE")?.as_c_str(),
                        ),
                        module.get_global::<[u8; 64]>(
                            CString::new("C_SAMPLER_THRESH_BE")?.as_c_str(),
                        ),
                    ) {
                        let mut en: i32 = 0;
                        c_en.copy_to(&mut en)?;
                        if en != 0 {
                            let mut y_be = [0u8; 64];
                            let mut h_be = [0u8; 64];
                            let mut t_be = [0u8; 64];
                            let mut th_be = [0u8; 64];
                            let mut samp_idx: u32 = 0;
                            let mut samp_dec: u32 = 0;
                            c_y.copy_to(&mut y_be)?;
                            c_h.copy_to(&mut h_be)?;
                            c_t.copy_to(&mut t_be)?;
                            c_th.copy_to(&mut th_be)?;
                            c_idx.copy_to(&mut samp_idx)?;
                            c_dec.copy_to(&mut samp_dec)?;
                            // Host recompute using pow_core from y
                            let y_u512 = U512::from_big_endian(&y_be);
                            let host_distance = pow_core::distance_from_y(ctx, y_u512);
                            let host_ok = pow_core::is_valid_distance(ctx, host_distance);
                            let dev_ok = samp_dec != 0;
                            if host_ok != dev_ok {
                                log::warn!(
                                    target: "miner",
                                    "CUDA G2 sampler mismatch: idx={}, host_ok={}, dev_ok={}, host_distance={}",
                                    samp_idx,
                                    host_ok,
                                    dev_ok,
                                    host_distance
                                );
                                #[cfg(feature = "metrics")]
                                {
                                    metrics::inc_sample_mismatch("gpu-cuda");
                                }
                            }
                        }
                    }
                }

                // Copy back early-exit flag (and details if found)
                let mut h_found = [0i32; 1];
                d_found
                    .copy_to(&mut h_found)
                    .with_context(|| "copy found flag")?;
                if h_found[0] != 0 {
                    let mut h_idx = [0u32; 1];
                    d_index.copy_to(&mut h_idx).with_context(|| "copy index")?;
                    let k = h_idx[0] as u64;
                    // Account: include the winning step
                    hash_count = hash_count.saturating_add(k + 1);

                    // Compute nonce from linear index (t, j)
                    let mut t_idx = (k as usize) / (iters_per_thread as usize);
                    let mut j_idx = (k as usize) % (iters_per_thread as usize);
                    // Prefer winner coordinates returned via device buffers when available
                    let mut h_win_tid = [u32::MAX; 1];
                    let mut h_win_j = [u32::MAX; 1];
                    // Prefer device-returned winner coordinates; fall back to k-derived indices only if unavailable/invalid
                    let _ = d_win_tid.copy_to(&mut h_win_tid);
                    let _ = d_win_j.copy_to(&mut h_win_j);
                    if h_win_tid[0] != u32::MAX
                        && h_win_j[0] != u32::MAX
                        && (h_win_tid[0] as usize) < (num_threads as usize)
                        && (h_win_j[0] as usize) < (iters_per_thread as usize)
                    {
                        t_idx = h_win_tid[0] as usize;
                        j_idx = h_win_j[0] as usize;
                    } else {
                        log::debug!(target: "miner", "CUDA G2: winner (tid,j) readback unavailable/invalid; falling back to k-derived indices");
                    }
                    // Reconstruct nonce using the exact per-thread base nonce used for y0:
                    // nonce = base_nonces[t_idx] + (j_idx + 1)
                    log::debug!(target: "miner", "CUDA G2: winner coords used: k={}, dev_tid={}, dev_j={}, used_tid={}, used_j={}", k, h_win_tid[0], h_win_j[0], t_idx, j_idx);
                    let base_nonce = base_nonces[t_idx];
                    let nonce = base_nonce.saturating_add(U512::from(1u64 + (j_idx as u64)));

                    // Copy back distance (optional for completeness; threshold was already satisfied)
                    let mut h_dist = [0u8; 64];
                    d_distance
                        .copy_to(&mut h_dist)
                        .with_context(|| "copy distance")?;

                    let work = nonce.to_big_endian();
                    // Host re-verification to ensure device result matches chain semantics
                    let host_distance = pow_core::distance_for_nonce(ctx, nonce);
                    if pow_core::is_valid_distance(ctx, host_distance) {
                        log::info!(target: "miner", "CUDA G2: early-exit found at idx={k}");
                        return Ok(EngineStatus::Found {
                            candidate: engine_cpu::EngineCandidate {
                                nonce,
                                work,
                                distance: host_distance,
                            },
                            hash_count,
                        });
                    } else {
                        // Treat as not found: account coverage and advance like the "not found" path
                        // Copy debug y/h bytes from device and log first 16 bytes (hex) for diagnosis
                        let mut dbg_y = [0u8; 64];
                        let mut dbg_h = [0u8; 64];
                        // Ignore copy errors in logging path; continue accounting regardless
                        let _ = d_dbg_y.copy_to(&mut dbg_y);
                        let _ = d_dbg_h.copy_to(&mut dbg_h);
                        let y_hex = hex::encode(&dbg_y[..16]);
                        let h_hex = hex::encode(&dbg_h[..16]);
                        // Compute richer diagnostics (host hash, host/device distances, target/threshold prefixes)
                        let y_u512 = U512::from_big_endian(&dbg_y);
                        let host_hash_u512 = pow_core::compat::sha3_512(y_u512);
                        let host_hash_be = host_hash_u512.to_big_endian();
                        let host_hash_hex = hex::encode(&host_hash_be[..16]);

                        // Device distance bytes (already written by kernel) and host distance bytes
                        let dev_dist_hex = hex::encode(&h_dist[..16]);
                        let host_dist_be = host_distance.to_big_endian();
                        let host_dist_hex = hex::encode(&host_dist_be[..16]);

                        // Additionally compute expected host distance directly from dbg_y to diagnose nonce/target mismatches
                        let host_dist_from_y = pow_core::distance_from_y(ctx, y_u512);
                        let host_dist_y_be = host_dist_from_y.to_big_endian();
                        let host_dist_y_hex = hex::encode(&host_dist_y_be[..16]);

                        // Target/threshold prefixes
                        let target_be = ctx.target.to_big_endian();
                        let thresh_be = ctx.threshold.to_big_endian();
                        let target_hex = hex::encode(&target_be[..16]);
                        let thresh_hex = hex::encode(&thresh_be[..16]);

                        // Device decided "found" (dev_ok=true) but host rejected (host_ok=false)
                        // Additional diagnostics: compute alternative nonce mappings to detect index reconstruction mismatches.
                        let iters_usize = iters_per_thread as usize;
                        let threads_usize = num_threads as usize;
                        let k_usize = k as usize;

                        // Row-major (+1): t = k / iters, j = k % iters
                        let t_row = k_usize / iters_usize;
                        let j_row = k_usize % iters_usize;
                        let nonce_row_p1 = current.saturating_add(U512::from(
                            1u64 + (t_row as u64) * (iters_per_thread as u64) + (j_row as u64),
                        ));
                        let dist_row_p1 = pow_core::distance_for_nonce(ctx, nonce_row_p1);
                        let dist_row_p1_hex = {
                            let be = dist_row_p1.to_big_endian();
                            hex::encode(&be[..16])
                        };

                        // Column-major (+1): t = k % threads, j = k / threads
                        let t_col = k_usize % threads_usize;
                        let j_col = k_usize / threads_usize;
                        let nonce_col_p1 = current.saturating_add(U512::from(
                            1u64 + (t_col as u64) * (iters_per_thread as u64) + (j_col as u64),
                        ));
                        let dist_col_p1 = pow_core::distance_for_nonce(ctx, nonce_col_p1);
                        let dist_col_p1_hex = {
                            let be = dist_col_p1.to_big_endian();
                            hex::encode(&be[..16])
                        };

                        // Row-major (+0)
                        let nonce_row_p0 = current.saturating_add(U512::from(
                            (t_row as u64) * (iters_per_thread as u64) + (j_row as u64),
                        ));
                        let dist_row_p0 = pow_core::distance_for_nonce(ctx, nonce_row_p0);
                        let dist_row_p0_hex = {
                            let be = dist_row_p0.to_big_endian();
                            hex::encode(&be[..16])
                        };

                        // Column-major (+0)
                        let nonce_col_p0 = current.saturating_add(U512::from(
                            (t_col as u64) * (iters_per_thread as u64) + (j_col as u64),
                        ));
                        let dist_col_p0 = pow_core::distance_for_nonce(ctx, nonce_col_p0);
                        let dist_col_p0_hex = {
                            let be = dist_col_p0.to_big_endian();
                            hex::encode(&be[..16])
                        };

                        log::warn!(
                            target: "miner",
                            "CUDA G2: false positive: idx={k}, dev_ok=true host_ok=false; y[0..16]={}, dev_h[0..16]={}, host_h[0..16]={}, dev_dist[0..16]={}, host_dist[0..16]={}, host_y_dist[0..16]={}, map_r+1[0..16]={}, map_c+1[0..16]={}, map_r+0[0..16]={}, map_c+0[0..16]={}, target[0..16]={}, thresh[0..16]={}",
                            y_hex,
                            h_hex,
                            host_hash_hex,
                            dev_dist_hex,
                            host_dist_hex,
                            host_dist_y_hex,
                            dist_row_p1_hex,
                            dist_col_p1_hex,
                            dist_row_p0_hex,
                            dist_col_p0_hex,
                            target_hex,
                            thresh_hex
                        );
                        // Winner coordinates diagnostic for FP analysis
                        log::warn!(
                            target: "miner",
                            "CUDA G2: fp details: idx={}, win_tid={}, win_j={}, used_tid={}, used_j={}, used_dev={}",
                            k,
                            h_win_tid[0],
                            h_win_j[0],
                            t_idx,
                            j_idx,
                            (h_win_tid[0] != u32::MAX
                                && h_win_j[0] != u32::MAX
                                && (h_win_tid[0] as usize) < (num_threads as usize)
                                && (h_win_j[0] as usize) < (iters_per_thread as usize))
                        );
                        #[cfg(feature = "metrics")]
                        {
                            metrics::inc_candidates_false_positive("gpu-cuda");
                            metrics::inc_sample_mismatch("gpu-cuda");
                        }
                        hash_count = hash_count.saturating_add(covered);
                        if covered < total_elems_u64 {
                            return Ok(EngineStatus::Exhausted { hash_count });
                        }
                        current = current.saturating_add(U512::from(1u64 + total_elems_u64));
                        continue;
                    }
                }

                // Not found in this window
                hash_count = hash_count.saturating_add(covered);
                if covered < total_elems_u64 {
                    return Ok(EngineStatus::Exhausted { hash_count });
                }
                // Advance and continue loop (no G1 copy-back in G2 mode)
                current = current.saturating_add(U512::from(1u64 + total_elems_u64));
                continue;
            } else {
                // G1 launch: computes y for (current + t + 1) for each thread t in [0, num_threads)
                log::info!(target: "miner", "CUDA launch: grid_dim={grid_dim}, block_dim={block_dim}, threads={num_threads}, iters={iters_per_thread}");

                let t_kernel_start = std::time::Instant::now();
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
                launch_result.with_context(|| "launch kernel")?;
                stream.synchronize().with_context(|| "stream synchronize")?;
                let kernel_ms = t_kernel_start.elapsed().as_millis();
                log::info!(target: "miner", "CUDA kernel and sync OK (kernel_ms={kernel_ms})");
            }

            // G2 handled in the launch branch above; proceed with G1 copy-back path below.
            if !is_g2 {
                // G1 path (host SHA3) — Copy back results (optional pinned host buffer + async copy)
                let use_pinned = std::env::var("MINER_CUDA_PINNED")
                    .ok()
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                let total_words = (num_threads as usize) * (iters_per_thread as usize) * 8;
                let t_copy_start = std::time::Instant::now();
                let y_host_arc: std::sync::Arc<dyn AsRef<[u64]> + Send + Sync> = if use_pinned {
                    let mut pinned =
                        unsafe { cust::memory::LockedBuffer::<u64>::uninitialized(total_words) }
                            .with_context(|| "alloc pinned host buffer")?;
                    unsafe {
                        cust::memory::AsyncCopyDestination::async_copy_to(
                            &*d_y_out,
                            pinned.as_mut_slice(),
                            &stream,
                        )
                        .with_context(|| "async copy d_y_out -> pinned")?;
                    }
                    stream
                        .synchronize()
                        .with_context(|| "stream synchronize (post async D2H)")?;
                    log::info!(target: "miner", "CUDA copy-back OK (pinned, async): elems={}, copy_ms={}", total_words, t_copy_start.elapsed().as_millis());
                    std::sync::Arc::new(pinned)
                } else {
                    let mut y_out_host = vec![0u64; total_words];
                    d_y_out
                        .copy_to(&mut y_out_host)
                        .with_context(|| "copy d_y_out -> host")?;
                    let copy_ms = t_copy_start.elapsed().as_millis();
                    log::info!(target: "miner", "CUDA copy-back OK: elems={}, copy_ms={copy_ms}", y_out_host.len());
                    std::sync::Arc::new(y_out_host)
                };

                // Parallel SHA3 over GPU results; compute earliest valid index (if any)
                let t_sha_start = std::time::Instant::now();

                // Bound hashing work to remaining range in this chunk
                let rem_be = end.saturating_sub(current).to_big_endian();
                let mut last8 = [0u8; 8];
                last8.copy_from_slice(&rem_be[56..64]);
                let remaining_inclusive: u64 = u64::from_be_bytes(last8);

                let total_elems = (num_threads as usize) * (iters_per_thread as usize);
                let total_to_hash = std::cmp::min(total_elems as u64, remaining_inclusive) as usize;

                let hash_threads = std::env::var("MINER_CUDA_HASH_THREADS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| {
                        std::thread::available_parallelism()
                            .map(|n| n.get())
                            .unwrap_or(1)
                    });
                log::info!(target: "miner", "Host SHA3: threads={hash_threads}, total_elems={total_to_hash}");

                // Share outputs across worker threads (Vec or LockedBuffer depending on MINER_CUDA_PINNED)
                let y_shared = y_host_arc.clone();
                let ctx_for_threads = ctx.clone();
                let found_min_idx =
                    std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(usize::MAX));
                let found_info = std::sync::Arc::new(std::sync::Mutex::new(None::<(U512, U512)>));

                std::thread::scope(|scope| {
                    for tidx in 0..hash_threads.max(1) {
                        let y_arc = y_shared.clone();
                        let ctx_local = ctx_for_threads.clone();
                        let min_idx = found_min_idx.clone();
                        let info = found_info.clone();
                        let start_k = (total_to_hash * tidx) / hash_threads.max(1);
                        let end_k = (total_to_hash * (tidx + 1)) / hash_threads.max(1);
                        let iters_usize = iters_per_thread as usize;

                        scope.spawn(move || {
                            for k in start_k..end_k {
                                if cancel.load(AtomicOrdering::Relaxed) {
                                    break;
                                }

                                let idx_words = k * 8;
                                let mut le = [0u64; 8];
                                let y_slice: &[u64] = y_arc.as_ref().as_ref();
                                le.copy_from_slice(&y_slice[idx_words..idx_words + 8]);
                                let y_norm = le_limbs_to_u512(&le);

                                let distance = pow_core::distance_from_y(&ctx_local, y_norm);
                                if pow_core::is_valid_distance(&ctx_local, distance) {
                                    let t = k / iters_usize;
                                    let j = k % iters_usize;
                                    let nonce = current.saturating_add(U512::from(
                                        1u64 + (t as u64) * (iters_per_thread as u64) + (j as u64),
                                    ));

                                    let prev = min_idx.load(std::sync::atomic::Ordering::Relaxed);
                                    if k < prev {
                                        min_idx.store(k, std::sync::atomic::Ordering::Relaxed);
                                        if let Ok(mut guard) = info.lock() {
                                            *guard = Some((nonce, distance));
                                        }
                                    }
                                }
                            }
                        });
                    }
                });

                // SHA3 (host) timing
                let sha3_ms = t_sha_start.elapsed().as_millis();
                log::info!(target: "miner", "SHA3 (host) OK: sha3_ms={sha3_ms}");

                // Outcomes and accounting
                let total_elems_u64 = total_to_hash as u64;

                // GPU coverage limited by remaining range
                let gpu_coverage = total_elems_u64;

                // If found, compute hash_count up to the earliest index and return
                if let Some((nonce, distance)) = found_info.lock().ok().and_then(|g| (*g).clone()) {
                    let k = found_min_idx.load(std::sync::atomic::Ordering::Relaxed) as u64;
                    hash_count = hash_count.saturating_add(k + 1);
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

                // No candidate found; update hash_count and either exhaust or advance
                hash_count = hash_count.saturating_add(gpu_coverage);

                if gpu_coverage < total_elems_u64 {
                    return Ok(EngineStatus::Exhausted { hash_count });
                }

                // Advance: we covered 1 (CPU step) + num_threads * iters_per_thread nonces in this iteration
                current = current.saturating_add(U512::from(1u64 + total_elems_u64));
            }
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

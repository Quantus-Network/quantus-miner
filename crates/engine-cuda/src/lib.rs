#![deny(rust_2018_idioms)]

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, sys as nvrtc_sys, CompileOptions, Ptx};
use engine_cpu::{CancelCheck, Candidate, EngineStatus, FoundOrigin, MinerEngine, Range};
use pow_core::{format_hashrate, format_u512, JobContext};
use primitive_types::U512;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const KERNEL_SRC: &str = include_str!("kernels/mining.cu");
const THREADS_PER_BLOCK: u32 = 256;
const MAX_BLOCKS: u32 = 4096;
const RESULTS_U32S: usize = 2;

#[repr(C)]
#[derive(Clone, Copy)]
struct MiningParams {
    prestate: [u32; 24],
    start_nonce: [u32; 16],
    difficulty_target: [u32; 16],
    dispatch_config: [u32; 3],
}

unsafe impl cudarc::driver::DeviceRepr for MiningParams {}

struct CudaDevice {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
    name: String,
}

struct WorkerBuffers {
    engine_id: usize,
    device_index: usize,
    stream: Arc<CudaStream>,
    results: CudaSlice<u32>,
    midstate: CudaSlice<u32>,
    start_nonce: CudaSlice<u32>,
    hashes: CudaSlice<u32>,
    mine: CudaFunction,
    hash: CudaFunction,
    busy: Duration,
    busy_since: Instant,
}

/// Native CUDA mining engine.
///
/// Search contract: the kernel evaluates every nonce in the range, but its
/// Goldilocks reduction skips two carry corrections (`reduce128` in
/// `kernels/mining.cu`). Each of the roughly 1,470 field multiplies per hash
/// can therefore be off by EPS mod p with probability about 2^-33, so about one
/// nonce in three million is hashed wrong on the GPU. A wrong hash that lands
/// below the target is rejected by CPU verification and the search resumes
/// after it. A wrong hash for a nonce that is actually valid is missed and the
/// range reports `Exhausted`. `hash_count` counts every evaluated nonce, wrong
/// ones included. The expected loss is about 3e-7 of solutions, far below the
/// throughput the shortcut buys. `hash_nonces` is subject to the same contract:
/// it is a kernel self-test, not a verifier; use `pow_core::hash_from_nonce`.
pub struct CudaEngine {
    engine_id: usize,
    devices: Vec<Arc<CudaDevice>>,
    device_counter: AtomicUsize,
    batch_size: u32,
    throttle_ms: u64,
}

static ENGINE_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static ASSIGNED_DEVICE: RefCell<Option<(usize, usize)>> = const { RefCell::new(None) };
    static WORKER_BUFFERS: RefCell<Option<WorkerBuffers>> = const { RefCell::new(None) };
    static DEVICE_LOST: RefCell<Option<usize>> = const { RefCell::new(None) };
}

impl CudaEngine {
    pub fn try_new(batch_size: u32, throttle_ms: u64) -> Result<Self, Box<dyn std::error::Error>> {
        if batch_size == 0 {
            return Err("batch_size must be non-zero".into());
        }

        match silent_catch(cudarc::driver::result::init) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                return Err(format!(
                    "CUDA driver is not available (need libcuda / nvidia driver): {e}"
                )
                .into());
            }
            Err(_) => {
                return Err("CUDA driver is not available (need libcuda / nvidia driver)".into());
            }
        }
        let count = cudarc::driver::result::device::get_count()
            .map_err(|e| format!("Failed to query CUDA device count: {e}"))?;
        if count <= 0 {
            return Err("No CUDA devices found".into());
        }

        let mut devices = Vec::new();
        let mut compiled: HashMap<(i32, i32), Ptx> = HashMap::new();
        for ordinal in 0..count as usize {
            let ctx = CudaContext::new(ordinal)
                .map_err(|e| format!("Failed to create CUDA context for device {ordinal}: {e}"))?;
            ctx.set_blocking_synchronize().map_err(|e| {
                format!("Failed to enable blocking CUDA synchronization on device {ordinal}: {e}")
            })?;
            let name = ctx
                .name()
                .unwrap_or_else(|_| format!("cuda-device-{ordinal}"));
            let (major, minor) = ctx.compute_capability().map_err(|e| {
                format!("Failed to query compute capability of CUDA device {ordinal}: {e}")
            })?;
            let ptx = match compiled.get(&(major, minor)) {
                Some(ptx) => ptx.clone(),
                None => {
                    let ptx = match silent_catch(|| compile_kernel(major, minor)) {
                        Ok(ptx) => ptx?,
                        Err(_) => {
                            return Err("CUDA NVRTC library is not available (need libnvrtc from the CUDA toolkit)".into());
                        }
                    };
                    compiled.insert((major, minor), ptx.clone());
                    ptx
                }
            };
            let module = ctx
                .load_module(ptx)
                .map_err(|e| format!("Failed to load CUDA mining module on {name}: {e}"))?;
            let mine = module
                .load_function("mining_main")
                .map_err(|e| format!("Failed to load mining_main on {name}: {e}"))?;
            let (regs, local, shared, constant) = (
                mine.num_regs()?,
                mine.local_size_bytes()?,
                mine.shared_size_bytes()?,
                mine.const_size_bytes()?,
            );
            log::info!(
                target: "cuda_engine",
                "CUDA device {ordinal}: {name} (sm_{major}{minor}; mining_main uses {regs} registers, {local} B local, {shared} B shared, {constant} B constant)"
            );
            if local > 0 {
                log::warn!(
                    target: "cuda_engine",
                    "mining_main spills {local} bytes per thread to local memory on {name}"
                );
            }
            devices.push(Arc::new(CudaDevice { ctx, module, name }));
        }

        log::info!(
            target: "cuda_engine",
            "CUDA engine initialized with {} device(s) (batch size: {batch_size} nonces, throttle: {throttle_ms}ms)",
            devices.len()
        );

        Ok(Self {
            engine_id: ENGINE_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            devices,
            device_counter: AtomicUsize::new(0),
            batch_size,
            throttle_ms,
        })
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    pub fn clear_worker_resources() {
        WORKER_BUFFERS.with(|b| {
            *b.borrow_mut() = None;
        });
        ASSIGNED_DEVICE.with(|a| {
            *a.borrow_mut() = None;
        });
        DEVICE_LOST.with(|lost| {
            *lost.borrow_mut() = None;
        });
    }

    pub fn hash_nonces(
        &self,
        header: [u8; 32],
        start: U512,
        count: u32,
    ) -> Result<Vec<U512>, Box<dyn std::error::Error>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let device = &self.devices[0];
        let mut buffers = create_buffers(self.engine_id, 0, device, count)?;
        let nonce_be = start.to_big_endian();
        let mid = pow_core::mining_midstate_u32s(header, nonce_be[..32].try_into().unwrap());
        let start_limbs = pow_core::u512_to_le_u32s(start);
        buffers.stream.memcpy_htod(&mid, &mut buffers.midstate)?;
        buffers
            .stream
            .memcpy_htod(&start_limbs, &mut buffers.start_nonce)?;

        let num_blocks = count.div_ceil(THREADS_PER_BLOCK).max(1);
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = buffers.stream.launch_builder(&buffers.hash);
        builder.arg(&mut buffers.hashes);
        builder.arg(&buffers.midstate);
        builder.arg(&buffers.start_nonce);
        builder.arg(&count);
        unsafe {
            builder.launch(cfg)?;
        }
        buffers.stream.synchronize()?;
        let raw = buffers.stream.clone_dtoh(&buffers.hashes)?;
        let mut out = Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let mut limbs = [0u32; 16];
            limbs.copy_from_slice(&raw[i * 16..(i + 1) * 16]);
            out.push(pow_core::u512_from_le_u32s(limbs));
        }
        Ok(out)
    }
}

fn nvrtc_supported_archs() -> Result<Vec<i32>, String> {
    let mut count = 0i32;
    unsafe { nvrtc_sys::nvrtcGetNumSupportedArchs(&mut count) }
        .result()
        .map_err(|e| format!("nvrtcGetNumSupportedArchs failed: {e:?}"))?;
    let mut archs = vec![0i32; count.max(0) as usize];
    unsafe { nvrtc_sys::nvrtcGetSupportedArchs(archs.as_mut_ptr()) }
        .result()
        .map_err(|e| format!("nvrtcGetSupportedArchs failed: {e:?}"))?;
    Ok(archs)
}

/// Compiles PTX for the newest virtual architecture NVRTC knows that the device
/// can run. The driver JIT then emits native code; on a 4090 that measured
/// equal to loading NVRTC's own cubin, so the PTX route is kept.
fn compile_kernel(major: i32, minor: i32) -> Result<Ptx, Box<dyn std::error::Error>> {
    let device_arch = major * 10 + minor;
    let target = nvrtc_supported_archs()?
        .into_iter()
        .filter(|&arch| arch <= device_arch)
        .max()
        .ok_or_else(|| format!("NVRTC supports no architecture at or below sm_{device_arch}"))?;
    if target != device_arch {
        log::warn!(
            target: "cuda_engine",
            "NVRTC does not know sm_{device_arch}; compiling PTX for compute_{target} instead"
        );
    }
    let opts = CompileOptions {
        options: vec![format!("--gpu-architecture=compute_{target}")],
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(KERNEL_SRC, opts).map_err(|e| {
        format!("NVRTC failed to compile the CUDA mining kernel for compute_{target}: {e}")
    })?;
    log::info!(
        target: "cuda_engine",
        "Compiled CUDA mining kernel PTX for compute_{target} (device sm_{device_arch}); the driver JIT emits native code"
    );
    Ok(ptx)
}

fn silent_catch<T>(f: impl FnOnce() -> T) -> std::thread::Result<T> {
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap();
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    std::panic::set_hook(hook);
    out
}

fn create_buffers(
    engine_id: usize,
    device_index: usize,
    device: &CudaDevice,
    hash_capacity: u32,
) -> Result<WorkerBuffers, Box<dyn std::error::Error>> {
    let stream = device.ctx.default_stream();
    let mine = device.module.load_function("mining_main")?;
    let hash = device.module.load_function("hash_nonces")?;
    Ok(WorkerBuffers {
        engine_id,
        device_index,
        results: stream.alloc_zeros::<u32>(RESULTS_U32S)?,
        midstate: stream.alloc_zeros::<u32>(24)?,
        start_nonce: stream.alloc_zeros::<u32>(16)?,
        hashes: stream.alloc_zeros::<u32>(hash_capacity.max(1) as usize * 16)?,
        mine,
        hash,
        stream,
        busy: Duration::ZERO,
        busy_since: Instant::now(),
    })
}

fn take_gpu_duty_cycle(buffers: &mut WorkerBuffers) -> Option<f64> {
    let wall = buffers.busy_since.elapsed();
    let duty = (wall >= Duration::from_secs(1))
        .then(|| 100.0 * buffers.busy.as_secs_f64() / wall.as_secs_f64());
    buffers.busy = Duration::ZERO;
    buffers.busy_since = Instant::now();
    duty
}

fn worker_buffers(
    engine: &CudaEngine,
    device_index: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    WORKER_BUFFERS.with(|cell| {
        let mut slot = cell.borrow_mut();
        let reuse = matches!(
            &*slot,
            Some(b) if b.engine_id == engine.engine_id && b.device_index == device_index
        );
        if !reuse {
            *slot = Some(create_buffers(
                engine.engine_id,
                device_index,
                &engine.devices[device_index],
                1,
            )?);
        }
        Ok(())
    })
}

enum BatchResult {
    Found {
        candidate: Candidate,
        hash_count: u64,
    },
    NotFound {
        hash_count: u64,
    },
    Rejected {
        logical_index: u64,
    },
    DeviceLost,
}

fn nonces_until_low64_carry(nonce: U512) -> U512 {
    (U512::one() << 64).saturating_sub(U512::from(nonce.low_u64()))
}

fn run_single_batch(
    buffers: &mut WorkerBuffers,
    ctx: &JobContext,
    batch_start: U512,
    batch_size: u32,
) -> BatchResult {
    let num_blocks = batch_size.div_ceil(THREADS_PER_BLOCK).clamp(1, MAX_BLOCKS);
    let total_threads = num_blocks * THREADS_PER_BLOCK;
    let nonces_per_thread = batch_size.div_ceil(total_threads).max(1);
    let dispatch = [total_threads, nonces_per_thread, batch_size];
    let nonce_be = batch_start.to_big_endian();
    let start_limbs = pow_core::u512_to_le_u32s(batch_start);
    let prestate = pow_core::mining_prestate_low64_u32s(ctx.header, nonce_be);
    let target = pow_core::u512_to_le_u32s(ctx.target);
    let params = MiningParams {
        prestate,
        start_nonce: start_limbs,
        difficulty_target: target,
        dispatch_config: dispatch,
    };

    let launch_start = Instant::now();
    if let Err(e) = (|| {
        buffers.stream.memset_zeros(&mut buffers.results)?;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = buffers.stream.launch_builder(&buffers.mine);
        builder.arg(&mut buffers.results);
        builder.arg(&params);
        unsafe {
            builder.launch(cfg)?;
        }
        buffers.stream.synchronize()?;
        Ok::<(), Box<dyn std::error::Error>>(())
    })() {
        log::error!(target: "cuda_engine", "CUDA batch failed: {e}");
        return BatchResult::DeviceLost;
    }
    buffers.busy += launch_start.elapsed();

    let result_u32s = match buffers.stream.clone_dtoh(&buffers.results) {
        Ok(v) => v,
        Err(e) => {
            log::error!(target: "cuda_engine", "CUDA result copy failed: {e}");
            return BatchResult::DeviceLost;
        }
    };

    let dispatched = (total_threads as u64 * nonces_per_thread as u64).min(batch_size as u64);
    if result_u32s[0] != 0 {
        let logical_index = result_u32s[1] as u64;
        if logical_index >= dispatched {
            log::error!(
                target: "cuda_engine",
                "CUDA returned out-of-range candidate index {logical_index} for {dispatched} dispatched nonces"
            );
            return BatchResult::DeviceLost;
        }
        let nonce = batch_start + U512::from(logical_index);
        let hash = pow_core::hash_from_nonce(ctx, nonce);
        if hash >= ctx.target {
            log::warn!(
                target: "cuda_engine",
                "CUDA candidate {} rejected by CPU verification (hash {} >= target {}); resuming after it",
                format_u512(nonce),
                format_u512(hash),
                format_u512(ctx.target)
            );
            return BatchResult::Rejected { logical_index };
        }
        return BatchResult::Found {
            candidate: Candidate {
                nonce,
                work: nonce.to_big_endian(),
                hash,
            },
            hash_count: dispatched,
        };
    }

    BatchResult::NotFound {
        hash_count: dispatched,
    }
}

impl MinerEngine for CudaEngine {
    fn name(&self) -> &'static str {
        "gpu-cuda"
    }

    fn prepare_context(&self, header_hash: [u8; 32], difficulty: U512) -> JobContext {
        JobContext::new(header_hash, difficulty)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn search_range(
        &self,
        ctx: &JobContext,
        range: Range,
        cancel: &dyn CancelCheck,
    ) -> EngineStatus {
        if self.devices.is_empty() {
            return EngineStatus::Exhausted { hash_count: 0 };
        }
        if DEVICE_LOST.with(|lost| *lost.borrow() == Some(self.engine_id)) {
            return EngineStatus::DeviceLost { hash_count: 0 };
        }
        if range.start > range.end {
            return EngineStatus::Exhausted { hash_count: 0 };
        }
        if cancel.is_cancelled() {
            return EngineStatus::Cancelled { hash_count: 0 };
        }

        let device_index = ASSIGNED_DEVICE.with(|assigned| {
            let mut assigned_ref = assigned.borrow_mut();
            match *assigned_ref {
                Some((engine_id, index)) if engine_id == self.engine_id => index,
                _ => {
                    let index = if self.devices.len() == 1 {
                        0
                    } else {
                        self.device_counter.fetch_add(1, Ordering::SeqCst) % self.devices.len()
                    };
                    *assigned_ref = Some((self.engine_id, index));
                    log::info!(
                        target: "cuda_engine",
                        "Worker thread assigned to CUDA device {} ({})",
                        index,
                        self.devices[index].name
                    );
                    index
                }
            }
        });

        if let Err(e) = worker_buffers(self, device_index) {
            log::error!(target: "cuda_engine", "CUDA buffer setup failed: {e}");
            DEVICE_LOST.with(|lost| *lost.borrow_mut() = Some(self.engine_id));
            return EngineStatus::DeviceLost { hash_count: 0 };
        }

        let search_start = Instant::now();
        let mut total_hashes: u64 = 0;
        let mut current_start = range.start;
        let mut batch_num = 0u64;

        let duty = WORKER_BUFFERS.with(|cell| {
            take_gpu_duty_cycle(
                cell.borrow_mut()
                    .as_mut()
                    .expect("CUDA buffers initialized"),
            )
        });
        log::info!(
            target: "cuda_engine",
            "CUDA {} search started: range {}..{}, batch size: {} nonces{}",
            device_index,
            format_u512(range.start),
            format_u512(range.end),
            self.batch_size,
            duty.map_or(String::new(), |d| format!(", GPU busy {d:.1}% since previous search"))
        );

        while current_start <= range.end {
            if cancel.is_cancelled() {
                return EngineStatus::Cancelled {
                    hash_count: total_hashes,
                };
            }

            let remaining = range
                .end
                .saturating_sub(current_start)
                .saturating_add(U512::one());
            let headroom = nonces_until_low64_carry(current_start);
            let cap = remaining.min(headroom);
            let batch_size_u512 = U512::from(self.batch_size);
            let this_batch_size: u32 = if cap > batch_size_u512 {
                self.batch_size
            } else {
                cap.low_u32()
            };

            let batch_result = WORKER_BUFFERS.with(|cell| {
                let mut slot = cell.borrow_mut();
                let buffers = slot.as_mut().expect("CUDA buffers initialized");
                run_single_batch(buffers, ctx, current_start, this_batch_size)
            });

            match batch_result {
                BatchResult::Found {
                    candidate,
                    hash_count,
                } => {
                    total_hashes += hash_count;
                    return EngineStatus::Found {
                        candidate,
                        hash_count: total_hashes,
                        origin: FoundOrigin::Cuda,
                    };
                }
                BatchResult::NotFound { hash_count } => {
                    total_hashes += hash_count;
                }
                BatchResult::Rejected { logical_index } => {
                    let skip = logical_index + 1;
                    total_hashes += skip;
                    current_start = current_start.saturating_add(U512::from(skip));
                    continue;
                }
                BatchResult::DeviceLost => {
                    DEVICE_LOST.with(|lost| *lost.borrow_mut() = Some(self.engine_id));
                    WORKER_BUFFERS.with(|res| *res.borrow_mut() = None);
                    return EngineStatus::DeviceLost {
                        hash_count: total_hashes,
                    };
                }
            }

            current_start = current_start.saturating_add(U512::from(this_batch_size));
            batch_num += 1;

            if self.throttle_ms > 0 && current_start <= range.end {
                let sleep_interval =
                    std::time::Duration::from_millis((self.throttle_ms / 10).max(1));
                let mut remaining = std::time::Duration::from_millis(self.throttle_ms);
                while remaining > std::time::Duration::ZERO {
                    if cancel.is_cancelled() {
                        return EngineStatus::Cancelled {
                            hash_count: total_hashes,
                        };
                    }
                    let sleep_time = remaining.min(sleep_interval);
                    std::thread::sleep(sleep_time);
                    remaining = remaining.saturating_sub(sleep_time);
                }
            }

            if batch_num.is_multiple_of(10) {
                let elapsed = search_start.elapsed();
                let hash_rate = total_hashes as f64 / elapsed.as_secs_f64();
                log::debug!(
                    target: "cuda_engine",
                    "CUDA {} batch {} complete: {} hashes so far ({:.2}s, {})",
                    device_index,
                    batch_num,
                    total_hashes,
                    elapsed.as_secs_f64(),
                    format_hashrate(hash_rate)
                );
            }
        }

        EngineStatus::Exhausted {
            hash_count: total_hashes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::nvrtc::compile_ptx;
    use engine_cpu::AtomicBoolCancelCheck;
    use std::sync::atomic::AtomicBool;

    fn decode32(s: &str) -> [u8; 32] {
        hex::decode(s).unwrap().try_into().unwrap()
    }

    fn decode64(s: &str) -> [u8; 64] {
        hex::decode(s).unwrap().try_into().unwrap()
    }

    fn engine_or_skip() -> Option<CudaEngine> {
        engine_or_skip_with(1024)
    }

    fn engine_or_skip_with(batch_size: u32) -> Option<CudaEngine> {
        match CudaEngine::try_new(batch_size, 0) {
            Ok(e) => Some(e),
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("not available") || msg.contains("No CUDA devices") {
                    eprintln!("skipping CUDA test: {e}");
                    None
                } else {
                    panic!("CUDA engine init failed: {e}");
                }
            }
        }
    }

    #[test]
    fn low64_batches_stop_before_carry() {
        assert_eq!(nonces_until_low64_carry(U512::from(u64::MAX)), U512::one());
        assert_eq!(
            nonces_until_low64_carry(U512::from(u64::MAX - 7)),
            U512::from(8u64)
        );
        assert_eq!(
            nonces_until_low64_carry((U512::one() << 192) + U512::from(1u64)),
            (U512::one() << 64) - U512::one()
        );
    }

    #[test]
    fn cuda_matches_nonce_hash_golden_vectors() {
        let Some(engine) = engine_or_skip() else {
            return;
        };
        for (i, v) in pow_core::NONCE_HASH_KVS.iter().enumerate() {
            let header = decode32(v.header);
            let nonce = U512::from_big_endian(&decode64(v.nonce));
            let want = U512::from_big_endian(&decode64(v.hash));
            let got = engine
                .hash_nonces(header, nonce, 1)
                .unwrap_or_else(|e| panic!("kv {i}: CUDA hash_nonces failed: {e}"));
            assert_eq!(got.len(), 1, "kv {i}");
            assert_eq!(got[0], want, "kv {i}: CUDA hash != golden");
        }
        CudaEngine::clear_worker_resources();
    }

    #[test]
    fn cuda_search_matches_golden_target_boundaries() {
        let Some(engine) = engine_or_skip() else {
            return;
        };
        let cancel = AtomicBool::new(false);
        for (i, v) in pow_core::NONCE_HASH_KVS.iter().enumerate() {
            let nonce = U512::from_big_endian(&decode64(v.nonce));
            let hash = U512::from_big_endian(&decode64(v.hash));
            for target in [hash - U512::one(), hash, hash + U512::one()] {
                let ctx = JobContext {
                    header: decode32(v.header),
                    difficulty: U512::one(),
                    target,
                };
                let status = engine.search_range(
                    &ctx,
                    Range {
                        start: nonce,
                        end: nonce,
                    },
                    &AtomicBoolCancelCheck(&cancel),
                );
                match status {
                    EngineStatus::Found { candidate, .. } if hash < target => {
                        assert_eq!(candidate.nonce, nonce, "kv {i}");
                        assert_eq!(candidate.hash, hash, "kv {i}");
                    }
                    EngineStatus::Exhausted { hash_count: 1 } if hash >= target => {}
                    other => panic!("kv {i}, target {target}: unexpected {other:?}"),
                }
            }
        }
        CudaEngine::clear_worker_resources();
    }

    #[test]
    fn cuda_hash_batches_match_cpu_across_nonce_carries() {
        let Some(engine) = engine_or_skip() else {
            return;
        };
        let header = decode32(pow_core::NONCE_HASH_KVS[4].header);
        let ctx = engine.prepare_context(header, U512::one());
        let high = U512::from(0xdead_beef_cafeu64) << 256;
        for bits in [32, 64, 128, 224] {
            let start = high + (U512::one() << bits) - U512::from(128u64);
            let hashes = engine.hash_nonces(header, start, 257).unwrap();
            for (i, hash) in hashes.into_iter().enumerate() {
                let nonce = start + U512::from(i);
                assert_eq!(
                    hash,
                    pow_core::hash_from_nonce(&ctx, nonce),
                    "nonce {nonce}"
                );
            }
        }
        CudaEngine::clear_worker_resources();
    }

    const GOLDILOCKS_P: u128 = 0xffff_ffff_0000_0001;
    const EPS: u64 = 0xffff_ffff;

    /// Bit-exact model of the kernel's `reduce128` and its deviation from
    /// `value mod p`, so the GPU can be checked exactly and the deviation
    /// bounded on the host.
    fn reduce128_model(value: u128) -> (u64, u128) {
        let (r0, r1, r2, r3) = (
            value as u32,
            (value >> 32) as u32,
            (value >> 64) as u32,
            (value >> 96) as u32,
        );
        let low = ((r1 as u64) << 32) | r0 as u64;
        let (folded, carry) = low.overflowing_add(r2 as u64 * EPS);
        let (subtrahend, subtrahend_wrapped) = r3.overflowing_add(carry as u32);
        let (hi, hi_wrapped) = ((folded >> 32) as u32).overflowing_add(carry as u32);
        let shifted = ((hi as u64) << 32) | (folded as u32 as u64);
        let (result, borrowed) = shifted.overflowing_sub(subtrahend as u64);
        let mut error = 0u128;
        if subtrahend_wrapped {
            error += 1 << 32;
        }
        if hi_wrapped {
            error += GOLDILOCKS_P - EPS as u128;
        }
        if borrowed {
            error += EPS as u128;
        }
        (result, error % GOLDILOCKS_P)
    }

    /// Bit-exact model of the kernel's `gf64_add` and its deviation from `a + b mod p`.
    fn gf64_add_model(a: u64, b: u64) -> (u64, u128) {
        let (sum, carry) = a.overflowing_add(b);
        if !carry {
            return (sum, 0);
        }
        let (folded, wrapped) = sum.overflowing_add(EPS);
        (
            folded,
            if wrapped {
                GOLDILOCKS_P - EPS as u128
            } else {
                0
            },
        )
    }

    /// Edge cases built to hit every carry path, followed by 4096 random values.
    fn reduction_inputs() -> (Vec<u128>, Vec<u128>) {
        let edges = [
            0,
            1,
            0xffff_fffe,
            0xffff_ffff,
            0x1_0000_0000,
            GOLDILOCKS_P as u64 - 1,
            GOLDILOCKS_P as u64,
            GOLDILOCKS_P as u64 + 1,
            u64::MAX,
        ];
        let mut edge_values = Vec::new();
        for high in edges {
            for low in edges {
                edge_values.push(((high as u128) << 64) | low as u128);
            }
        }
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut random = Vec::new();
        for _ in 0..4096 {
            let mut value = 0u128;
            for _ in 0..2 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                value = (value << 64) | seed as u128;
            }
            random.push(value);
        }
        (edge_values, random)
    }

    #[test]
    fn reduction_models_deviate_only_on_carry_wraps() {
        let (edge_values, random) = reduction_inputs();
        for value in edge_values.iter().chain(&random) {
            let (result, error) = reduce128_model(*value);
            assert_eq!(
                result as u128 % GOLDILOCKS_P,
                (value % GOLDILOCKS_P + error) % GOLDILOCKS_P,
                "reduce128 model invariant broken for {value:032x}"
            );
        }
        assert_eq!(reduce128_model(1 << 96).1, EPS as u128, "2^96 borrows");
        let slips = random
            .iter()
            .filter(|v| reduce128_model(**v).1 != 0)
            .count();
        assert_eq!(
            slips,
            0,
            "slipped reductions among {} random inputs",
            random.len()
        );

        let mut random_slips = 0;
        for pair in random.windows(2) {
            let (a, b) = (pair[0] as u64, pair[1] as u64);
            let (result, error) = gf64_add_model(a, b);
            assert_eq!(
                result as u128 % GOLDILOCKS_P,
                (a as u128 + b as u128 + error) % GOLDILOCKS_P,
                "gf64_add model invariant broken for {a:016x} + {b:016x}"
            );
            random_slips += (error != 0) as usize;
        }
        assert_eq!(
            gf64_add_model(u64::MAX, u64::MAX).1,
            GOLDILOCKS_P - EPS as u128
        );
        assert_eq!(random_slips, 0, "slipped additions among random inputs");
    }

    #[test]
    fn reduction_slips_are_absent_in_millions_of_random_products() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let slips = (0..1 << 22)
            .filter(|_| reduce128_model(next() as u128 * next() as u128).1 != 0)
            .count();
        assert_eq!(slips, 0, "reducer slipped on random 64x64-bit products");
    }

    #[test]
    fn cuda_field_arithmetic_matches_models_bit_exactly() {
        let Some(engine) = engine_or_skip() else {
            return;
        };
        let (edge_values, random) = reduction_inputs();
        let values: Vec<u128> = edge_values.into_iter().chain(random).collect();
        let input: Vec<u32> = values
            .iter()
            .flat_map(|v| (0..4).map(move |i| (v >> (32 * i)) as u32))
            .collect();
        let source = format!(
            "{KERNEL_SRC}\n\
             extern \"C\" __global__ void arith_test(const u32 *input, u64 *reduced, u64 *added, u32 count) {{\n\
                 u32 i = blockIdx.x * blockDim.x + threadIdx.x;\n\
                 if (i >= count) return;\n\
                 reduced[i] = reduce128(input[4*i], input[4*i+1], input[4*i+2], input[4*i+3]);\n\
                 u64 a = ((u64)input[4*i+1] << 32) | input[4*i];\n\
                 u64 b = ((u64)input[4*i+3] << 32) | input[4*i+2];\n\
                 added[i] = gf64_add(a, b);\n\
             }}"
        );
        let device = &engine.devices[0];
        let module = device
            .ctx
            .load_module(compile_ptx(source).unwrap())
            .unwrap();
        let function = module.load_function("arith_test").unwrap();
        let stream = device.ctx.default_stream();
        let input = stream.clone_htod(&input).unwrap();
        let mut reduced = stream.alloc_zeros::<u64>(values.len()).unwrap();
        let mut added = stream.alloc_zeros::<u64>(values.len()).unwrap();
        let count = values.len() as u32;
        let mut launch = stream.launch_builder(&function);
        launch
            .arg(&input)
            .arg(&mut reduced)
            .arg(&mut added)
            .arg(&count);
        unsafe {
            launch.launch(LaunchConfig::for_num_elems(count)).unwrap();
        }
        stream.synchronize().unwrap();
        let reduced = stream.clone_dtoh(&reduced).unwrap();
        let added = stream.clone_dtoh(&added).unwrap();
        for (i, value) in values.iter().enumerate() {
            assert_eq!(
                reduced[i],
                reduce128_model(*value).0,
                "reduce128 of {value:032x}"
            );
            let (a, b) = (*value as u64, (*value >> 64) as u64);
            assert_eq!(
                added[i],
                gf64_add_model(a, b).0,
                "gf64_add of {a:016x} + {b:016x}"
            );
        }
        CudaEngine::clear_worker_resources();
    }

    #[test]
    fn cuda_found_batch_counts_every_dispatched_nonce() {
        let batch_size = 4_000_000u32;
        let Some(engine) = engine_or_skip_with(batch_size) else {
            return;
        };
        let total_threads =
            batch_size.div_ceil(THREADS_PER_BLOCK).min(MAX_BLOCKS) * THREADS_PER_BLOCK;
        assert!(
            batch_size > total_threads,
            "batch must span several nonces per thread"
        );
        let header = decode32(pow_core::NONCE_HASH_KVS[1].header);
        let ctx = engine.prepare_context(header, U512::one());
        let start = U512::from(0xfeed_face_0000_0000u64);
        let end = start + U512::from(batch_size - 1);
        let cancel = AtomicBool::new(false);
        match engine.search_range(&ctx, Range { start, end }, &AtomicBoolCancelCheck(&cancel)) {
            EngineStatus::Found {
                candidate,
                hash_count,
                ..
            } => {
                assert!(candidate.nonce >= start && candidate.nonce <= end);
                assert_eq!(
                    pow_core::hash_from_nonce(&ctx, candidate.nonce),
                    candidate.hash
                );
                assert_eq!(hash_count, batch_size as u64);
            }
            other => panic!("expected Found, got {other:?}"),
        }
        CudaEngine::clear_worker_resources();
    }

    #[test]
    fn cuda_search_finds_cpu_verified_solution() {
        let Some(engine) = engine_or_skip() else {
            return;
        };
        let header = decode32(pow_core::NONCE_HASH_KVS[2].header);
        let difficulty = U512::from(1u64);
        let ctx = engine.prepare_context(header, difficulty);
        let start = U512::from(0x1234_5678_90ab_cdefu64);
        let range = Range {
            start,
            end: start + U512::from(16u64),
        };
        let cancel = AtomicBool::new(false);
        match engine.search_range(&ctx, range, &AtomicBoolCancelCheck(&cancel)) {
            EngineStatus::Found { candidate, .. } => {
                let cpu = pow_core::hash_from_nonce(&ctx, candidate.nonce);
                assert_eq!(cpu, candidate.hash);
                assert!(cpu < ctx.target);
            }
            other => panic!("expected Found, got {other:?}"),
        }
        CudaEngine::clear_worker_resources();
    }
}

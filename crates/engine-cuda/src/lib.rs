#![deny(rust_2018_idioms)]

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use engine_cpu::{CancelCheck, Candidate, EngineStatus, FoundOrigin, MinerEngine, Range};
use pow_core::{format_hashrate, format_u512, JobContext};
use primitive_types::U512;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const KERNEL_SRC: &str = include_str!("kernels/mining.cu");
const THREADS_PER_BLOCK: u32 = 256;
const MAX_BLOCKS: u32 = 4096;
const RESULTS_U32S: usize = 1 + 16 + 16;

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
    target: CudaSlice<u32>,
    dispatch: CudaSlice<u32>,
    hashes: CudaSlice<u32>,
    mine: CudaFunction,
    hash: CudaFunction,
}

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

        let ptx = match silent_catch(|| compile_ptx(KERNEL_SRC)) {
            Ok(Ok(ptx)) => ptx,
            Ok(Err(e)) => {
                return Err(format!("NVRTC failed to compile the CUDA mining kernel: {e}").into());
            }
            Err(_) => {
                return Err(
                    "CUDA NVRTC library is not available (need libnvrtc from the CUDA toolkit)"
                        .into(),
                );
            }
        };

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
        for ordinal in 0..count as usize {
            let ctx = CudaContext::new(ordinal)
                .map_err(|e| format!("Failed to create CUDA context for device {ordinal}: {e}"))?;
            let name = ctx
                .name()
                .unwrap_or_else(|_| format!("cuda-device-{ordinal}"));
            let module = ctx
                .load_module(ptx.clone())
                .map_err(|e| format!("Failed to load CUDA mining module on {name}: {e}"))?;
            log::info!(target: "cuda_engine", "CUDA device {ordinal}: {name}");
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
        target: stream.alloc_zeros::<u32>(16)?,
        dispatch: stream.alloc_zeros::<u32>(3)?,
        hashes: stream.alloc_zeros::<u32>(hash_capacity.max(1) as usize * 16)?,
        mine,
        hash,
        stream,
    })
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
    DeviceLost,
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
    let start_limbs = pow_core::u512_to_le_u32s(batch_start);
    let nonce_be = batch_start.to_big_endian();
    let mid = pow_core::mining_midstate_u32s(ctx.header, nonce_be[..32].try_into().unwrap());
    let target = pow_core::u512_to_le_u32s(ctx.target);

    if let Err(e) = (|| {
        buffers
            .stream
            .memcpy_htod(&dispatch, &mut buffers.dispatch)?;
        buffers
            .stream
            .memcpy_htod(&start_limbs, &mut buffers.start_nonce)?;
        buffers.stream.memcpy_htod(&mid, &mut buffers.midstate)?;
        buffers.stream.memcpy_htod(&target, &mut buffers.target)?;
        buffers.stream.memset_zeros(&mut buffers.results)?;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = buffers.stream.launch_builder(&buffers.mine);
        builder.arg(&mut buffers.results);
        builder.arg(&buffers.midstate);
        builder.arg(&buffers.start_nonce);
        builder.arg(&buffers.target);
        builder.arg(&buffers.dispatch);
        unsafe {
            builder.launch(cfg)?;
        }
        buffers.stream.synchronize()?;
        Ok::<(), Box<dyn std::error::Error>>(())
    })() {
        log::error!(target: "cuda_engine", "CUDA batch failed: {e}");
        return BatchResult::DeviceLost;
    }

    let result_u32s = match buffers.stream.clone_dtoh(&buffers.results) {
        Ok(v) => v,
        Err(e) => {
            log::error!(target: "cuda_engine", "CUDA result copy failed: {e}");
            return BatchResult::DeviceLost;
        }
    };

    let dispatched = (total_threads as u64 * nonces_per_thread as u64).min(batch_size as u64);
    if result_u32s[0] != 0 {
        let mut nonce_limbs = [0u32; 16];
        let mut hash_limbs = [0u32; 16];
        nonce_limbs.copy_from_slice(&result_u32s[1..17]);
        hash_limbs.copy_from_slice(&result_u32s[17..33]);
        let nonce = pow_core::u512_from_le_u32s(nonce_limbs);
        let hash = pow_core::u512_from_le_u32s(hash_limbs);
        let hashes_computed = if nonce >= batch_start {
            let logical_index = (nonce - batch_start).as_u64();
            let winning_iteration = logical_index % (nonces_per_thread as u64);
            (total_threads as u64 * (winning_iteration + 1)).min(dispatched)
        } else {
            dispatched
        };
        return BatchResult::Found {
            candidate: Candidate {
                nonce,
                work: nonce.to_big_endian(),
                hash,
            },
            hash_count: hashes_computed,
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

        let search_start = std::time::Instant::now();
        let mut total_hashes: u64 = 0;
        let mut current_start = range.start;
        let mut batch_num = 0u64;

        log::info!(
            target: "cuda_engine",
            "CUDA {} search started: range {}..{}, batch size: {} nonces",
            device_index,
            format_u512(range.start),
            format_u512(range.end),
            self.batch_size
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
            let headroom =
                (U512::one() << 256) - (current_start & ((U512::one() << 256) - U512::one()));
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
    use engine_cpu::AtomicBoolCancelCheck;
    use std::sync::atomic::AtomicBool;

    fn decode32(s: &str) -> [u8; 32] {
        hex::decode(s).unwrap().try_into().unwrap()
    }

    fn decode64(s: &str) -> [u8; 64] {
        hex::decode(s).unwrap().try_into().unwrap()
    }

    fn engine_or_skip() -> Option<CudaEngine> {
        match CudaEngine::try_new(1024, 0) {
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

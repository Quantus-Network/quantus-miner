//! Mining service for Quantus external miners.
//!
//! This module provides the core mining functionality:
//! - Engine initialization (CPU and GPU)
//! - Persistent worker thread pool for efficient job processing
//! - QUIC-based communication with the node

#![deny(rust_2018_idioms)]
#![forbid(unsafe_code)]

pub mod quic;

use crossbeam_channel::{bounded, Receiver, Sender};
use engine_cpu::{EngineCandidate, EngineRange, MinerEngine};
use pow_core::format_u512;
use primitive_types::U512;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

/// Service runtime configuration provided by the CLI/binary.
#[derive(Clone, Debug)]
pub struct ServiceConfig {
    /// Address of the node to connect to (e.g., "127.0.0.1:9833").
    pub node_addr: std::net::SocketAddr,
    /// Number of CPU worker threads to use for mining (None = auto-detect)
    pub cpu_workers: Option<usize>,
    /// Number of GPU devices to use for mining (None = auto-detect)
    pub gpu_devices: Option<usize>,
}

/// Engine type for tracking metrics per compute type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineType {
    Cpu,
    Gpu,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            node_addr: "127.0.0.1:9833".parse().unwrap(),
            cpu_workers: None,
            gpu_devices: None,
        }
    }
}

/// Result from a single worker thread.
#[derive(Debug, Clone)]
pub struct WorkerResult {
    pub thread_id: usize,
    /// The type of engine (CPU or GPU) that produced this result.
    pub engine_type: EngineType,
    /// The winning candidate, if found.
    pub candidate: Option<MiningCandidate>,
    /// Number of hashes computed by this worker.
    pub hash_count: u64,
    /// Whether this worker has finished its range.
    pub completed: bool,
}

/// A successful mining candidate.
#[derive(Debug, Clone)]
pub struct MiningCandidate {
    pub nonce: U512,
    pub work: [u8; 64],
    pub hash: U512,
}

/// Generate a random U512 nonce starting point.
fn generate_random_nonce() -> U512 {
    let mut bytes = [0u8; 64];
    getrandom::getrandom(&mut bytes).expect("Failed to generate random bytes");
    U512::from_big_endian(&bytes)
}

// ---------------------------------------------------------------------------
// Persistent Worker Pool
// ---------------------------------------------------------------------------

/// A job to be executed by worker threads.
#[derive(Clone)]
pub struct MiningJob {
    /// Job context with header hash and difficulty
    pub ctx: pow_core::JobContext,
    /// Epoch number to detect stale results after job transitions
    pub epoch: u64,
}

/// Persistent worker thread pool that keeps threads alive between jobs.
///
/// This avoids the overhead of spawning new threads and reinitializing
/// GPU resources for each mining job.
pub struct WorkerPool {
    /// Senders for dispatching jobs to workers (one per worker)
    job_senders: Vec<Sender<MiningJob>>,
    /// Receiver for collecting results from all workers
    result_rx: Receiver<WorkerResult>,
    /// Shared cancellation flag for all workers
    cancel_flag: Arc<AtomicBool>,
    /// Job epoch counter - incremented on each new job to detect stale results
    job_epoch: Arc<AtomicU64>,
    /// Thread handles (for cleanup)
    _handles: Vec<thread::JoinHandle<()>>,
    /// Number of CPU workers
    cpu_worker_count: usize,
    /// Number of GPU workers
    gpu_worker_count: usize,
}

impl WorkerPool {
    /// Create a new persistent worker pool.
    pub fn new(
        cpu_engine: Option<Arc<dyn MinerEngine>>,
        gpu_engine: Option<Arc<dyn MinerEngine>>,
        cpu_workers: usize,
        gpu_devices: usize,
    ) -> Self {
        let total_workers = cpu_workers + gpu_devices;
        let (result_tx, result_rx) = bounded(total_workers * 64);
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let job_epoch = Arc::new(AtomicU64::new(0));

        let mut job_senders = Vec::with_capacity(total_workers);
        let mut handles = Vec::with_capacity(total_workers);
        let mut thread_id = 0;

        log::info!(
            "Creating persistent worker pool: {} CPU + {} GPU workers",
            cpu_workers,
            gpu_devices
        );

        // Spawn CPU workers
        if cpu_workers > 0 {
            if let Some(ref engine) = cpu_engine {
                for _ in 0..cpu_workers {
                    let (job_tx, job_rx) = bounded::<MiningJob>(1);
                    job_senders.push(job_tx);

                    let eng = engine.clone();
                    let tx = result_tx.clone();
                    let cancel = cancel_flag.clone();
                    let epoch = job_epoch.clone();
                    let tid = thread_id;

                    let handle = thread::spawn(move || {
                        worker_loop(tid, EngineType::Cpu, eng, job_rx, tx, cancel, epoch);
                    });
                    handles.push(handle);
                    thread_id += 1;
                }
            }
        }

        // Spawn GPU workers
        if gpu_devices > 0 {
            if let Some(ref engine) = gpu_engine {
                for _ in 0..gpu_devices {
                    let (job_tx, job_rx) = bounded::<MiningJob>(1);
                    job_senders.push(job_tx);

                    let eng = engine.clone();
                    let tx = result_tx.clone();
                    let cancel = cancel_flag.clone();
                    let epoch = job_epoch.clone();
                    let tid = thread_id;

                    let handle = thread::spawn(move || {
                        worker_loop(tid, EngineType::Gpu, eng, job_rx, tx, cancel, epoch);
                    });
                    handles.push(handle);
                    thread_id += 1;
                }
            }
        }

        Self {
            job_senders,
            result_rx,
            cancel_flag,
            job_epoch,
            _handles: handles,
            cpu_worker_count: cpu_workers,
            gpu_worker_count: gpu_devices,
        }
    }

    /// Start a new mining job. Cancels any currently running job first.
    pub fn start_job(&self, header_hash: [u8; 32], difficulty: U512) {
        // Increment epoch FIRST - this ensures any in-flight results from the old job
        // will be detected as stale when workers check the epoch before sending results
        let new_epoch = self.job_epoch.fetch_add(1, Ordering::SeqCst) + 1;

        // Cancel any running job
        self.cancel_flag.store(true, Ordering::SeqCst);

        // Brief pause to let workers see the cancellation
        std::thread::sleep(std::time::Duration::from_millis(1));

        // Reset cancel flag for new job
        self.cancel_flag.store(false, Ordering::SeqCst);

        // Create job context (shared across all workers)
        let ctx = pow_core::JobContext::new(header_hash, difficulty);
        let job = MiningJob {
            ctx,
            epoch: new_epoch,
        };

        // Dispatch job to all workers
        for tx in &self.job_senders {
            // Non-blocking send - if worker is still processing old job, it will
            // see the cancel flag and exit soon
            let _ = tx.try_send(job.clone());
        }

        log::debug!(
            "Job dispatched to {} workers (epoch {})",
            self.job_senders.len(),
            new_epoch
        );
    }

    /// Cancel the current job.
    pub fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
    }

    /// Get the result receiver for collecting worker results.
    pub fn result_receiver(&self) -> &Receiver<WorkerResult> {
        &self.result_rx
    }

    /// Get the shared cancel flag.
    pub fn cancel_flag(&self) -> &Arc<AtomicBool> {
        &self.cancel_flag
    }

    /// Total number of workers.
    pub fn worker_count(&self) -> usize {
        self.job_senders.len()
    }

    /// Number of CPU workers.
    pub fn cpu_worker_count(&self) -> usize {
        self.cpu_worker_count
    }

    /// Number of GPU workers.
    pub fn gpu_worker_count(&self) -> usize {
        self.gpu_worker_count
    }
}

/// Main loop for a persistent worker thread.
fn worker_loop(
    thread_id: usize,
    engine_type: EngineType,
    engine: Arc<dyn MinerEngine>,
    job_rx: Receiver<MiningJob>,
    result_tx: Sender<WorkerResult>,
    cancel_flag: Arc<AtomicBool>,
    current_epoch: Arc<AtomicU64>,
) {
    let type_str = match engine_type {
        EngineType::Cpu => "CPU",
        EngineType::Gpu => "GPU",
    };

    log::info!("{} worker {} started (persistent)", type_str, thread_id);

    // Main job processing loop
    loop {
        // Wait for a job
        let job = match job_rx.recv() {
            Ok(job) => job,
            Err(_) => {
                // Channel closed, pool is shutting down
                log::debug!("{} worker {} shutting down", type_str, thread_id);
                break;
            }
        };

        // Capture the job's epoch for later validation
        let job_epoch = job.epoch;

        // Check if already cancelled before starting
        if cancel_flag.load(Ordering::Relaxed) {
            log::debug!("{} worker {} skipping cancelled job", type_str, thread_id);
            continue;
        }

        // Generate random starting nonce for this job
        let start = generate_random_nonce();
        let end = U512::MAX;

        log::debug!(
            "{} worker {} processing job (epoch {}): range {} to {}",
            type_str,
            thread_id,
            job_epoch,
            format_u512(start),
            format_u512(end)
        );

        // Execute the search
        let range = EngineRange { start, end };
        let result = engine.search_range(&job.ctx, range, &cancel_flag);

        // Check if epoch changed during search - if so, this result is stale
        let actual_epoch = current_epoch.load(Ordering::SeqCst);
        if actual_epoch != job_epoch {
            log::debug!(
                "{} worker {} discarding stale result (job epoch {} != current epoch {})",
                type_str,
                thread_id,
                job_epoch,
                actual_epoch
            );
            // Still send hash count for metrics, but without the candidate
            let hash_count = match result {
                engine_cpu::EngineStatus::Found { hash_count, .. } => hash_count,
                engine_cpu::EngineStatus::Exhausted { hash_count } => hash_count,
                engine_cpu::EngineStatus::Cancelled { hash_count } => hash_count,
                engine_cpu::EngineStatus::Running { .. } => 0,
            };
            let _ = result_tx.try_send(WorkerResult {
                thread_id,
                engine_type,
                candidate: None, // Discard the stale candidate
                hash_count,
                completed: true,
            });
            continue;
        }

        // Process result
        let (candidate, hash_count) = match result {
            engine_cpu::EngineStatus::Found {
                candidate: EngineCandidate { nonce, work, hash },
                hash_count,
                ..
            } => {
                log::info!(
                    "🎉 {} worker {} found solution! Nonce: {}, Hash: {} (epoch {})",
                    type_str,
                    thread_id,
                    format_u512(nonce),
                    format_u512(hash),
                    job_epoch
                );
                (Some(MiningCandidate { nonce, work, hash }), hash_count)
            }
            engine_cpu::EngineStatus::Exhausted { hash_count } => {
                log::debug!(
                    "{} worker {} exhausted range ({} hashes)",
                    type_str,
                    thread_id,
                    hash_count
                );
                (None, hash_count)
            }
            engine_cpu::EngineStatus::Cancelled { hash_count } => {
                log::debug!(
                    "{} worker {} cancelled ({} hashes)",
                    type_str,
                    thread_id,
                    hash_count
                );
                (None, hash_count)
            }
            engine_cpu::EngineStatus::Running { .. } => {
                // Should not happen for synchronous search
                (None, 0)
            }
        };

        // Send result (non-blocking to avoid deadlock if receiver is full)
        let _ = result_tx.try_send(WorkerResult {
            thread_id,
            engine_type,
            candidate,
            hash_count,
            completed: true,
        });
    }

    // Clean up GPU resources on thread exit
    if engine_type == EngineType::Gpu {
        engine_gpu::GpuEngine::clear_worker_resources();
    }

    log::debug!("{} worker {} exited", type_str, thread_id);
}

/// Resolve GPU configuration and initialize the engine.
pub fn resolve_gpu_configuration(
    requested_devices: Option<usize>,
) -> anyhow::Result<(Option<Arc<dyn MinerEngine>>, usize)> {
    // Explicit 0 means no GPU
    if requested_devices == Some(0) {
        return Ok((None, 0));
    }

    // Try to initialize GPU engine
    let engine = match engine_gpu::GpuEngine::try_new() {
        Ok(e) => e,
        Err(e) => {
            if requested_devices.is_some() {
                anyhow::bail!("Failed to initialize GPU engine: {}", e);
            }
            log::info!("No GPU available: {}", e);
            return Ok((None, 0));
        }
    };

    let available = engine.device_count();
    let count = match requested_devices {
        Some(n) if n > available => {
            anyhow::bail!(
                "Requested {} GPU devices but only {} available",
                n,
                available
            );
        }
        Some(n) => n,
        None if available == 0 => {
            log::info!("No GPU devices found");
            return Ok((None, 0));
        }
        None => {
            log::info!("Auto-detected {} GPU device(s)", available);
            available
        }
    };

    Ok((Some(Arc::new(engine)), count))
}

/// Start the miner service with the given configuration.
pub async fn run(config: ServiceConfig) -> anyhow::Result<()> {
    // Detect effective CPU count
    let effective_cpus = num_cpus::get().max(1);

    // Resolve GPU configuration
    let (gpu_engine, gpu_devices) = resolve_gpu_configuration(config.gpu_devices)?;

    // Resolve CPU workers
    let cpu_workers = config.cpu_workers.unwrap_or_else(|| {
        let default = (effective_cpus / 2).max(1);
        log::info!(
            "Auto-detected {} CPU workers (of {} available)",
            default,
            effective_cpus
        );
        default
    });

    // Validate: must have at least one worker
    if cpu_workers == 0 && gpu_devices == 0 {
        anyhow::bail!("No workers configured. Specify --cpu-workers > 0 or --gpu-devices > 0.");
    }

    // Create CPU engine
    let cpu_engine: Option<Arc<dyn MinerEngine>> = if cpu_workers > 0 {
        Some(Arc::new(engine_cpu::FastCpuEngine::new()))
    } else {
        None
    };

    // Log configuration
    log::info!(
        "🚀 Mining configuration: {} CPU workers, {} GPU devices",
        cpu_workers,
        gpu_devices
    );

    if let Some(ref engine) = cpu_engine {
        log::info!("🖥️  CPU engine: {}", engine.name());
    }
    if let Some(ref engine) = gpu_engine {
        log::info!("🎮 GPU engine: {}", engine.name());
    }

    log::info!(
        "⛏️  Mining service ready with {} total workers",
        cpu_workers + gpu_devices
    );

    // Connect to node and start mining
    log::info!("🌐 Connecting to node at {}", config.node_addr);
    quic::connect_and_mine(
        config.node_addr,
        cpu_engine,
        gpu_engine,
        cpu_workers,
        gpu_devices,
    )
    .await
}

//! Mining service for Quantus external miners.
//!
//! This module provides the core mining functionality:
//! - Engine initialization (CPU and GPU)
//! - Worker thread spawning and coordination
//! - QUIC-based communication with the node

#![deny(rust_2018_idioms)]
#![forbid(unsafe_code)]

pub mod quic;

use crossbeam_channel::{bounded, Receiver, Sender};
use engine_cpu::{EngineCandidate, EngineRange, MinerEngine};
use primitive_types::U512;
use std::sync::atomic::AtomicBool;
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
    /// Optional target duration for GPU batches in milliseconds.
    pub gpu_batch_duration_ms: Option<u64>,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            node_addr: "127.0.0.1:9833".parse().unwrap(),
            cpu_workers: None,
            gpu_devices: None,
            gpu_batch_duration_ms: None,
        }
    }
}

/// Result from a single worker thread.
#[derive(Debug, Clone)]
pub struct WorkerResult {
    pub thread_id: usize,
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

/// Spawn mining worker threads and return a receiver for results.
///
/// Workers will search the nonce range and send results via the channel.
/// Set `cancel_flag` to true to stop all workers early.
pub fn spawn_mining_workers(
    header_hash: [u8; 32],
    difficulty: U512,
    nonce_start: U512,
    nonce_end: U512,
    cpu_engine: Option<Arc<dyn MinerEngine>>,
    gpu_engine: Option<Arc<dyn MinerEngine>>,
    cpu_workers: usize,
    gpu_devices: usize,
    cancel_flag: Arc<AtomicBool>,
) -> (Receiver<WorkerResult>, Vec<thread::JoinHandle<()>>) {
    let total_workers = cpu_workers + gpu_devices;
    let chan_capacity = total_workers.saturating_mul(64).max(256);
    let (sender, receiver) = bounded(chan_capacity);

    let total_range = nonce_end
        .saturating_sub(nonce_start)
        .saturating_add(U512::from(1u64));

    let mut current_start = nonce_start;
    let mut thread_id = 0;
    let mut handles = Vec::with_capacity(total_workers);

    log::info!(
        "Starting mining with {} CPU + {} GPU workers",
        cpu_workers,
        gpu_devices
    );

    // Spawn CPU workers
    if cpu_workers > 0 {
        if let Some(ref engine) = cpu_engine {
            let ctx = engine.prepare_context(header_hash, difficulty);
            let cpu_range_end = current_start.saturating_add(
                total_range
                    .saturating_mul(U512::from(cpu_workers))
                    / U512::from(total_workers),
            );
            let partitions = compute_partitions(current_start, cpu_range_end.saturating_sub(U512::from(1u64)), cpu_workers);
            current_start = cpu_range_end;

            for (start, end) in partitions {
                let cancel = cancel_flag.clone();
                let tx = sender.clone();
                let ctx = ctx.clone();
                let eng = engine.clone();
                let tid = thread_id;

                let handle = thread::spawn(move || {
                    run_worker(tid, "CPU", eng.as_ref(), ctx, start, end, cancel, tx);
                });
                handles.push(handle);
                thread_id += 1;
            }
        }
    }

    // Spawn GPU workers
    if gpu_devices > 0 {
        if let Some(ref engine) = gpu_engine {
            let ctx = engine.prepare_context(header_hash, difficulty);
            let partitions = compute_partitions(current_start, nonce_end, gpu_devices);

            for (start, end) in partitions {
                let cancel = cancel_flag.clone();
                let tx = sender.clone();
                let ctx = ctx.clone();
                let eng = engine.clone();
                let tid = thread_id;

                let handle = thread::spawn(move || {
                    run_worker(tid, "GPU", eng.as_ref(), ctx, start, end, cancel, tx);
                });
                handles.push(handle);
                thread_id += 1;
            }
        }
    }

    (receiver, handles)
}

/// Run a single mining worker.
fn run_worker(
    thread_id: usize,
    engine_type: &str,
    engine: &dyn MinerEngine,
    ctx: pow_core::JobContext,
    start: U512,
    end: U512,
    cancel_flag: Arc<AtomicBool>,
    sender: Sender<WorkerResult>,
) {
    log::info!(
        "{} thread {} started: range {} to {}",
        engine_type,
        thread_id,
        start,
        end
    );

    let range = EngineRange { start, end };
    let result = engine.search_range(&ctx, range, &cancel_flag);

    let (candidate, hash_count) = match result {
        engine_cpu::EngineStatus::Found {
            candidate: EngineCandidate { nonce, work, hash },
            hash_count,
            ..
        } => {
            log::info!(
                "🎉 {} thread {} found solution! Nonce: {}, Hash: {:x}",
                engine_type,
                thread_id,
                nonce,
                hash
            );
            (
                Some(MiningCandidate { nonce, work, hash }),
                hash_count,
            )
        }
        engine_cpu::EngineStatus::Exhausted { hash_count } => {
            log::debug!(
                "{} thread {} exhausted range ({} hashes)",
                engine_type,
                thread_id,
                hash_count
            );
            (None, hash_count)
        }
        engine_cpu::EngineStatus::Cancelled { hash_count } => {
            log::debug!(
                "{} thread {} cancelled ({} hashes)",
                engine_type,
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

    let _ = sender.try_send(WorkerResult {
        thread_id,
        candidate,
        hash_count,
        completed: true,
    });

    // Clear GPU thread-local resources
    if engine_type == "GPU" {
        engine_gpu::GpuEngine::clear_worker_resources();
    }

    log::debug!("{} thread {} finished", engine_type, thread_id);
}

/// Compute partitions of a nonce range for the given number of workers.
fn compute_partitions(start: U512, end: U512, workers: usize) -> Vec<(U512, U512)> {
    if workers == 0 {
        return vec![];
    }

    let total_range = end.saturating_sub(start).saturating_add(U512::from(1u64));
    let workers_u512 = U512::from(workers as u64);
    let range_per = total_range / workers_u512;
    let remainder = total_range % workers_u512;

    let mut partitions = Vec::with_capacity(workers);
    let mut current = start;

    for i in 0..workers {
        let mut partition_end = current.saturating_add(range_per).saturating_sub(U512::from(1u64));
        if i == workers - 1 {
            partition_end = partition_end.saturating_add(remainder);
        }
        if partition_end > end {
            partition_end = end;
        }
        partitions.push((current, partition_end));
        current = partition_end.saturating_add(U512::from(1u64));
    }

    partitions
}

/// Resolve GPU configuration and initialize the engine.
pub fn resolve_gpu_configuration(
    requested_devices: Option<usize>,
    gpu_batch_duration_ms: Option<u64>,
) -> anyhow::Result<(Option<Arc<dyn MinerEngine>>, usize)> {
    // Explicit 0 means no GPU
    if requested_devices == Some(0) {
        return Ok((None, 0));
    }

    // Try to initialize GPU engine
    let duration = std::time::Duration::from_millis(gpu_batch_duration_ms.unwrap_or(3000));
    let engine = match engine_gpu::GpuEngine::try_new(duration) {
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
            anyhow::bail!("Requested {} GPU devices but only {} available", n, available);
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
    let (gpu_engine, gpu_devices) =
        resolve_gpu_configuration(config.gpu_devices, config.gpu_batch_duration_ms)?;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_partitions_single_worker() {
        let partitions = compute_partitions(U512::from(0u64), U512::from(99u64), 1);
        assert_eq!(partitions.len(), 1);
        assert_eq!(partitions[0], (U512::from(0u64), U512::from(99u64)));
    }

    #[test]
    fn test_compute_partitions_multiple_workers() {
        let partitions = compute_partitions(U512::from(0u64), U512::from(99u64), 4);
        assert_eq!(partitions.len(), 4);
        // Each gets 25 nonces
        assert_eq!(partitions[0], (U512::from(0u64), U512::from(24u64)));
        assert_eq!(partitions[1], (U512::from(25u64), U512::from(49u64)));
        assert_eq!(partitions[2], (U512::from(50u64), U512::from(74u64)));
        assert_eq!(partitions[3], (U512::from(75u64), U512::from(99u64)));
    }

    #[test]
    fn test_compute_partitions_zero_workers() {
        let partitions = compute_partitions(U512::from(0u64), U512::from(99u64), 0);
        assert!(partitions.is_empty());
    }
}

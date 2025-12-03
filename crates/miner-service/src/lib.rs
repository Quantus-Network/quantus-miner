#![deny(rust_2018_idioms)]
#![forbid(unsafe_code)]

use crossbeam_channel::{bounded, Receiver, Sender};
use engine_cpu::{EngineCandidate, EngineRange, MinerEngine};
use primitive_types::U512;
use quantus_miner_api::*;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use tokio::sync::Mutex;
use warp::{Filter, Rejection, Reply};

const THREAD_RATE_EMA_ALPHA: f64 = 0.2;

/// Service runtime configuration provided by the CLI/binary.
#[derive(Clone, Debug)]
pub struct ServiceConfig {
    /// Port for the HTTP miner API.
    pub port: u16,
    /// Number of worker threads (logical CPUs) to use for mining (defaults to all available if None).
    pub workers: Option<usize>,
    /// Optional metrics port. When Some, metrics endpoint starts; when None, metrics are disabled.
    pub metrics_port: Option<u16>,
    /// Target milliseconds for per-thread progress updates (chunking). If None, defaults to 2000ms.
    pub progress_chunk_ms: Option<u64>,
    /// Optional starting value for the manipulator engine's solved-blocks throttle index.
    pub manip_solved_blocks: Option<u64>,
    /// Optional base sleep per batch in nanoseconds for manipulator engine (default 500_000ns).
    pub manip_base_delay_ns: Option<u64>,
    /// Optional number of nonce attempts between sleeps for manipulator engine (default 10_000).
    pub manip_step_batch: Option<u64>,
    /// Optional cap on solved-blocks throttle index for manipulator engine.
    pub manip_throttle_cap: Option<u64>,
    /// Engine selection (future use). For now, CPU baseline/fast engines are supported.
    pub engine: EngineSelection,
}

/// Engine selection enum for future extensibility.
#[derive(Clone, Debug)]
pub enum EngineSelection {
    CpuBaseline,
    CpuFast,
    CpuChainManipulator,
    Gpu,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            port: 9833,
            workers: None,
            metrics_port: None,
            progress_chunk_ms: None,
            manip_solved_blocks: None,
            manip_base_delay_ns: None,
            manip_step_batch: None,
            manip_throttle_cap: None,
            engine: EngineSelection::CpuBaseline,
        }
    }
}

impl fmt::Display for ServiceConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let engine = match self.engine {
            EngineSelection::CpuBaseline => "cpu-baseline",
            EngineSelection::CpuFast => "cpu-fast",
            EngineSelection::CpuChainManipulator => "cpu-chain-manipulator",
            EngineSelection::Gpu => "gpu",
        };
        write!(
            f,
            "port={}, workers={:?}, engine={}, metrics_port={:?}, progress_chunk_ms={:?}, manip_solved_blocks={:?}, manip_base_delay_ns={:?}, manip_step_batch={:?}, manip_throttle_cap={:?}",
            self.port,
            self.workers,
            engine,
            self.metrics_port,
            self.progress_chunk_ms,
            self.manip_solved_blocks,
            self.manip_base_delay_ns,
            self.manip_step_batch,
            self.manip_throttle_cap
        )
    }
}

/// The core service state: job registry, chosen engine, and thread configuration.
#[derive(Clone)]
pub struct MiningService {
    pub jobs: Arc<Mutex<HashMap<String, MiningJob>>>,
    pub workers: usize,
    pub engine: Arc<dyn MinerEngine>,
    /// Target milliseconds for per-thread progress updates (chunking).
    pub progress_chunk_ms: u64,
    /// Gauge of currently running jobs (for metrics)
    pub active_jobs_gauge: Arc<tokio::sync::Mutex<i64>>,
}

impl MiningService {
    pub fn new(workers: usize, engine: Arc<dyn MinerEngine>, progress_chunk_ms: u64) -> Self {
        Self {
            jobs: Arc::new(Mutex::new(HashMap::new())),
            workers,
            engine,
            progress_chunk_ms,
            active_jobs_gauge: Arc::new(tokio::sync::Mutex::new(0)),
        }
    }

    pub async fn add_job(&self, job_id: String, mut job: MiningJob) -> Result<(), String> {
        let mut jobs = self.jobs.lock().await;
        if jobs.contains_key(&job_id) {
            log::warn!("Attempted to add duplicate job ID: {job_id}");
            return Err("Job already exists".to_string());
        }

        log::debug!(target: "miner", "Adding job: {} with {} workers", job_id, self.workers);
        job.job_id = Some(job_id.clone());
        #[cfg(feature = "metrics")]
        {
            metrics::set_job_status_gauge(self.engine.name(), &job_id, "running", 1);
            metrics::set_job_status_gauge(self.engine.name(), &job_id, "completed", 0);
            metrics::set_job_status_gauge(self.engine.name(), &job_id, "failed", 0);
            metrics::set_job_status_gauge(self.engine.name(), &job_id, "cancelled", 0);
            // increment active jobs
            {
                let mut g = self.active_jobs_gauge.lock().await;
                *g += 1;
                metrics::set_active_jobs(*g);
            }
        }
        job.start_mining(self.engine.clone(), self.workers, self.progress_chunk_ms);
        jobs.insert(job_id, job);
        Ok(())
    }

    pub async fn get_job(&self, job_id: &str) -> Option<MiningJob> {
        let jobs = self.jobs.lock().await;
        jobs.get(job_id).cloned()
    }

    pub async fn mark_job_result_served(&self, job_id: &str) {
        let mut jobs = self.jobs.lock().await;
        if let Some(job) = jobs.get_mut(job_id) {
            if !job.result_served {
                job.result_served = true;
                log::info!(target: "miner", "Result served and marked for job: {job_id}");
                #[cfg(feature = "metrics")]
                {
                    // reuse existing counter to trace served events
                    metrics::inc_mine_requests("result_served");
                }
            }
        }
    }

    pub async fn remove_job(&self, job_id: &str) -> Option<MiningJob> {
        let mut jobs = self.jobs.lock().await;
        if let Some(mut job) = jobs.remove(job_id) {
            log::debug!(target: "miner", "Removing job: {job_id}");
            job.cancel();
            Some(job)
        } else {
            None
        }
    }

    pub async fn cancel_job(&self, job_id: &str) -> bool {
        let mut jobs = self.jobs.lock().await;
        if let Some(job) = jobs.get_mut(job_id) {
            job.cancel();
            true
        } else {
            false
        }
    }

    /// Periodically polls running jobs for results and advances their status.
    pub async fn start_mining_loop(&self) {
        let jobs = self.jobs.clone();
        log::debug!(target: "miner", "Starting mining loop...");

        tokio::spawn(async move {
            let mut last_watchdog = std::time::Instant::now();
            let mut iter: u64 = 0;
            loop {
                iter += 1;
                let mut jobs_guard = jobs.lock().await;

                jobs_guard.retain(|job_id, job| {
                    if job.status == JobStatus::Running && job.update_from_results() {
                        log::debug!(target: "miner",
                            "Job {} finished with status {:?}, hashes: {}, time: {:?}",
                            job_id,
                            job.status,
                            job.total_hash_count,
                            job.start_time.elapsed()
                        );
                    }

                    // Retain running jobs, completed-but-not-yet-served jobs, or anything recent (<5m)
                    let retain = job.status == JobStatus::Running
                        || (job.status == JobStatus::Completed && !job.result_served)
                        || job.start_time.elapsed().as_secs() < 300;
                    if !retain {
                        log::debug!(target: "miner", "Cleaning up old job {job_id}");
                    }
                    retain
                });

                #[cfg(feature = "metrics")]
                {
                    // Aggregate and per-job hash-rate estimate across running jobs (nonces/sec)
                    // Aggregate hash-rate across running jobs from last recorded per-job rates
                    let mut total_rate = 0.0;
                    let mut running_jobs = 0i64;
                    for (job_id, job) in jobs_guard.iter() {
                        if job.status == JobStatus::Running {
                            running_jobs += 1;
                            total_rate += job.last_hash_rate;
                            metrics::set_job_hash_rate(job.engine_name, job_id, job.last_hash_rate);
                        }
                    }
                    metrics::set_hash_rate(total_rate);
                    metrics::set_active_jobs(running_jobs);
                }
                let do_watchdog = last_watchdog.elapsed().as_secs() >= 30;
                let (total, running, completed, failed, cancelled) = if do_watchdog {
                    let mut running = 0usize;
                    let mut completed = 0usize;
                    let mut cancelled = 0usize;
                    let mut failed = 0usize;
                    let total = jobs_guard.len();
                    for (_id, job) in jobs_guard.iter() {
                        match job.status {
                            JobStatus::Running => running += 1,
                            JobStatus::Completed => completed += 1,
                            JobStatus::Cancelled => cancelled += 1,
                            JobStatus::Failed => failed += 1,
                        }
                    }
                    (total, running, completed, failed, cancelled)
                } else {
                    (0, 0, 0, 0, 0)
                };
                drop(jobs_guard);
                if do_watchdog {
                    log::info!(
                        target: "miner",
                        "Watchdog: jobs total={}, running={}, completed={}, failed={}, cancelled={}, loop_iter={}",
                        total, running, completed, failed, cancelled, iter
                    );
                    last_watchdog = std::time::Instant::now();
                }
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        });
    }
}

/// Mining job status enumeration for orchestration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Aggregated best result for a job.
#[derive(Debug, Clone)]
pub struct MiningJobResult {
    pub nonce: U512,
    pub work: [u8; 64],
    pub hash: U512,
}

/// Mining job data structure stored in the service.
#[derive(Debug)]
pub struct MiningJob {
    pub header_hash: [u8; 32],
    pub difficulty: U512,
    pub nonce_start: U512,
    pub nonce_end: U512,

    pub status: JobStatus,
    pub start_time: Instant,
    pub total_hash_count: u64,
    pub last_hash_rate: f64,
    pub best_result: Option<MiningJobResult>,

    pub engine_name: &'static str,
    pub job_id: Option<String>,
    pub cancel_flag: Arc<AtomicBool>,
    pub result_receiver: Option<Receiver<ThreadResult>>,
    pub thread_handles: Vec<thread::JoinHandle<()>>,
    pub thread_last_update: std::collections::HashMap<usize, std::time::Instant>,
    pub thread_rate_ema: std::collections::HashMap<usize, f64>,
    completed_threads: usize,
    pub result_served: bool,
}

impl Clone for MiningJob {
    fn clone(&self) -> Self {
        MiningJob {
            header_hash: self.header_hash,
            difficulty: self.difficulty,
            nonce_start: self.nonce_start,
            nonce_end: self.nonce_end,

            status: self.status.clone(),
            start_time: self.start_time,
            total_hash_count: self.total_hash_count,
            last_hash_rate: self.last_hash_rate,
            best_result: self.best_result.clone(),
            engine_name: self.engine_name,
            job_id: self.job_id.clone(),

            cancel_flag: self.cancel_flag.clone(),
            // Do not clone crossbeam receiver or thread handles; they are runtime artifacts.
            result_receiver: None,
            thread_handles: Vec::new(),
            thread_last_update: self.thread_last_update.clone(),
            thread_rate_ema: self.thread_rate_ema.clone(),
            completed_threads: self.completed_threads,
            result_served: self.result_served,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ThreadResult {
    thread_id: usize,
    result: Option<MiningJobResult>,
    hash_count: u64,
    origin: Option<engine_cpu::FoundOrigin>,
    completed: bool,
}

impl MiningJob {
    pub fn new(
        header_hash: [u8; 32],
        difficulty: U512,
        nonce_start: U512,
        nonce_end: U512,
    ) -> Self {
        MiningJob {
            header_hash,
            difficulty,
            nonce_start,
            nonce_end,
            status: JobStatus::Running,
            start_time: Instant::now(),
            total_hash_count: 0,
            last_hash_rate: 0.0,
            best_result: None,
            engine_name: "unknown",
            job_id: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            result_receiver: None,
            thread_handles: Vec::new(),
            thread_last_update: std::collections::HashMap::new(),
            thread_rate_ema: std::collections::HashMap::new(),
            completed_threads: 0,
            result_served: false,
        }
    }

    pub fn start_mining(
        &mut self,
        engine: Arc<dyn MinerEngine>,
        workers: usize,
        progress_chunk_ms: u64,
    ) {
        let chan_capacity = std::env::var("MINER_RESULT_CHANNEL_CAP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| workers.saturating_mul(64).max(256));
        let (sender, receiver) = bounded(chan_capacity);
        self.result_receiver = Some(receiver);
        self.engine_name = engine.name();

        // Partition nonce range safely
        let partitions = compute_partitions(self.nonce_start, self.nonce_end, workers);

        log::debug!(
            target: "miner",
            "Starting mining with {} workers, total range: {} ({} partitions)",
            workers,
            partitions.total_range,
            partitions.ranges.len()
        );

        // Prepare shared job context once per job.
        let ctx = engine.prepare_context(self.header_hash, self.difficulty);

        for (thread_id, (start, end)) in partitions.ranges.into_iter().enumerate() {
            let cancel_flag = self.cancel_flag.clone();
            let sender = sender.clone();
            let ctx = ctx.clone();
            let engine = engine.clone();

            let job_id = self.job_id.clone().unwrap_or_else(|| "unknown".to_string());
            let handle = thread::spawn(move || {
                mine_range_with_engine(
                    thread_id,
                    job_id,
                    engine.as_ref(),
                    ctx,
                    EngineRange { start, end },
                    cancel_flag,
                    sender,
                    progress_chunk_ms,
                );
            });

            self.thread_handles.push(handle);
        }
    }

    pub fn cancel(&mut self) {
        log::debug!(target: "miner", "Cancelling mining job: {}", self.job_id.as_ref().unwrap());
        self.cancel_flag.store(true, Ordering::Relaxed);
        self.status = JobStatus::Cancelled;
        #[cfg(feature = "metrics")]
        {
            metrics::inc_job_status("cancelled");
            if let Some(job_id) = &self.job_id {
                metrics::inc_jobs_by_engine(self.engine_name, "cancelled");
                metrics::set_job_status_gauge(self.engine_name, job_id, "running", 0);
                metrics::set_job_status_gauge(self.engine_name, job_id, "completed", 0);
                metrics::set_job_status_gauge(self.engine_name, job_id, "failed", 0);
                metrics::set_job_status_gauge(self.engine_name, job_id, "cancelled", 1);
                // Remove job hash rate series on cancellation (avoid scrape-timing artifacts)
                self.last_hash_rate = 0.0;
                metrics::remove_job_hash_rate(self.engine_name, job_id);
                metrics::remove_job_metrics(self.engine_name, job_id);
                metrics::remove_thread_metrics_for_job(self.engine_name, job_id);
                // Remove all per-thread hash rate series on cancellation and clear tracking
                for (tid, _) in self.thread_rate_ema.iter() {
                    metrics::remove_thread_hash_rate(self.engine_name, job_id, &tid.to_string());
                }
                self.thread_last_update.clear();
                self.thread_rate_ema.clear();
            }
        }

        while let Some(handle) = self.thread_handles.pop() {
            let _ = handle.join();
        }
    }

    pub fn update_from_results(&mut self) -> bool {
        let receiver = match &self.result_receiver {
            Some(r) => r,
            None => return false,
        };

        while let Ok(thread_result) = receiver.try_recv() {
            self.total_hash_count += thread_result.hash_count;
            #[cfg(feature = "metrics")]
            {
                metrics::inc_hashes(thread_result.hash_count);
                if let Some(job_id) = &self.job_id {
                    metrics::inc_job_hashes(self.engine_name, job_id, thread_result.hash_count);
                    metrics::inc_thread_hashes(
                        self.engine_name,
                        job_id,
                        &thread_result.thread_id.to_string(),
                        thread_result.hash_count,
                    );
                    // Compute per-thread delta hash rate since the last update for this thread
                    let now = std::time::Instant::now();
                    let last = self
                        .thread_last_update
                        .get(&thread_result.thread_id)
                        .copied()
                        .unwrap_or(self.start_time);
                    let dt = now.duration_since(last).as_secs_f64();
                    if dt > 0.0 && thread_result.hash_count > 0 {
                        let instant_rate = thread_result.hash_count as f64 / dt;
                        // Exponential Moving Average smoothing
                        let prev = self
                            .thread_rate_ema
                            .get(&thread_result.thread_id)
                            .copied()
                            .unwrap_or(instant_rate);
                        let ema = THREAD_RATE_EMA_ALPHA * instant_rate
                            + (1.0 - THREAD_RATE_EMA_ALPHA) * prev;
                        metrics::set_thread_hash_rate(
                            self.engine_name,
                            job_id,
                            &thread_result.thread_id.to_string(),
                            ema,
                        );
                        // Store updated EMA for next delta
                        self.thread_rate_ema.insert(thread_result.thread_id, ema);
                    }
                    // Update last-seen timestamp for this thread
                    self.thread_last_update.insert(thread_result.thread_id, now);

                    // Update per-job hash rate as the sum of per-thread EMAs and publish
                    let job_rate: f64 = self.thread_rate_ema.values().copied().sum();
                    self.last_hash_rate = job_rate;
                    metrics::set_job_hash_rate(self.engine_name, job_id, job_rate);
                }
            }

            if thread_result.completed {
                self.completed_threads += 1;
                #[cfg(feature = "metrics")]
                {
                    if let Some(job_id) = &self.job_id {
                        // Remove this thread's hash rate series on completion
                        metrics::remove_thread_hash_rate(
                            self.engine_name,
                            job_id,
                            &thread_result.thread_id.to_string(),
                        );
                    }
                    // Cleanup per-thread tracking on completion
                    self.thread_last_update.remove(&thread_result.thread_id);
                    self.thread_rate_ema.remove(&thread_result.thread_id);
                }
            }

            if let Some(result) = thread_result.result {
                let is_better = self
                    .best_result
                    .as_ref()
                    .is_none_or(|current_best| result.hash < current_best.hash);

                if is_better {
                    log::debug!(target: "miner",
                        "Found better result from thread {}: distance = {}, nonce = {}",
                        thread_result.thread_id,
                        result.hash,
                        result.nonce
                    );
                    self.best_result = Some(result.clone());
                    self.cancel_flag.store(true, Ordering::Relaxed);
                    // Result is now ready to be fetched via /result
                    log::info!(target: "miner", "Result ready: engine={}, nonce={}, distance={}",
                        self.engine_name, result.nonce, result.hash);
                    #[cfg(feature = "metrics")]
                    {
                        // reuse existing http metric bucket for visibility until dedicated counters exist
                        metrics::inc_mine_requests("result_ready");
                        if let Some(job_id) = &self.job_id {
                            let origin_label = match thread_result.origin {
                                Some(engine_cpu::FoundOrigin::Cpu) => "cpu",
                                Some(engine_cpu::FoundOrigin::GpuG1) => "gpu-g1",
                                Some(engine_cpu::FoundOrigin::GpuG2) => "gpu-g2",
                                _ => "unknown",
                            };
                            metrics::set_job_found_origin(self.engine_name, job_id, origin_label);
                        }
                    }
                }
            }
        }

        if self.status == JobStatus::Running {
            if self.best_result.is_some() {
                self.status = JobStatus::Completed;
                #[cfg(feature = "metrics")]
                {
                    metrics::inc_job_status("completed");
                    if let Some(job_id) = &self.job_id {
                        metrics::inc_jobs_by_engine(self.engine_name, "completed");
                        metrics::inc_candidates_found(self.engine_name, job_id);
                        // TODO(metrics): engine-level false-positive metric is emitted in engine implementations (e.g., gpu-cuda G2 host re-verification)
                        metrics::set_job_status_gauge(self.engine_name, job_id, "running", 0);
                        metrics::set_job_status_gauge(self.engine_name, job_id, "completed", 1);
                        metrics::set_job_status_gauge(self.engine_name, job_id, "failed", 0);
                        metrics::set_job_status_gauge(self.engine_name, job_id, "cancelled", 0);
                        // Remove job hash rate on completion and clear per-thread series
                        self.last_hash_rate = 0.0;
                        metrics::remove_job_hash_rate(self.engine_name, job_id);
                        metrics::remove_job_metrics(self.engine_name, job_id);
                        metrics::remove_thread_metrics_for_job(self.engine_name, job_id);
                        for (tid, _) in self.thread_rate_ema.iter() {
                            metrics::remove_thread_hash_rate(
                                self.engine_name,
                                job_id,
                                &tid.to_string(),
                            );
                        }
                        self.thread_last_update.clear();
                        self.thread_rate_ema.clear();
                    }
                }
            } else if self.completed_threads >= self.thread_handles.len()
                && !self.thread_handles.is_empty()
            {
                self.status = JobStatus::Failed;
                #[cfg(feature = "metrics")]
                {
                    metrics::inc_job_status("failed");
                    if let Some(job_id) = &self.job_id {
                        metrics::inc_jobs_by_engine(self.engine_name, "failed");
                        metrics::set_job_status_gauge(self.engine_name, job_id, "running", 0);
                        metrics::set_job_status_gauge(self.engine_name, job_id, "completed", 0);
                        metrics::set_job_status_gauge(self.engine_name, job_id, "failed", 1);
                        metrics::set_job_status_gauge(self.engine_name, job_id, "cancelled", 0);
                        // Remove job hash rate on failure and clear per-thread series
                        self.last_hash_rate = 0.0;
                        metrics::remove_job_hash_rate(self.engine_name, job_id);
                        for (tid, _) in self.thread_rate_ema.iter() {
                            metrics::remove_thread_hash_rate(
                                self.engine_name,
                                job_id,
                                &tid.to_string(),
                            );
                        }
                        self.thread_last_update.clear();
                        self.thread_rate_ema.clear();
                    }
                }
            }
        }

        self.status != JobStatus::Running
    }
}

#[allow(clippy::too_many_arguments)] // Reason: worker runner needs job_id, engine, context, range, cancel flag, sender, and chunking config
fn mine_range_with_engine(
    thread_id: usize,
    job_id: String,
    engine: &dyn MinerEngine,
    ctx: pow_core::JobContext,
    range: EngineRange,
    cancel_flag: Arc<AtomicBool>,
    sender: Sender<ThreadResult>,
    progress_chunk_ms: u64,
) {
    log::debug!(
        target: "miner",
        "Job {} thread {} mining range {} to {} (inclusive)",
        job_id,
        thread_id,
        range.start,
        range.end
    );

    // Chunk the range into subranges to emit periodic progress updates for metrics
    let mut current_start = range.start;
    let end = range.end;
    // Derive a rough chunk size from target milliseconds and an estimate of hashes/sec.
    // Start with a conservative default of 100k ops/sec per thread and scale by target ms.
    let target_ms = progress_chunk_ms;
    let est_ops_per_sec: u64 = 100_000;
    let derived_chunk: u64 = ((est_ops_per_sec.saturating_mul(target_ms)) / 1000).max(5_000);
    let chunk_size = U512::from(derived_chunk);
    let mut done = false;

    while current_start <= end && !cancel_flag.load(Ordering::Relaxed) {
        let mut current_end = current_start
            .saturating_add(chunk_size)
            .saturating_sub(U512::from(1u64));
        if current_end > end {
            current_end = end;
        }

        let sub_range = EngineRange {
            start: current_start,
            end: current_end,
        };

        log::debug!(
            target: "miner",
            "Job {} thread {} processing subrange {}..{} (inclusive)",
            job_id,
            thread_id,
            sub_range.start,
            sub_range.end
        );
        let status = engine.search_range(&ctx, sub_range.clone(), &cancel_flag);

        match status {
            engine_cpu::EngineStatus::Found {
                candidate: EngineCandidate { nonce, work, hash },
                hash_count,
                origin,
            } => {
                // Send final result with found candidate and the hashes covered in this subrange
                let final_result = ThreadResult {
                    thread_id,
                    result: Some(MiningJobResult { nonce, work, hash }),
                    hash_count,
                    origin: Some(origin),
                    completed: true,
                };
                if sender.try_send(final_result).is_err() {
                    log::warn!(target: "miner", "Job {job_id} thread {thread_id} failed to send final result");
                }
                done = true;
                break;
            }
            engine_cpu::EngineStatus::Exhausted { hash_count } => {
                // Send intermediate progress update for this chunk
                let update = ThreadResult {
                    thread_id,
                    result: None,
                    hash_count,
                    origin: None,
                    completed: false,
                };
                if sender.try_send(update).is_err() {
                    log::warn!(target: "miner", "Job {job_id} thread {thread_id} failed to send progress update");
                    break;
                } else {
                    log::debug!(
                        target: "miner",
                        "Job {} thread {} progress: hashed={} in subrange {}..{}",
                        job_id,
                        thread_id,
                        hash_count,
                        sub_range.start,
                        sub_range.end
                    );
                }
            }
            engine_cpu::EngineStatus::Cancelled { hash_count } => {
                // Send last progress update and stop
                let update = ThreadResult {
                    thread_id,
                    result: None,
                    hash_count,
                    origin: None,
                    completed: false,
                };
                if sender.try_send(update).is_err() {
                    log::warn!(target: "miner", "Job {job_id} thread {thread_id} failed to send cancel update");
                }
                done = true;
                break;
            }
            engine_cpu::EngineStatus::Running { .. } => {
                // Not expected from a synchronous engine chunk call; ignore
            }
        }

        if current_end == end {
            break;
        }
        current_start = current_end.saturating_add(U512::from(1u64));
    }

    if !done {
        // Signal thread completion with no final candidate
        let final_result = ThreadResult {
            thread_id,
            result: None,
            hash_count: 0,
            origin: None,
            completed: true,
        };
        if sender.try_send(final_result).is_err() {
            log::warn!(target: "miner", "Job {job_id} thread {thread_id} failed to send completion status after chunked search");
        }
    }

    log::debug!(target: "miner", "Job {job_id} thread {thread_id} completed.");
}

/// Validates incoming mining requests for structural correctness.
pub fn validate_mining_request(request: &MiningRequest) -> Result<(), String> {
    if request.job_id.is_empty() {
        return Err("job_id cannot be empty".to_string());
    }

    if request.mining_hash.len() != 64 {
        return Err("mining_hash must be 64 hex characters".to_string());
    }
    if hex::decode(&request.mining_hash).is_err() {
        return Err("mining_hash must be valid hex".to_string());
    }

    if U512::from_dec_str(&request.distance_threshold).is_err() {
        return Err("distance_threshold must be a valid decimal number".to_string());
    }

    if request.nonce_start.len() != 128 {
        return Err("nonce_start must be 128 hex characters".to_string());
    }
    if request.nonce_end.len() != 128 {
        return Err("nonce_end must be 128 hex characters".to_string());
    }

    let nonce_start = U512::from_str_radix(&request.nonce_start, 16)
        .map_err(|_| "nonce_start must be valid hex".to_string())?;
    let nonce_end = U512::from_str_radix(&request.nonce_end, 16)
        .map_err(|_| "nonce_end must be valid hex".to_string())?;

    if nonce_start > nonce_end {
        return Err("nonce_start must be <= nonce_end".to_string());
    }

    Ok(())
}

/// HTTP handler for POST /mine
pub async fn handle_mine_request(
    request: MiningRequest,
    state: MiningService,
) -> Result<impl Reply, Rejection> {
    log::debug!("Received mine request: {request:?}");
    if let Err(e) = validate_mining_request(&request) {
        log::warn!("Invalid mine request ({}): {}", request.job_id, e);
        #[cfg(feature = "metrics")]
        {
            metrics::inc_mine_requests("invalid");
        }
        return Ok(warp::reply::with_status(
            warp::reply::json(&MiningResponse {
                status: ApiResponseStatus::Error,
                job_id: request.job_id,
                message: Some(e),
            }),
            warp::http::StatusCode::BAD_REQUEST,
        ));
    }

    let header_hash: [u8; 32] = hex::decode(&request.mining_hash)
        .unwrap()
        .try_into()
        .expect("Validated hex string is 32 bytes");
    let difficulty = U512::from_dec_str(&request.distance_threshold).unwrap();
    let nonce_start = U512::from_str_radix(&request.nonce_start, 16).unwrap();
    let nonce_end = U512::from_str_radix(&request.nonce_end, 16).unwrap();

    let job = MiningJob::new(header_hash, difficulty, nonce_start, nonce_end);

    match state.add_job(request.job_id.clone(), job).await {
        Ok(_) => {
            log::debug!(target: "miner", "Accepted mine request for job ID: {}", request.job_id);
            #[cfg(feature = "metrics")]
            {
                metrics::inc_mine_requests("accepted");
            }
            Ok(warp::reply::with_status(
                warp::reply::json(&MiningResponse {
                    status: ApiResponseStatus::Accepted,
                    job_id: request.job_id,
                    message: None,
                }),
                warp::http::StatusCode::OK,
            ))
        }
        Err(e) => {
            log::error!("Failed to add job {}: {}", request.job_id, e);
            #[cfg(feature = "metrics")]
            {
                let result = if e.contains("already") {
                    "duplicate"
                } else {
                    "error"
                };
                metrics::inc_mine_requests(result);
            }
            Ok(warp::reply::with_status(
                warp::reply::json(&MiningResponse {
                    status: ApiResponseStatus::Error,
                    job_id: request.job_id,
                    message: Some(e),
                }),
                warp::http::StatusCode::CONFLICT,
            ))
        }
    }
}

/// HTTP handler for GET /result/{job_id}
pub async fn handle_result_request(
    job_id: String,
    state: MiningService,
) -> Result<impl Reply, Rejection> {
    log::debug!("Received result request for job: {job_id}");

    let job = match state.get_job(&job_id).await {
        Some(job) => job,
        None => {
            log::warn!("Result request for unknown job: {job_id}");
            return Ok(warp::reply::with_status(
                warp::reply::json(&quantus_miner_api::MiningResult {
                    status: ApiResponseStatus::NotFound,
                    job_id,
                    nonce: None,
                    work: None,
                    hash_count: 0,
                    elapsed_time: 0.0,
                }),
                warp::http::StatusCode::NOT_FOUND,
            ));
        }
    };

    log::debug!(
        target: "miner",
        "Result polling watchdog: job_id={}, status={:?}, result_served={}, elapsed_s={:.3}",
        job_id,
        job.status,
        job.result_served,
        job.start_time.elapsed().as_secs_f64()
    );
    let status = match job.status {
        JobStatus::Running => ApiResponseStatus::Running,
        JobStatus::Completed => ApiResponseStatus::Completed,
        JobStatus::Failed => ApiResponseStatus::Failed,
        JobStatus::Cancelled => ApiResponseStatus::Cancelled,
    };

    let (nonce_hex, work_hex) = match &job.best_result {
        Some(result) => (
            Some(format!("{:x}", result.nonce)),
            Some(hex::encode(result.work)),
        ),
        None => (None, None),
    };

    // Inline re-verify using the exact nonce bytes we will return
    if let Some(result) = &job.best_result {
        let nonce_be = result.nonce.to_big_endian();
        let (ok, hash_result) = pow_core::is_valid_nonce(job.header_hash, nonce_be, job.difficulty);
        log::info!(
            target: "miner",
            "Serving result: job_id={}, engine={}, ok={}, hash={}, difficulty={}",
            job_id,
            job.engine_name,
            ok,
            hash_result,
            job.difficulty
        );
        #[cfg(feature = "metrics")]
        {
            metrics::inc_mine_requests("result_served");
        }
        // Mark served to keep job until at least one fetch succeeds
        state.mark_job_result_served(&job_id).await;
    }

    let elapsed_time = job.start_time.elapsed().as_secs_f64();

    Ok(warp::reply::with_status(
        warp::reply::json(&quantus_miner_api::MiningResult {
            status,
            job_id,
            nonce: nonce_hex,
            work: work_hex,
            hash_count: job.total_hash_count,
            elapsed_time,
        }),
        warp::http::StatusCode::OK,
    ))
}

/// HTTP handler for POST /cancel/{job_id}
pub async fn handle_cancel_request(
    job_id: String,
    state: MiningService,
) -> Result<impl Reply, Rejection> {
    log::debug!("Received cancel request for job: {job_id}");

    if state.cancel_job(&job_id).await {
        log::debug!(target: "miner", "Successfully cancelled job: {job_id}");
        Ok(warp::reply::with_status(
            warp::reply::json(&MiningResponse {
                status: ApiResponseStatus::Cancelled,
                job_id,
                message: None,
            }),
            warp::http::StatusCode::OK,
        ))
    } else {
        log::warn!("Cancel request for unknown job: {job_id}");
        Ok(warp::reply::with_status(
            warp::reply::json(&MiningResponse {
                status: ApiResponseStatus::NotFound,
                job_id,
                message: Some("Job not found".to_string()),
            }),
            warp::http::StatusCode::NOT_FOUND,
        ))
    }
}

/// Build the warp routes for the miner API using the provided service state.
pub fn build_routes(
    state: MiningService,
) -> impl Filter<Extract = (impl Reply,), Error = Rejection> + Clone {
    let state_clone = state.clone();
    let state_filter = warp::any().map(move || state_clone.clone());

    let mine_route = warp::post()
        .and(warp::path("mine"))
        .and(warp::body::json())
        .and(state_filter.clone())
        .and_then(handle_mine_request);

    let result_route = warp::get()
        .and(warp::path("result"))
        .and(warp::path::param())
        .and(state_filter.clone())
        .and_then(handle_result_request);

    let cancel_route = warp::post()
        .and(warp::path("cancel"))
        .and(warp::path::param())
        .and(state_filter.clone())
        .and_then(handle_cancel_request);

    mine_route.or(result_route).or(cancel_route)
}

/// Helper structures and functions for safe range partitioning (placed before use)
#[derive(Debug, Clone)]
struct Partitions {
    total_range: U512,
    ranges: Vec<(U512, U512)>,
}

/// Compute safe, inclusive partitions of [start, end] into `workers` slices.
/// Guarantees coverage without overflow and clamps to `end`.
fn compute_partitions(start: U512, end: U512, workers: usize) -> Partitions {
    // total inclusive range
    let total_range = end.saturating_sub(start).saturating_add(U512::from(1u64));

    let workers = workers.max(1);
    let divisor = U512::from(workers as u64).max(U512::from(1u64));
    let range_per = total_range / divisor;
    let remainder = total_range % divisor;

    let mut ranges = Vec::with_capacity(workers);
    for i in 0..workers {
        let idx = U512::from(i as u64);
        let s = start.saturating_add(range_per.saturating_mul(idx));
        let mut e = s.saturating_add(range_per).saturating_sub(U512::from(1u64));
        if i == workers - 1 {
            e = e.saturating_add(remainder);
        }
        if e > end {
            e = end;
        }
        ranges.push((s, e));
    }

    Partitions {
        total_range,
        ranges,
    }
}

/// Start the miner service with the given configuration.
/// - Spawns the mining loop.
/// - Optionally exposes a metrics endpoint if `metrics_port` is provided and the `metrics` feature is enabled.
/// - Serves the HTTP API on `config.port`.
pub async fn run(config: ServiceConfig) -> anyhow::Result<()> {
    // Determine available logical CPUs and the cpuset mask (if any), preferring cgroup v2.
    fn detect_effective_cpus_and_mask() -> (usize, Option<String>) {
        // Try cgroup v2 effective cpuset
        if let Ok(mask) = std::fs::read_to_string("/sys/fs/cgroup/cpuset.cpus.effective") {
            let trimmed = mask.trim();
            if let Some(count) = parse_cpuset_to_count(trimmed) {
                return (count.max(1), Some(trimmed.to_string()));
            }
        }
        // Fallback to legacy cgroup v1 path
        if let Ok(mask) = std::fs::read_to_string("/sys/fs/cgroup/cpuset/cpuset.cpus") {
            let trimmed = mask.trim();
            if let Some(count) = parse_cpuset_to_count(trimmed) {
                return (count.max(1), Some(trimmed.to_string()));
            }
        }
        // Fallback to all logical CPUs
        (num_cpus::get().max(1), None)
    }

    // Parse cpuset list/ranges like "0-3,6,8-11" into a count
    fn parse_cpuset_to_count(s: &str) -> Option<usize> {
        if s.is_empty() {
            return None;
        }
        let mut count: usize = 0;
        for part in s.split(',') {
            let p = part.trim();
            if p.is_empty() {
                continue;
            }
            if let Some((a, b)) = p.split_once('-') {
                let start = a.trim().parse::<usize>().ok()?;
                let end = b.trim().parse::<usize>().ok()?;
                if end < start {
                    return None;
                }
                count += end - start + 1;
            } else {
                // single cpu id
                let _ = p.parse::<usize>().ok()?;
                count += 1;
            }
        }
        Some(count)
    }
    // keep run(config) open; do not close here

    // Detect effective CPU pool for this process (cpuset if available).
    let (effective_cpus, cpuset_mask) = detect_effective_cpus_and_mask();
    if let Some(mask) = cpuset_mask.as_ref() {
        log::debug!(target: "miner", "Detected cpuset mask: {mask}");
    } else {
        log::debug!(target: "miner", "No cpuset mask detected; using full logical CPU count");
    }
    #[cfg(feature = "metrics")]
    {
        // Expose effective CPUs as a gauge for dashboards/alerts.
        metrics::set_effective_cpus(effective_cpus as i64);
    }

    // Default workers: leave at least half of resources for other processes.
    // Use max(1, effective_cpus / 2), but also cap at effective_cpus - 1 if possible.
    let default_workers = effective_cpus
        .saturating_sub(effective_cpus / 2)
        .min(effective_cpus.saturating_sub(1))
        .max(1);

    // Resolve workers from user config, clamped to [1, effective_cpus].
    let mut workers = match config.workers {
        Some(n) if n > 0 => {
            if n > effective_cpus {
                log::warn!(
                    "Requested {n} workers exceeds available logical CPUs in cpuset ({effective_cpus}). Clamping to {effective_cpus}."
                );
                effective_cpus
            } else {
                n
            }
        }
        Some(_) => {
            log::warn!("Workers must be positive. Falling back to default.");
            default_workers
        }
        None => {
            log::info!(
                "No --workers specified. Defaulting to {} (leaving ~{} for other processes) based on effective {} logical CPUs.",
                default_workers,
                effective_cpus.saturating_sub(default_workers),
                effective_cpus
            );
            default_workers
        }
    };

    // Final safety: never exceed effective_cpus.
    if workers > effective_cpus {
        workers = effective_cpus.max(1);
    }
    
    // Force workers=1 for GPU engine to prevent concurrent buffer mapping panics
    if matches!(config.engine, EngineSelection::Gpu) {
        if workers > 1 {
            log::info!(
                "GPU engine selected. Forcing workers to 1 (was {}) to avoid buffer contention.",
                workers
            );
            workers = 1;
        }
    }


    log::info!(
        "Using {workers} worker thread(s) for mining (effective logical CPUs available: {effective_cpus})"
    );

    // Select engine
    #[allow(unused_mut)]
    let mut engine: Arc<dyn MinerEngine> = match config.engine {
        EngineSelection::CpuBaseline => Arc::new(engine_cpu::BaselineCpuEngine::new()),
        EngineSelection::CpuFast => Arc::new(engine_cpu::FastCpuEngine::new()),
        EngineSelection::CpuChainManipulator => {
            let mut eng = engine_cpu::ChainEngine::new();
            // Apply optional throttle parameters if provided.
            if let Some(base) = config.manip_base_delay_ns {
                log::debug!(target: "miner", "Manipulator base_delay_ns overridden via config: {base} ns");
                eng.base_delay_ns = base;
            }
            if let Some(step) = config.manip_step_batch {
                log::debug!(target: "miner", "Manipulator step_batch overridden via config: {step}");
                eng.step_batch = step;
            }
            if let Some(cap) = config.manip_throttle_cap {
                log::debug!(target: "miner", "Manipulator throttle_cap set via config: {cap}");
                eng.throttle_cap = Some(cap);
            }
            // If a starting throttle index is provided, set it here for "pick up where we left off".
            if let Some(n) = config.manip_solved_blocks {
                log::debug!(target: "miner", "Manipulator starting throttle index (solved_blocks) set via config: {n}");
                eng.job_index.store(n, std::sync::atomic::Ordering::Relaxed);
            }
            Arc::new(eng)
        }
        EngineSelection::Gpu => {
            #[cfg(feature = "gpu")]
            {
                Arc::new(engine_gpu::GpuEngine::new())
            }
            #[cfg(not(feature = "gpu"))]
            {
                log::error!("Requested engine gpu, but this binary was built without the 'gpu' feature. Rebuild miner-service with --features gpu.");
                return Err(anyhow::anyhow!(
                    "engine 'gpu' not built (missing 'gpu' feature)"
                ));
            }
        }
    };
    log::info!("Using engine: {}", engine.name());
    log::info!("Service configuration: {config}");

    let progress_chunk_ms = config.progress_chunk_ms.unwrap_or(2000);
    let service = MiningService::new(workers, engine.clone(), progress_chunk_ms);

    // Start mining loop
    service.start_mining_loop().await;

    // Telemetry bootstrap from environment variables (optional)
    let telemetry_handle_opt = {
        let endpoints: Vec<String> = std::env::var("MINER_TELEMETRY_ENDPOINTS")
            .ok()
            .map(|s| {
                s.split(',')
                    .map(|p| p.trim().to_string())
                    .filter(|p| !p.is_empty())
                    .collect()
            })
            .unwrap_or_default();

        let enabled = std::env::var("MINER_TELEMETRY_ENABLED")
            .ok()
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(!endpoints.is_empty());

        let verbosity = std::env::var("MINER_TELEMETRY_VERBOSITY")
            .ok()
            .and_then(|v| v.parse::<u8>().ok())
            .unwrap_or(0);

        if enabled && !endpoints.is_empty() {
            let interval_secs = std::env::var("MINER_TELEMETRY_INTERVAL_SECS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok());
            let chain = std::env::var("MINER_TELEMETRY_CHAIN").ok();
            let genesis = std::env::var("MINER_TELEMETRY_GENESIS").ok();

            // Optional linked node info
            let link = miner_telemetry::TelemetryNodeLink {
                node_telemetry_id: std::env::var("MINER_TELEMETRY_NODE_ID").ok(),
                node_peer_id: std::env::var("MINER_TELEMETRY_NODE_PEER_ID").ok(),
                node_name: std::env::var("MINER_TELEMETRY_NODE_NAME").ok(),
                node_version: std::env::var("MINER_TELEMETRY_NODE_VERSION").ok(),
                chain: chain.clone(),
                genesis_hash: genesis.clone(),
            };
            let default_link = if link.node_telemetry_id.is_some()
                || link.node_peer_id.is_some()
                || link.node_name.is_some()
                || link.node_version.is_some()
                || link.chain.is_some()
                || link.genesis_hash.is_some()
            {
                Some(link)
            } else {
                None
            };

            let cfg = miner_telemetry::TelemetryConfig {
                enabled,
                endpoints,
                verbosity,
                name: Some("quantus-miner".to_string()),
                implementation: Some("quantus-miner".to_string()),
                version: Some(
                    option_env!("MINER_VERSION")
                        .unwrap_or(env!("CARGO_PKG_VERSION"))
                        .to_string(),
                ),
                chain,
                genesis_hash: genesis,
                interval_secs,
                default_link,
            };
            Some(miner_telemetry::start(cfg))
        } else {
            None
        }
    };

    if let Some(telemetry) = telemetry_handle_opt {
        log::info!("Telemetry enabled; session_id={}", telemetry.session_id());
        telemetry.emit_system_connected(None).await;

        let telemetry_handle = telemetry.clone();
        let svc = service.clone();
        let engine_name_str = engine.name().to_string();
        let interval_secs = std::env::var("MINER_TELEMETRY_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(15);
        let start_instant = std::time::Instant::now();

        tokio::spawn(async move {
            loop {
                let (active_jobs, total_rate) = {
                    let jobs = svc.jobs.lock().await;
                    let mut running = 0i64;
                    let mut rate = 0.0;
                    for (_id, job) in jobs.iter() {
                        if job.status == JobStatus::Running {
                            running += 1;
                            rate += job.last_hash_rate;
                        }
                    }
                    (running, rate)
                };

                let uptime_ms = start_instant.elapsed().as_millis() as u64;

                let interval = miner_telemetry::SystemInterval {
                    uptime_ms,
                    engine: Some(engine_name_str.clone()),
                    workers: Some(svc.workers as u32),
                    hash_rate: Some(total_rate),
                    active_jobs: Some(active_jobs),
                    linked_node_hint: None,
                };

                telemetry_handle.emit_system_interval(&interval, None).await;

                tokio::time::sleep(tokio::time::Duration::from_secs(interval_secs)).await;
            }
        });
    } else {
        log::info!("Telemetry disabled (no endpoints configured)");
    }

    // Optionally start metrics exporter if enabled via CLI and feature flag.
    if let Some(port) = config.metrics_port {
        #[cfg(feature = "metrics")]
        {
            log::info!("Starting metrics endpoint on 0.0.0.0:{port}");
            metrics::start_http_exporter(port).await?;
        }
        #[cfg(not(feature = "metrics"))]
        {
            log::warn!(
                "Metrics port provided ({port}), but 'metrics' feature is not enabled. Skipping."
            );
        }
    } else {
        log::info!("Metrics disabled (no --metrics-port provided)");
    }

    // Build routes
    let routes = build_routes(service);

    // Start server
    let addr = ([0, 0, 0, 0], config.port);
    let socket = std::net::SocketAddr::from(addr);
    log::info!("Server starting on {socket}");
    warp::serve(routes).run(socket).await;

    Ok(())
}

#[cfg(test)]
mod tests {

    use super::{JobStatus, MiningJob, MiningService};
    use engine_cpu::MinerEngine;
    use primitive_types::U512;
    use std::sync::Arc;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_mining_state_add_get_remove() {
        let engine: Arc<dyn MinerEngine> = Arc::new(engine_cpu::BaselineCpuEngine::new());
        let state = MiningService::new(2, engine, 2000);

        let job = MiningJob::new(
            [1u8; 32],
            U512::from(1000000u64),
            U512::zero(),
            U512::from(1000u64),
        );
        assert!(state.add_job("test".to_string(), job).await.is_ok());
        assert!(state.get_job("test").await.is_some());
        assert!(state.remove_job("test").await.is_some());
        assert!(state.get_job("test").await.is_none());
    }

    #[tokio::test]
    async fn test_job_lifecycle_fail() {
        // Test that a job fails if no nonce is found (threshold too strict).
        // To make this deterministic, avoid nonce=0, which some math paths treat as special.
        let engine: Arc<dyn MinerEngine> = Arc::new(engine_cpu::BaselineCpuEngine::new());
        let state = MiningService::new(2, engine, 2000);
        state.start_mining_loop().await;

        // Impossible difficulty with a nonce range that excludes 0
        let header_hash = [1u8; 32];
        let difficulty = U512::MAX;
        let nonce_start = U512::from(1);
        let nonce_end = U512::from(100);

        let job = MiningJob::new(header_hash, difficulty, nonce_start, nonce_end);
        state.add_job("fail_job".to_string(), job).await.unwrap();

        let mut finished_job = None;
        for _ in 0..50 {
            // Poll for 5 seconds max (50 * 100ms)
            let job_status = state.get_job("fail_job").await.unwrap();
            if job_status.status != JobStatus::Running {
                finished_job = Some(job_status);
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }

        let finished_job = finished_job.expect("Job did not finish in time");
        assert_eq!(finished_job.status, JobStatus::Failed);
    }

    #[cfg(test)]
    mod partition_tests {
        use crate::compute_partitions;
        use primitive_types::U512;

        fn u(x: u64) -> U512 {
            U512::from(x)
        }

        #[test]
        fn partitions_cover_entire_range_even_workers() {
            // Range [0, 99] split into 4 workers
            let p = compute_partitions(u(0), u(99), 4);
            assert_eq!(p.total_range, u(100));
            // Check coverage and ordering
            let mut covered = 0u64;
            let mut prev_end = u(0);
            for (i, (s, e)) in p.ranges.iter().enumerate() {
                if i == 0 {
                    assert_eq!(*s, u(0));
                } else {
                    assert_eq!(*s, prev_end.saturating_add(u(1)));
                }
                assert!(e >= s);
                covered += (e.saturating_sub(*s).saturating_add(u(1))).as_u64();
                prev_end = *e;
            }
            assert_eq!(covered, 100);
            assert_eq!(prev_end, u(99));
        }

        #[test]
        fn partitions_cover_entire_range_odd_workers() {
            // Range [0, 99] split into 3 workers
            let p = compute_partitions(u(0), u(99), 3);
            assert_eq!(p.total_range, u(100));
            // Ensure last range takes remainder and we end exactly at 99
            assert_eq!(p.ranges.len(), 3);
            assert_eq!(p.ranges[0], (u(0), u(32)));
            assert_eq!(p.ranges[1], (u(33), u(65)));
            assert_eq!(p.ranges[2], (u(66), u(99)));
        }

        #[test]
        fn partitions_handles_start_eq_end() {
            // Range [42, 42] split into any workers -> only last partition reaches the end
            let p = compute_partitions(u(42), u(42), 5);
            assert_eq!(p.total_range, u(1));
            assert_eq!(p.ranges.len(), 5);
            // All but the last partition may have end < start (empty), but no overflow, and last ends at 42
            assert_eq!(p.ranges[4].1, u(42));
        }

        #[test]
        fn partitions_huge_values_no_overflow() {
            // Use near-max U512 values to ensure no overflow
            let start = U512::from(0);
            let end = U512::MAX.saturating_sub(u(10_000)); // keep a finite total_range
            let p = compute_partitions(start, end, 3);
            // Check that each end <= original end
            for (s, e) in p.ranges {
                assert!(e >= s);
                assert!(e <= end);
            }
        }
    }

    #[tokio::test]
    async fn test_job_lifecycle_success() {
        // Test that a job completes successfully
        let engine: Arc<dyn MinerEngine> = Arc::new(engine_cpu::BaselineCpuEngine::new());
        let state = MiningService::new(2, engine, 2000);
        state.start_mining_loop().await;

        // Easy difficulty
        let header_hash = [1u8; 32];
        let difficulty = U512::from(1u64); // Easiest difficulty
        let nonce_start = U512::from(0);
        let nonce_end = U512::from(10000);

        let job = MiningJob::new(header_hash, difficulty, nonce_start, nonce_end);
        state.add_job("success_job".to_string(), job).await.unwrap();

        let mut finished_job = None;
        for _ in 0..50 {
            // Poll for 5 seconds max
            let job_status = state.get_job("success_job").await.unwrap();
            if job_status.status != JobStatus::Running {
                finished_job = Some(job_status);
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }

        let finished_job = finished_job.expect("Job did not finish in time");
        assert_eq!(finished_job.status, JobStatus::Completed);
        assert!(finished_job.best_result.is_some());
    }

    #[test]
    fn validate_mining_request_rejects_bad_inputs() {
        // Helper to build a baseline-valid request we can mutate per case
        fn valid_req() -> quantus_miner_api::MiningRequest {
            quantus_miner_api::MiningRequest {
                job_id: "job-1".to_string(),
                // 64 hex chars (32 bytes)
                mining_hash: "11".repeat(32),
                distance_threshold: "1".to_string(),
                // 128 hex chars (64 bytes)
                nonce_start: "00".repeat(64),
                nonce_end: format!("{:0128x}", 1u8),
            }
        }

        // 1) Empty job_id
        {
            let mut r = valid_req();
            r.job_id = "".to_string();
            let e = super::validate_mining_request(&r).unwrap_err();
            assert!(e.contains("job_id cannot be empty"));
        }

        // 2) Bad mining_hash length
        {
            let mut r = valid_req();
            r.mining_hash = "aa".repeat(31); // 62 chars
            let e = super::validate_mining_request(&r).unwrap_err();
            assert!(e.contains("mining_hash must be 64 hex characters"));
        }

        // 3) Bad mining_hash hex
        {
            let mut r = valid_req();
            r.mining_hash = "zz".repeat(32);
            let e = super::validate_mining_request(&r).unwrap_err();
            assert!(e.contains("mining_hash must be valid hex"));
        }

        // 4) Bad difficulty decimal
        {
            let mut r = valid_req();
            r.distance_threshold = "not-a-decimal".to_string();
            let e = super::validate_mining_request(&r).unwrap_err();
            assert!(e.contains("distance_threshold must be a valid decimal number"));
        }

        // 5) Bad nonce_start length
        {
            let mut r = valid_req();
            r.nonce_start = "00".repeat(63); // 126 chars
            let e = super::validate_mining_request(&r).unwrap_err();
            assert!(e.contains("nonce_start must be 128 hex characters"));
        }

        // 6) Bad nonce_end length
        {
            let mut r = valid_req();
            r.nonce_end = "00".repeat(63); // 126 chars
            let e = super::validate_mining_request(&r).unwrap_err();
            assert!(e.contains("nonce_end must be 128 hex characters"));
        }

        // 7) Bad nonce_start hex
        {
            let mut r = valid_req();
            r.nonce_start = "0g".repeat(64);
            let e = super::validate_mining_request(&r).unwrap_err();
            assert!(e.contains("nonce_start must be valid hex"));
        }

        // 8) Bad nonce_end hex
        {
            let mut r = valid_req();
            r.nonce_end = "0g".repeat(64);
            let e = super::validate_mining_request(&r).unwrap_err();
            assert!(e.contains("nonce_end must be valid hex"));
        }

        // 9) nonce_start > nonce_end
        {
            let mut r = valid_req();
            // start = 2, end = 1
            r.nonce_start = format!("{:0128x}", 2u8);
            r.nonce_end = format!("{:0128x}", 1u8);
            let e = super::validate_mining_request(&r).unwrap_err();
            assert!(e.contains("nonce_start must be <= nonce_end"));
        }
    }

    #[tokio::test]
    async fn http_endpoints_handle_basic_flows() {
        use warp::test::request;

        let engine: Arc<dyn engine_cpu::MinerEngine> =
            Arc::new(engine_cpu::BaselineCpuEngine::new());
        let service = super::MiningService::new(2, engine, 2000);
        service.start_mining_loop().await;

        // Build routes
        let routes = super::build_routes(service.clone());

        // 1) GET /result for unknown job -> 404
        let res = request()
            .method("GET")
            .path("/result/unknown")
            .reply(&routes)
            .await;
        assert_eq!(res.status(), warp::http::StatusCode::NOT_FOUND);

        // 2) POST /cancel for unknown job -> 404
        let res = request()
            .method("POST")
            .path("/cancel/unknown")
            .reply(&routes)
            .await;
        assert_eq!(res.status(), warp::http::StatusCode::NOT_FOUND);

        // 3) POST /mine valid -> 200 Accepted, duplicate -> 409
        let req = quantus_miner_api::MiningRequest {
            job_id: "job-http-1".to_string(),
            mining_hash: "11".repeat(32), // 64 hex chars
            distance_threshold: "99999999999999".to_string(), // hard, likely to fail later; OK for accept flow
            nonce_start: "00".repeat(64),                     // 128 hex chars
            nonce_end: format!("{:0128x}", 1u8),
        };

        let res = request()
            .method("POST")
            .path("/mine")
            .json(&req)
            .reply(&routes)
            .await;
        assert_eq!(res.status(), warp::http::StatusCode::OK);

        let res_dup = request()
            .method("POST")
            .path("/mine")
            .json(&req)
            .reply(&routes)
            .await;
        assert_eq!(res_dup.status(), warp::http::StatusCode::CONFLICT);
    }

    #[test]
    fn chunked_mining_sends_progress_and_completion() {
        use crossbeam_channel::bounded;
        use std::sync::atomic::AtomicBool;

        // Baseline engine, hard difficulty to force Exhausted path for the sub-range
        let engine = engine_cpu::BaselineCpuEngine::new();
        let header = [3u8; 32];
        let difficulty = U512::MAX;
        let ctx = engine.prepare_context(header, difficulty);

        // Small range; chunking derives a large chunk size, so it will be a single chunk,
        // which still exercises the Exhausted -> progress update and final completion paths.
        let range = super::EngineRange {
            start: U512::from(1u64),
            end: U512::from(10_000u64),
        };

        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = bounded::<super::ThreadResult>(8);

        super::mine_range_with_engine(
            0,
            "test-job".to_string(),
            &engine,
            ctx,
            range,
            cancel,
            tx,
            10, // ms (min chunk size is 5k; fine for testing progress + completion)
        );

        // Expect at least two messages: one progress (completed=false) and one final completion (completed=true)
        let first = rx.recv().expect("expected first progress update");
        assert!(
            !first.completed,
            "first message should be a progress update"
        );
        assert!(
            first.hash_count > 0,
            "progress update should report non-zero hash_count"
        );

        // Drain messages until we see the completion marker.
        let mut final_msg = first;
        while !final_msg.completed {
            final_msg = rx.recv().expect("expected next message");
        }
        assert!(
            final_msg.completed,
            "final message should indicate thread completion"
        );
        assert_eq!(
            final_msg.hash_count, 0,
            "final completion message carries zero hash_count"
        );
    }
}

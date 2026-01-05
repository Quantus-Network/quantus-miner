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

/// Service runtime configuration provided by the CLI/binary.
#[derive(Clone, Debug)]
pub struct ServiceConfig {
    /// Port for the HTTP miner API.
    pub port: u16,
    /// Number of CPU worker threads to use for mining (None = auto-detect)
    pub cpu_workers: Option<usize>,
    /// Number of GPU devices to use for mining (None = auto-detect)
    pub gpu_devices: Option<usize>,
    /// Optional metrics port. When Some, metrics endpoint starts; when None, metrics are disabled.
    pub metrics_port: Option<u16>,
    /// How often to report mining progress (in milliseconds). If None, defaults to 10000ms.
    pub progress_interval_ms: Option<u64>,
    /// Size of work chunks to process before reporting progress (in number of hashes). If None, uses engine-specific defaults.
    pub chunk_size: Option<u64>,
    /// Optional starting value for the manipulator engine's solved-blocks throttle index.
    pub manip_solved_blocks: Option<u64>,
    /// Optional base sleep per batch in nanoseconds for manipulator engine (default 500_000ns).
    pub manip_base_delay_ns: Option<u64>,
    /// Optional number of nonce attempts between sleeps for manipulator engine (default 10_000).
    pub manip_step_batch: Option<u64>,
    /// Optional cap on solved-blocks throttle index for manipulator engine.
    pub manip_throttle_cap: Option<u64>,
    /// Optional target duration for GPU batches in milliseconds.
    pub gpu_batch_duration_ms: Option<u64>,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            port: 9833,
            cpu_workers: None,
            gpu_devices: None,
            metrics_port: None,
            progress_interval_ms: None,
            chunk_size: None,
            manip_solved_blocks: None,
            manip_base_delay_ns: None,
            manip_step_batch: None,
            manip_throttle_cap: None,
            gpu_batch_duration_ms: None,
        }
    }
}

impl fmt::Display for ServiceConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "port={}, cpu_workers={:?}, gpu_devices={:?}, metrics_port={:?}, progress_interval_ms={:?}, chunk_size={:?}, manip_solved_blocks={:?}, manip_base_delay_ns={:?}, manip_step_batch={:?}, manip_throttle_cap={:?}, gpu_batch_duration_ms={:?}",
            self.port,
            self.cpu_workers,
            self.gpu_devices,
            self.metrics_port,
            self.progress_interval_ms,
            self.chunk_size,
            self.manip_solved_blocks,
            self.manip_base_delay_ns,
            self.manip_step_batch,
            self.manip_throttle_cap,
            self.gpu_batch_duration_ms
        )
    }
}

/// The core service state: job registry, CPU/GPU engines, and thread configuration.
#[derive(Clone)]
pub struct MiningService {
    pub jobs: Arc<Mutex<HashMap<String, MiningJob>>>,
    pub cpu_workers: usize,
    pub gpu_devices: usize,
    pub cpu_engine: Option<Arc<dyn MinerEngine>>,
    pub gpu_engine: Option<Arc<dyn MinerEngine>>,
    /// How often to report mining progress (in milliseconds).
    pub progress_interval_ms: u64,
    /// Work chunk size (number of hashes to process before progress update).
    pub chunk_size: Option<u64>,
    /// Gauge of currently running jobs (for metrics)
    pub active_jobs_gauge: Arc<tokio::sync::Mutex<i64>>,
}

impl MiningService {
    fn new(
        cpu_workers: usize,
        gpu_devices: usize,
        cpu_engine: Option<Arc<dyn MinerEngine>>,
        gpu_engine: Option<Arc<dyn MinerEngine>>,
        progress_interval_ms: u64,
        chunk_size: Option<u64>,
    ) -> Self {
        Self {
            jobs: Arc::new(Mutex::new(HashMap::new())),
            cpu_workers,
            gpu_devices,
            cpu_engine,
            gpu_engine,
            progress_interval_ms,
            chunk_size,
            active_jobs_gauge: Arc::new(tokio::sync::Mutex::new(0)),
        }
    }

    pub async fn add_job(&self, job_id: String, mut job: MiningJob) -> Result<(), String> {
        let mut jobs = self.jobs.lock().await;
        if jobs.contains_key(&job_id) {
            log::warn!("Attempted to add duplicate job ID: {job_id}");
            return Err("Job already exists".to_string());
        }

        log::info!("Adding mining job: {}", job_id,);
        job.job_id = Some(job_id.clone());
        #[cfg(feature = "metrics")]
        {
            let engine_name = if self.cpu_workers > 0 && self.gpu_devices > 0 {
                "hybrid"
            } else if self.gpu_devices > 0 {
                "gpu"
            } else {
                "cpu"
            };
            metrics::set_job_status_gauge(engine_name, &job_id, "running", 1);
            metrics::set_job_status_gauge(engine_name, &job_id, "completed", 0);
            metrics::set_job_status_gauge(engine_name, &job_id, "failed", 0);
            metrics::set_job_status_gauge(engine_name, &job_id, "cancelled", 0);
            // increment active jobs
            {
                let mut g = self.active_jobs_gauge.lock().await;
                *g += 1;
                metrics::set_active_jobs(*g);
            }
        }
        job.start_mining(
            self.cpu_workers,
            self.gpu_devices,
            self.cpu_engine.clone(),
            self.gpu_engine.clone(),
            self.progress_interval_ms,
            self.chunk_size,
        );
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
                log::info!("Mining result served for job: {job_id}");
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
            log::info!("Removing mining job: {job_id}");
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
        log::info!("🔄 Starting job monitoring loop...");

        tokio::spawn(async move {
            let mut last_watchdog = std::time::Instant::now();
            let service_start = std::time::Instant::now();
            loop {
                let mut jobs_guard = jobs.lock().await;

                jobs_guard.retain(|job_id, job| {
                    let was_running = job.status == JobStatus::Running;
                    // Always update from results to drain thread completion messages
                    // regardless of job status (until we decide to drop the job)
                    let now_not_running = job.update_from_results();

                    if was_running && now_not_running {
                        log::info!(
                            "Mining job {} finished with status {:?}, hashes: {}, time: {:?}",
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
                        log::info!("🧹 Cleaning up completed job {}", job_id);
                    }
                    retain
                });

                #[cfg(feature = "metrics")]
                {
                    // Update active jobs gauge
                    let mut running_jobs = 0i64;
                    for (_job_id, job) in jobs_guard.iter() {
                        if job.status == JobStatus::Running {
                            running_jobs += 1;
                        }
                    }
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
                    let uptime = service_start.elapsed();
                    let uptime_str = if uptime.as_secs() < 60 {
                        format!("{}s", uptime.as_secs())
                    } else if uptime.as_secs() < 3600 {
                        format!("{}m {}s", uptime.as_secs() / 60, uptime.as_secs() % 60)
                    } else {
                        format!(
                            "{}h {}m",
                            uptime.as_secs() / 3600,
                            (uptime.as_secs() % 3600) / 60
                        )
                    };

                    #[cfg(feature = "metrics")]
                    let total_hash_rate = metrics::get_hash_rate();
                    #[cfg(not(feature = "metrics"))]
                    let total_hash_rate = 0.0;

                    if total == 0 {
                        log::info!(
                            "📊 Mining service healthy - uptime: {} - waiting for jobs",
                            uptime_str
                        );
                    } else if running == 0 {
                        log::info!(
                            "📊 Mining status - uptime: {} - no active jobs - {} completed, {} cancelled, {} failed",
                            uptime_str, completed, cancelled, failed
                        );
                    } else {
                        let hash_rate_str = if total_hash_rate >= 1_000_000.0 {
                            format!("{:.1}M", total_hash_rate / 1_000_000.0)
                        } else if total_hash_rate >= 1_000.0 {
                            format!("{:.1}K", total_hash_rate / 1_000.0)
                        } else if total_hash_rate > 0.0 {
                            format!("{:.0}", total_hash_rate)
                        } else {
                            "starting...".to_string()
                        };

                        log::info!(
                            "📊 Mining status - uptime: {} - jobs: {} active, {} completed, {} cancelled, {} failed - hash rate: {} H/s",
                            uptime_str, running, completed, cancelled, failed, hash_rate_str
                        );
                    }
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
    pub thread_total_hashes: std::collections::HashMap<usize, u64>,
    pub thread_final_rates: Vec<f64>,
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
            thread_total_hashes: self.thread_total_hashes.clone(),
            thread_final_rates: self.thread_final_rates.clone(),
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
    duration: std::time::Duration,
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
            thread_total_hashes: std::collections::HashMap::new(),
            thread_final_rates: Vec::new(),
            completed_threads: 0,
            result_served: false,
        }
    }

    pub fn start_mining(
        &mut self,
        cpu_workers: usize,
        gpu_devices: usize,
        cpu_engine: Option<Arc<dyn MinerEngine>>,
        gpu_engine: Option<Arc<dyn MinerEngine>>,
        progress_interval_ms: u64,
        chunk_size: Option<u64>,
    ) {
        let total_workers = cpu_workers + gpu_devices;
        let chan_capacity = std::env::var("MINER_RESULT_CHANNEL_CAP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| total_workers.saturating_mul(64).max(256));
        let (sender, receiver) = bounded(chan_capacity);
        self.result_receiver = Some(receiver);

        // Set engine name based on what's available
        self.engine_name = if cpu_workers > 0 && gpu_devices > 0 {
            "hybrid"
        } else if gpu_devices > 0 {
            "gpu"
        } else {
            "cpu"
        };

        // Partition nonce range between CPU and GPU workers
        let total_range = self
            .nonce_end
            .saturating_sub(self.nonce_start)
            .saturating_add(primitive_types::U512::from(1u64));
        let mut current_start = self.nonce_start;
        let mut thread_id = 0;

        log::info!(
            "Starting mining with {} CPU + {} GPU workers, total range: {}",
            cpu_workers,
            gpu_devices,
            total_range
        );

        // Create CPU worker threads
        if cpu_workers > 0 {
            if let Some(cpu_engine) = cpu_engine {
                let ctx = cpu_engine.prepare_context(self.header_hash, self.difficulty);
                let cpu_partitions = compute_partitions(
                    current_start,
                    current_start
                        .saturating_add(
                            total_range.saturating_mul(primitive_types::U512::from(cpu_workers))
                                / primitive_types::U512::from(total_workers),
                        )
                        .saturating_sub(primitive_types::U512::from(1u64)),
                    cpu_workers,
                );
                current_start = current_start.saturating_add(
                    total_range.saturating_mul(primitive_types::U512::from(cpu_workers))
                        / primitive_types::U512::from(total_workers),
                );

                for (start, end) in cpu_partitions.ranges.into_iter() {
                    let cancel_flag = self.cancel_flag.clone();
                    let sender = sender.clone();
                    let ctx = ctx.clone();
                    let engine = cpu_engine.clone();
                    let job_id = self.job_id.clone().unwrap_or_else(|| "unknown".to_string());

                    let handle = thread::spawn(move || {
                        mine_range_with_engine_typed(
                            thread_id,
                            job_id,
                            engine.as_ref(),
                            "CPU",
                            ctx,
                            EngineRange { start, end },
                            cancel_flag,
                            sender,
                            progress_interval_ms,
                            chunk_size,
                        );
                    });
                    self.thread_handles.push(handle);
                    thread_id += 1;
                }
            }
        }

        // Create GPU worker threads
        if gpu_devices > 0 {
            if let Some(gpu_engine) = gpu_engine {
                let ctx = gpu_engine.prepare_context(self.header_hash, self.difficulty);
                let gpu_partitions = compute_partitions(current_start, self.nonce_end, gpu_devices);

                for (start, end) in gpu_partitions.ranges.into_iter() {
                    let cancel_flag = self.cancel_flag.clone();
                    let sender = sender.clone();
                    let ctx = ctx.clone();
                    let engine = gpu_engine.clone();
                    let job_id = self.job_id.clone().unwrap_or_else(|| "unknown".to_string());

                    let handle = thread::spawn(move || {
                        mine_range_with_engine_typed(
                            thread_id,
                            job_id,
                            engine.as_ref(),
                            "GPU",
                            ctx,
                            EngineRange { start, end },
                            cancel_flag,
                            sender,
                            progress_interval_ms,
                            chunk_size,
                        );
                    });
                    self.thread_handles.push(handle);
                    thread_id += 1;
                }
            }
        }
    }

    pub fn cancel(&mut self) {
        log::info!(
            "Cancelling mining job: {}",
            self.job_id.as_ref().unwrap_or(&"unknown".to_string())
        );
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
                // Remove all per-thread hash rate series on cancellation
                for (tid, _) in self.thread_total_hashes.iter() {
                    metrics::remove_thread_hash_rate(self.engine_name, job_id, &tid.to_string());
                }
                self.thread_total_hashes.clear();
            }
        }

        while let Some(handle) = self.thread_handles.pop() {
            // Do not wait for threads to finish; detach them so they can exit gracefully
            // when they detect the cancellation flag. This prevents blocking the control loop.
            drop(handle);
        }
    }

    pub fn update_from_results(&mut self) -> bool {
        let receiver = match &self.result_receiver {
            Some(r) => r,
            None => return false,
        };

        while let Ok(thread_result) = receiver.try_recv() {
            self.total_hash_count += thread_result.hash_count;
            *self.thread_total_hashes.entry(thread_result.thread_id).or_default() +=
                thread_result.hash_count;

            #[cfg(feature = "metrics")]
            {
                metrics::inc_hashes(thread_result.hash_count);
                metrics::record_mining_segment(thread_result.hash_count, thread_result.duration);

                // Only update job-specific metrics if the job is still considered running.
                // Once completed/failed/cancelled, we stop updating job metrics to avoid
                // resurrecting series that were cleaned up.
                if self.status == JobStatus::Running {
                    if let Some(job_id) = &self.job_id {
                        metrics::inc_job_hashes(self.engine_name, job_id, thread_result.hash_count);
                        metrics::inc_thread_hashes(
                            self.engine_name,
                            job_id,
                            &thread_result.thread_id.to_string(),
                            thread_result.hash_count,
                        );

                        // Simple per-job hash rate based on total progress
                        let elapsed = self.start_time.elapsed().as_secs_f64();
                        if elapsed > 0.0 {
                            let job_rate = self.total_hash_count as f64 / elapsed;
                            self.last_hash_rate = job_rate;
                            metrics::set_job_hash_rate(self.engine_name, job_id, job_rate);
                        }
                    }
                }
            }

            if thread_result.completed {
                self.completed_threads += 1;
                let thread_total = *self
                    .thread_total_hashes
                    .get(&thread_result.thread_id)
                    .unwrap_or(&0);
                let elapsed = self.start_time.elapsed().as_secs_f64();

                if elapsed > 0.0 && thread_total > 0 {
                    let thread_rate = thread_total as f64 / elapsed;
                    self.thread_final_rates.push(thread_rate);

                    log::info!(
                        "Thread {} finished - Rate: {:.2} H/s ({} hashes in {:.2}s)",
                        thread_result.thread_id,
                        thread_rate,
                        thread_total,
                        elapsed
                    );

                    #[cfg(feature = "metrics")]
                    if let Some(job_id) = &self.job_id {
                        // Report final thread rate before cleaning up
                        metrics::set_thread_hash_rate(
                            self.engine_name,
                            job_id,
                            &thread_result.thread_id.to_string(),
                            thread_rate,
                        );
                    }
                }

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
                        for (tid, _) in self.thread_total_hashes.iter() {
                            metrics::remove_thread_hash_rate(
                                self.engine_name,
                                job_id,
                                &tid.to_string(),
                            );
                        }
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
                        for (tid, _) in self.thread_total_hashes.iter() {
                            metrics::remove_thread_hash_rate(
                                self.engine_name,
                                job_id,
                                &tid.to_string(),
                            );
                        }
                    }
                }
            }
        }

        self.status != JobStatus::Running
    }
}

#[allow(clippy::too_many_arguments)] // Reason: worker runner needs job_id, engine, context, range, cancel flag, sender, and chunking config
fn mine_range_with_engine_typed(
    thread_id: usize,
    job_id: String,
    engine: &dyn MinerEngine,
    engine_type: &str,
    ctx: pow_core::JobContext,
    range: EngineRange,
    cancel_flag: Arc<AtomicBool>,
    sender: Sender<ThreadResult>,
    progress_interval_ms: u64,
    chunk_size: Option<u64>,
) {
    if engine_type == "CPU" {
        log::info!(
            target: "miner-service",
            "CPU thread {} search started: Job {} range {} to {} (inclusive)",
            thread_id,
            job_id,
            range.start,
            range.end
        );
    } else {
        log::debug!(
            target: "miner-service",
            "⛏️  Job {} {} thread {} mining range {} to {} (inclusive)",
            job_id,
            engine_type,
            thread_id,
            range.start,
            range.end
        );
    }

    // Chunk the range into subranges to emit periodic progress updates for metrics
    let mut current_start = range.start;
    let end = range.end;
    // Derive a rough chunk size from target milliseconds and an estimate of hashes/sec.
    let target_ms = progress_interval_ms;
    let chunk_size_u64 = chunk_size.unwrap_or_else(|| {
        // Use engine-specific defaults if not configured
        if engine.name().contains("gpu") {
            // GPU can handle much larger chunks efficiently.
            // Increased to 4B to allow auto-tuned batches (e.g. 3s @ 1GH/s = 3B) to grow sufficiently.
            4_000_000_000
        } else if engine.name() == "hybrid" {
            // Hybrid engines use GPU-sized chunks since they route to GPU workers
            4_000_000_000
        } else {
            // CPU uses time-based chunks
            let est_ops_per_sec = 100_000u64; // 100K ops/sec for CPU
            ((est_ops_per_sec.saturating_mul(target_ms)) / 1000).max(5_000)
        }
    });

    // Log the effective chunk size to assist in debugging/tuning
    if engine_type == "GPU" {
        log::info!(
            target: "miner-service",
            "Job {} GPU thread {} configured with chunk size: {}",
            job_id,
            thread_id,
            chunk_size_u64
        );
    } else {
        log::debug!(
            target: "miner-service",
            "Job {} {} thread {} configured with chunk size: {}",
            job_id,
            engine_type,
            thread_id,
            chunk_size_u64
        );
    }

    let chunk_size = U512::from(chunk_size_u64);
    let mut done = false;
    let mut total_hashes_processed = 0u64;

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
            "Job {} {} thread {} starting search on subrange {}..{} (inclusive)",
            job_id,
            engine_type,
            thread_id,
            sub_range.start,
            sub_range.end
        );
        let start_time = Instant::now();
        let status = engine.search_range(&ctx, sub_range.clone(), &cancel_flag);
        let status_str = match status {
            engine_cpu::EngineStatus::Found { .. } => "found",
            engine_cpu::EngineStatus::Exhausted { .. } => "exhausted",
            engine_cpu::EngineStatus::Cancelled { .. } => "cancelled",
            engine_cpu::EngineStatus::Running { .. } => "running",
        };
        log::info!(
            target: "miner-service",
            "{} thread {} finished search, status: {} job: {}",
            engine_type,
            thread_id,
            status_str,
            job_id,
        );

        match status {
            engine_cpu::EngineStatus::Found {
                candidate: EngineCandidate { nonce, work, hash },
                hash_count,
                origin,
            } => {
                let duration = start_time.elapsed();
                // Send final result with found candidate and the hashes covered in this subrange
                let final_result = ThreadResult {
                    thread_id,
                    result: Some(MiningJobResult { nonce, work, hash }),
                    hash_count,
                    origin: Some(origin),
                    duration,
                    completed: true,
                };
                log::info!(
                    "🎉 Solution found! Job {} {} thread {} - Nonce: {}, Hash: {:x}",
                    job_id,
                    engine_type,
                    thread_id,
                    nonce,
                    hash
                );
                if sender.try_send(final_result).is_err() {
                    log::warn!(
                        "Job {} thread {} failed to send final result",
                        job_id,
                        thread_id
                    );
                }
                done = true;
                break;
            }
            engine_cpu::EngineStatus::Exhausted { hash_count } => {
                let duration = start_time.elapsed();
                total_hashes_processed += hash_count;
                // Send intermediate progress update for this chunk
                let update = ThreadResult {
                    thread_id,
                    result: None,
                    hash_count,
                    origin: None,
                    duration,
                    completed: false,
                };
                if sender.try_send(update).is_err() {
                    log::warn!(
                        "Job {} thread {} failed to send progress update",
                        job_id,
                        thread_id
                    );
                    break;
                } else {
                    log::info!(
                        "⛏️  Job {} {} thread {} processed {} hashes (range: {}..{})",
                        job_id,
                        engine_type,
                        thread_id,
                        hash_count,
                        sub_range.start,
                        sub_range.end
                    );
                }
            }
            engine_cpu::EngineStatus::Cancelled { hash_count } => {
                let duration = start_time.elapsed();
                total_hashes_processed += hash_count;
                // Send last progress update and stop
                let update = ThreadResult {
                    thread_id,
                    result: None,
                    hash_count,
                    origin: None,
                    duration,
                    completed: false,
                };
                if sender.try_send(update).is_err() {
                    log::warn!(target: "miner", "Job {job_id} thread {thread_id} failed to send cancel update");
                }
                if engine_type == "CPU" {
                    log::info!(
                        target: "miner-service",
                        "CPU thread {} search cancelled: Job {} processed {} hashes (total: {})",
                        thread_id,
                        job_id,
                        hash_count,
                        total_hashes_processed
                    );
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

    // Check if loop exited due to cancellation flag
    if !done && cancel_flag.load(Ordering::Relaxed) {
        if engine_type == "CPU" {
            log::info!(
                target: "miner-service",
                "CPU thread {} search cancelled (flag set): Job {} processed {} hashes",
                thread_id,
                job_id,
                total_hashes_processed
            );
        }
        done = true;
    }

    if !done {
        // Signal thread completion with no final candidate
        let final_result = ThreadResult {
            thread_id,
            result: None,
            hash_count: 0,
            origin: None,
            duration: std::time::Duration::from_secs(0),
            completed: true,
        };
        if sender.try_send(final_result).is_err() {
            log::warn!(target: "miner", "Job {job_id} thread {thread_id} failed to send completion status after chunked search");
        }
        if engine_type == "CPU" {
            log::info!(
                target: "miner-service",
                "CPU thread {} search completed: Job {} exhausted range, processed {} hashes",
                thread_id,
                job_id,
                total_hashes_processed
            );
        }
    }

    // Explicitly clear thread-local GPU resources to avoid TLS order panic
    if engine_type == "GPU" {
        engine_gpu::GpuEngine::clear_worker_resources();
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
    log::debug!(target: "miner-servce", "Mine request: {request:?}");
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
            log::debug!(target: "miner-servce", "Accepted mine request for job ID: {}", request.job_id);
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
    log::debug!(target: "miner-servce", "Result request: {job_id}");

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

    let elapsed = job.start_time.elapsed().as_secs_f64();
    match job.status {
        JobStatus::Running => {
            log::debug!(
                "🔍 Job {} still running - elapsed: {:.1}s - hash rate: {:.0} H/s",
                job_id,
                elapsed,
                job.last_hash_rate
            );
        }
        JobStatus::Completed if !job.result_served => {
            log::info!(
                "✅ Job {} completed - ready for pickup - elapsed: {:.1}s - {} hashes",
                job_id,
                elapsed,
                job.total_hash_count
            );
        }
        JobStatus::Completed => {
            log::debug!(
                "📤 Job {} result already served - elapsed: {:.1}s",
                job_id,
                elapsed
            );
        }
        _ => {
            log::debug!(
                "🔄 Job {} status: {:?} - elapsed: {:.1}s",
                job_id,
                job.status,
                elapsed
            );
        }
    }
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
    log::info!(target: "miner-servce", "Cancel job: {job_id}");

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

    Partitions { ranges }
}

pub fn resolve_gpu_configuration(
    requested_devices: Option<usize>,
    gpu_batch_duration_ms: Option<u64>,
) -> anyhow::Result<(Option<Arc<dyn MinerEngine>>, usize)> {
    // Default to 3000ms if not specified
    let duration = std::time::Duration::from_millis(gpu_batch_duration_ms.unwrap_or(3000));

    if let Some(req_count) = requested_devices {
        if req_count == 0 {
            return Ok((None, 0));
        }
        // Explicit request > 0
        let engine = engine_gpu::GpuEngine::try_new(duration)
            .map_err(|e| anyhow::anyhow!("Failed to initialize GPU engine: {}", e))?;

        let available = engine.device_count();
        if req_count > available {
            return Err(anyhow::anyhow!(
                "Requested {} GPU devices but only {} device(s) are available.",
                req_count,
                available
            ));
        }
        Ok((Some(Arc::new(engine)), req_count))
    } else {
        // Auto-detect
        match engine_gpu::GpuEngine::try_new(duration) {
            Ok(engine) => {
                let available = engine.device_count();
                if available > 0 {
                    log::info!(
                        "Auto-detected {} GPU device(s). Using all available GPUs.",
                        available
                    );
                    Ok((Some(Arc::new(engine)), available))
                } else {
                    log::info!("GPU auto-detection found 0 devices. Defaulting to CPU only.");
                    Ok((None, 0))
                }
            }
            Err(e) => {
                log::info!("GPU auto-detection failed (no suitable GPU found): {}. Defaulting to CPU only.", e);
                Ok((None, 0))
            }
        }
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

    // Initialize GPU engine if requested or for auto-detection
    let (gpu_engine, gpu_devices): (Option<Arc<dyn MinerEngine>>, usize) =
        match resolve_gpu_configuration(config.gpu_devices, config.gpu_batch_duration_ms) {
            Ok((engine, count)) => (engine, count),
            Err(e) => {
                log::error!("❌ ERROR: {}", e);
                if config.gpu_devices.is_some() {
                    log::error!("   Please check your --gpu-devices setting or GPU hardware.");
                    std::process::exit(1);
                } else {
                    (None, 0)
                }
            }
        };

    // Calculate CPU workers
    let cpu_workers = config.cpu_workers.unwrap_or_else(|| {
        if gpu_devices == 0 {
            // CPU-only mode: use default CPU allocation
            let default_cpu_workers = effective_cpus
                .saturating_sub(effective_cpus / 2)
                .min(effective_cpus.saturating_sub(1))
                .max(1);
            log::info!(
                "No CPU workers specified. Defaulting to {} CPU workers (leaving ~{} for other processes).",
                default_cpu_workers,
                effective_cpus.saturating_sub(default_cpu_workers)
            );
            default_cpu_workers
        } else {
            // Hybrid mode: default to half of effective CPUs for CPU
            let default_cpu_workers = effective_cpus / 2;
            log::info!(
                "Hybrid mode: defaulting to {} CPU workers",
                default_cpu_workers
            );
            default_cpu_workers
        }
    });

    let total_workers = cpu_workers + gpu_devices;

    let (cpu_workers, gpu_devices) = if total_workers == 0 {
        log::warn!(
            "No workers specified. Defaulting to CPU-only mode with {} workers.",
            effective_cpus
        );
        (effective_cpus, 0)
    } else {
        (cpu_workers, gpu_devices)
    };

    // Create engines based on worker configuration
    let cpu_engine = if cpu_workers > 0 {
        Some(Arc::new(engine_cpu::FastCpuEngine::new()) as Arc<dyn MinerEngine>)
    } else {
        None
    };

    // Log the mining configuration
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

    log::info!("⚙️  Service configuration: {config}");

    let progress_interval_ms = config.progress_interval_ms.unwrap_or(10000);
    let service = MiningService::new(
        cpu_workers,
        gpu_devices,
        cpu_engine,
        gpu_engine,
        progress_interval_ms,
        config.chunk_size,
    );

    log::info!(
        "⛏️  Mining service ready with {} total worker threads",
        total_workers
    );
    if let Some(chunk_size) = config.chunk_size {
        log::info!(
            "📦 Custom chunk size: {} hashes per progress update",
            chunk_size
        );
    }
    log::info!("⏰ Progress reporting interval: {}ms", progress_interval_ms);

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
        let engine_name_str = if cpu_workers > 0 && gpu_devices > 0 {
            "hybrid".to_string()
        } else if gpu_devices > 0 {
            "gpu".to_string()
        } else {
            "cpu".to_string()
        };
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
                    workers: Some((svc.cpu_workers + svc.gpu_devices) as u32),
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
    log::info!("🌐 HTTP API server starting on http://{}", socket);
    log::info!("📡 Mining endpoints available:");
    log::info!("   POST http://{}/mine - Submit mining jobs", socket);
    log::info!(
        "   GET  http://{}/result/{{job_id}} - Check mining results",
        socket
    );
    if config.metrics_port.is_some() {
        log::info!(
            "   GET  http://{}:{}/metrics - Prometheus metrics",
            socket.ip(),
            config.metrics_port.unwrap()
        );
    }
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
        let cpu_engine: Arc<dyn MinerEngine> = Arc::new(engine_cpu::FastCpuEngine::new());
        let state = MiningService::new(2, 0, Some(cpu_engine), None, 2000, None);

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
        let cpu_engine: Arc<dyn MinerEngine> = Arc::new(engine_cpu::FastCpuEngine::new());
        let state = MiningService::new(1, 0, Some(cpu_engine), None, 2000, None);
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
            // total_range field removed, verify by checking all ranges sum to 100
            let total: u64 = p.ranges.iter().map(|(s, e)| (e - s + 1).low_u64()).sum();
            assert_eq!(total, 100);
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
            // total_range field removed, verify by checking all ranges sum to 100
            let total: u64 = p.ranges.iter().map(|(s, e)| (e - s + 1).low_u64()).sum();
            assert_eq!(total, 100);
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
            // total_range field removed, verify by checking range sums to 1
            let total: u64 = p
                .ranges
                .iter()
                .map(|(s, e)| {
                    if e >= s {
                        (e - s + 1).low_u64()
                    } else {
                        0 // Empty range
                    }
                })
                .sum();
            assert_eq!(total, 1);
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
        let cpu_engine: Arc<dyn MinerEngine> = Arc::new(engine_cpu::FastCpuEngine::new());
        let state = MiningService::new(2, 0, Some(cpu_engine), None, 2000, None);
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

        let cpu_engine: Arc<dyn MinerEngine> = Arc::new(engine_cpu::BaselineCpuEngine::new());
        let service = MiningService::new(2, 0, Some(cpu_engine), None, 1000, None);
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
        let engine = engine_cpu::FastCpuEngine::new();
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

        super::mine_range_with_engine_typed(
            0,
            "test-job".to_string(),
            &engine,
            "CPU",
            ctx,
            range,
            cancel,
            tx,
            10,   // ms (min chunk size is 5k; fine for testing progress + completion)
            None, // chunk_size
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

    #[tokio::test]
    async fn test_cancel_returns_immediately() {
        use engine_cpu::{EngineRange, EngineStatus, MinerEngine};
        use pow_core::JobContext;
        use std::any::Any;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::time::{Duration, Instant};

        struct SlowEngine;

        impl MinerEngine for SlowEngine {
            fn name(&self) -> &'static str {
                "slow-engine"
            }
            fn prepare_context(&self, header_hash: [u8; 32], difficulty: U512) -> JobContext {
                JobContext::new(header_hash, difficulty)
            }
            fn search_range(
                &self,
                _ctx: &JobContext,
                _range: EngineRange,
                cancel: &AtomicBool,
            ) -> EngineStatus {
                // Sleep to simulate a long-running batch on GPU
                std::thread::sleep(Duration::from_secs(2));

                if cancel.load(Ordering::Relaxed) {
                    EngineStatus::Cancelled { hash_count: 0 }
                } else {
                    EngineStatus::Exhausted { hash_count: 0 }
                }
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
        }

        let engine = Arc::new(SlowEngine);
        let service = MiningService::new(1, 0, Some(engine), None, 10000, None);

        let job = MiningJob::new([0u8; 32], U512::MAX, U512::zero(), U512::from(1000u64));

        service
            .add_job("slow-job".to_string(), job)
            .await
            .expect("Failed to add job");

        // Give thread a moment to start and enter sleep
        tokio::time::sleep(Duration::from_millis(100)).await;

        let start = Instant::now();
        service.cancel_job("slow-job").await;
        let elapsed = start.elapsed();

        // Cancel should be near-instant (<< 2 seconds)
        assert!(
            elapsed < Duration::from_millis(500),
            "Cancel took too long: {:?} (expected < 500ms)",
            elapsed
        );

        // Verify job is actually cancelled
        let job = service.get_job("slow-job").await.unwrap();
        assert_eq!(job.status, JobStatus::Cancelled);
    }
}

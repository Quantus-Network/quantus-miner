#![deny(rust_2018_idioms)]
#![forbid(unsafe_code)]

use crossbeam_channel::{bounded, Receiver, Sender};
use engine_cpu::{EngineCandidate, EngineRange, MinerEngine};
use primitive_types::U512;
use resonance_miner_api::*;
use std::collections::HashMap;
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
    /// Engine selection (future use). For now, CPU baseline/fast engines are supported.
    pub engine: EngineSelection,
}

/// Engine selection enum for future extensibility.
#[derive(Clone, Debug)]
pub enum EngineSelection {
    CpuBaseline,
    CpuFast,
    // Cuda,
    // OpenCl,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            port: 9833,
            workers: None,
            metrics_port: None,
            progress_chunk_ms: None,
            engine: EngineSelection::CpuBaseline,
        }
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
            log::warn!("Attempted to add duplicate job ID: {}", job_id);
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

    pub async fn remove_job(&self, job_id: &str) -> Option<MiningJob> {
        let mut jobs = self.jobs.lock().await;
        if let Some(mut job) = jobs.remove(job_id) {
            log::debug!(target: "miner", "Removing job: {}", job_id);
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
            loop {
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

                    // Retain jobs that are running or recently finished (e.g., within 5 minutes)
                    let retain = job.status == JobStatus::Running
                        || job.start_time.elapsed().as_secs() < 300;
                    if !retain {
                        log::debug!(target: "miner", "Cleaning up old job {}", job_id);
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
                drop(jobs_guard);
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
    pub distance: U512,
}

/// Mining job data structure stored in the service.
#[derive(Debug)]
pub struct MiningJob {
    pub header_hash: [u8; 32],
    pub distance_threshold: U512,
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
}

impl Clone for MiningJob {
    fn clone(&self) -> Self {
        MiningJob {
            header_hash: self.header_hash,
            distance_threshold: self.distance_threshold,
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
        }
    }
}

#[derive(Debug, Clone)]
pub struct ThreadResult {
    thread_id: usize,
    result: Option<MiningJobResult>,
    hash_count: u64,
    completed: bool,
}

impl MiningJob {
    pub fn new(
        header_hash: [u8; 32],
        distance_threshold: U512,
        nonce_start: U512,
        nonce_end: U512,
    ) -> Self {
        MiningJob {
            header_hash,
            distance_threshold,
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
        }
    }

    pub fn start_mining(
        &mut self,
        engine: Arc<dyn MinerEngine>,
        workers: usize,
        progress_chunk_ms: u64,
    ) {
        let (sender, receiver) = bounded(workers * 2);
        self.result_receiver = Some(receiver);
        self.engine_name = engine.name();

        let total_range = (self.nonce_end - self.nonce_start).saturating_add(U512::from(1));
        let range_per_worker = total_range / U512::from(workers as u64);
        let remainder = total_range % U512::from(workers as u64);

        log::debug!(
            target: "miner",
            "Starting mining with {} workers, total range: {}, range per worker: {}",
            workers,
            total_range,
            range_per_worker
        );

        // Prepare shared job context once per job.
        let ctx = engine.prepare_context(self.header_hash, self.distance_threshold);

        for thread_id in 0..workers {
            let start = self.nonce_start + range_per_worker * U512::from(thread_id as u64);
            let mut end = start + range_per_worker - U512::from(1);

            if thread_id == workers - 1 {
                end += remainder;
            }
            if end > self.nonce_end {
                end = self.nonce_end;
            }

            let cancel_flag = self.cancel_flag.clone();
            let sender = sender.clone();
            let ctx = ctx.clone();
            let engine = engine.clone();
            let progress_chunk_ms = progress_chunk_ms;

            let handle = thread::spawn(move || {
                mine_range_with_engine(
                    thread_id,
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
        log::debug!(target: "miner", "Cancelling mining job");
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
                    .is_none_or(|current_best| result.distance < current_best.distance);

                if is_better {
                    log::debug!(target: "miner",
                        "Found better result from thread {}: distance = {}, nonce = {}",
                        thread_result.thread_id,
                        result.distance,
                        result.nonce
                    );
                    self.best_result = Some(result);
                    self.cancel_flag.store(true, Ordering::Relaxed);
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
                        metrics::set_job_status_gauge(self.engine_name, job_id, "running", 0);
                        metrics::set_job_status_gauge(self.engine_name, job_id, "completed", 1);
                        metrics::set_job_status_gauge(self.engine_name, job_id, "failed", 0);
                        metrics::set_job_status_gauge(self.engine_name, job_id, "cancelled", 0);
                        // Remove job hash rate on completion and clear per-thread series
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

fn mine_range_with_engine(
    thread_id: usize,
    engine: &dyn MinerEngine,
    ctx: pow_core::JobContext,
    range: EngineRange,
    cancel_flag: Arc<AtomicBool>,
    sender: Sender<ThreadResult>,
    progress_chunk_ms: u64,
) {
    log::debug!(
        "Thread {} mining range {} to {} (inclusive)",
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

        let status = engine.search_range(&ctx, sub_range.clone(), &cancel_flag);

        match status {
            engine_cpu::EngineStatus::Found {
                candidate:
                    EngineCandidate {
                        nonce,
                        work,
                        distance,
                    },
                hash_count,
            } => {
                // Send final result with found candidate and the hashes covered in this subrange
                let final_result = ThreadResult {
                    thread_id,
                    result: Some(MiningJobResult {
                        nonce,
                        work,
                        distance,
                    }),
                    hash_count,
                    completed: true,
                };
                if sender.send(final_result).is_err() {
                    log::warn!("Thread {} failed to send final result", thread_id);
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
                    completed: false,
                };
                if sender.send(update).is_err() {
                    log::warn!("Thread {} failed to send progress update", thread_id);
                    break;
                }
            }
            engine_cpu::EngineStatus::Cancelled { hash_count } => {
                // Send last progress update and stop
                let update = ThreadResult {
                    thread_id,
                    result: None,
                    hash_count,
                    completed: false,
                };
                if sender.send(update).is_err() {
                    log::warn!("Thread {} failed to send cancel update", thread_id);
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
            completed: true,
        };
        if sender.send(final_result).is_err() {
            log::warn!(
                "Thread {} failed to send completion status after chunked search",
                thread_id
            );
        }
    }

    log::debug!("Thread {} completed.", thread_id);
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
    log::debug!("Received mine request: {:?}", request);
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
    let distance_threshold = U512::from_dec_str(&request.distance_threshold).unwrap();
    let nonce_start = U512::from_str_radix(&request.nonce_start, 16).unwrap();
    let nonce_end = U512::from_str_radix(&request.nonce_end, 16).unwrap();

    let job = MiningJob::new(header_hash, distance_threshold, nonce_start, nonce_end);

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
    log::debug!("Received result request for job: {}", job_id);

    let job = match state.get_job(&job_id).await {
        Some(job) => job,
        None => {
            log::warn!("Result request for unknown job: {}", job_id);
            return Ok(warp::reply::with_status(
                warp::reply::json(&resonance_miner_api::MiningResult {
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

    let status = match job.status {
        JobStatus::Running => ApiResponseStatus::Running,
        JobStatus::Completed => ApiResponseStatus::Completed,
        JobStatus::Failed => ApiResponseStatus::Failed,
        JobStatus::Cancelled => ApiResponseStatus::Cancelled,
    };

    let (nonce, work) = match &job.best_result {
        Some(result) => (
            Some(format!("{:x}", result.nonce)),
            Some(hex::encode(result.work)),
        ),
        None => (None, None),
    };

    let elapsed_time = job.start_time.elapsed().as_secs_f64();

    Ok(warp::reply::with_status(
        warp::reply::json(&resonance_miner_api::MiningResult {
            status: status,
            job_id,
            nonce,
            work,
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
    log::debug!("Received cancel request for job: {}", job_id);

    if state.cancel_job(&job_id).await {
        log::debug!(target: "miner", "Successfully cancelled job: {}", job_id);
        Ok(warp::reply::with_status(
            warp::reply::json(&MiningResponse {
                status: ApiResponseStatus::Cancelled,
                job_id,
                message: None,
            }),
            warp::http::StatusCode::OK,
        ))
    } else {
        log::warn!("Cancel request for unknown job: {}", job_id);
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

/// Start the miner service with the given configuration.
/// - Spawns the mining loop.
/// - Optionally exposes a metrics endpoint if `metrics_port` is provided and the `metrics` feature is enabled.
/// - Serves the HTTP API on `config.port`.
pub async fn run(config: ServiceConfig) -> anyhow::Result<()> {
    // Determine available logical CPUs, preferring the cgroup cpuset when present.
    fn detect_effective_cpus() -> usize {
        // Try cgroup v2 effective cpuset
        if let Ok(mask) = std::fs::read_to_string("/sys/fs/cgroup/cpuset.cpus.effective") {
            if let Some(count) = parse_cpuset_to_count(mask.trim()) {
                return count.max(1);
            }
        }
        // Fallback to legacy cgroup v1 path
        if let Ok(mask) = std::fs::read_to_string("/sys/fs/cgroup/cpuset/cpuset.cpus") {
            if let Some(count) = parse_cpuset_to_count(mask.trim()) {
                return count.max(1);
            }
        }
        // Fallback to all logical CPUs
        num_cpus::get().max(1)
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

    // Detect effective CPU pool for this process (cpuset if available).
    let effective_cpus = detect_effective_cpus();

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
                    "Requested {} workers exceeds available logical CPUs in cpuset ({}). Clamping to {}.",
                    n,
                    effective_cpus,
                    effective_cpus
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

    log::info!(
        "Using {} worker thread(s) for mining (effective logical CPUs available: {})",
        workers,
        effective_cpus
    );

    // Select engine
    #[allow(unused_mut)]
    let mut engine: Arc<dyn MinerEngine> = match config.engine {
        EngineSelection::CpuBaseline => Arc::new(engine_cpu::BaselineCpuEngine::new()),
        EngineSelection::CpuFast => Arc::new(engine_cpu::FastCpuEngine::new()),
    };
    log::info!("Using engine: {}", engine.name());

    let progress_chunk_ms = config.progress_chunk_ms.unwrap_or(2000);
    let service = MiningService::new(workers, engine.clone(), progress_chunk_ms);

    // Start mining loop
    service.start_mining_loop().await;

    // Optionally start metrics exporter if enabled via CLI and feature flag.
    if let Some(port) = config.metrics_port {
        #[cfg(feature = "metrics")]
        {
            log::info!("Starting metrics endpoint on 0.0.0.0:{}", port);
            metrics::start_http_exporter(port).await?;
        }
        #[cfg(not(feature = "metrics"))]
        {
            log::warn!(
                "Metrics port provided ({}), but 'metrics' feature is not enabled. Skipping.",
                port
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
    log::info!("Server starting on {}", socket);
    warp::serve(routes).run(socket).await;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_mining_state_add_get_remove() {
        let engine: Arc<dyn MinerEngine> = Arc::new(engine_cpu::BaselineCpuEngine::new());
        let state = MiningService::new(2, engine);

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
        // Test that a job fails if no nonce is found (threshold too strict)
        let engine: Arc<dyn MinerEngine> = Arc::new(engine_cpu::BaselineCpuEngine::new());
        let state = MiningService::new(2, engine);
        state.start_mining_loop().await;

        // Impossible threshold
        let header_hash = [1u8; 32];
        let distance_threshold = U512::zero();
        let nonce_start = U512::from(0);
        let nonce_end = U512::from(100);

        let job = MiningJob::new(header_hash, distance_threshold, nonce_start, nonce_end);
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
        assert!(finished_job.total_hash_count >= 0);
    }

    #[tokio::test]
    async fn test_job_lifecycle_success() {
        // Test that a job completes successfully
        let engine: Arc<dyn MinerEngine> = Arc::new(engine_cpu::BaselineCpuEngine::new());
        let state = MiningService::new(2, engine);
        state.start_mining_loop().await;

        // Easy threshold
        let header_hash = [1u8; 32];
        let distance_threshold = U512::MAX; // Easiest difficulty
        let nonce_start = U512::from(0);
        let nonce_end = U512::from(10000);

        let job = MiningJob::new(header_hash, distance_threshold, nonce_start, nonce_end);
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
}

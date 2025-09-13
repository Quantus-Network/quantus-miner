#![deny(rust_2018_idioms)]
#![forbid(unsafe_code)]

//! Minimal metrics scaffolding for the Quantus External Miner.
//!
//! - Provides a global Prometheus registry and a handful of default metrics.
//! - Exposes helper functions to update metrics from the service layer.
//! - Optionally runs a Warp-based HTTP endpoint (/metrics) when the
//!   `http-exporter` feature is enabled. This is gated at runtime by
//!   the presence of a metrics port in the CLI (`--metrics-port`).
//!
//! Default metrics:
//! - miner_jobs_total{status}        : number of jobs by terminal state
//! - miner_hashes_total              : total nonces tested
//! - miner_hash_rate                 : current estimated hash rate (nonces/sec)
//! - miner_http_requests_total{code,endpoint} : HTTP request counts for miner API
//!
//! Notes:
//! - The HTTP exporter is spawned as a background task and does not block.
//! - If the `http-exporter` feature is disabled, `start_http_exporter` becomes
//!   a no-op that returns immediately.
//! - When jobs/threads end, prefer removing gauge label children (series) rather than
//!   writing a zero; this avoids scrape-timing artifacts and produces cleaner rollups.
//! - Service-level observability: expose an `active_jobs` gauge and a `mine_requests_total`
//!   counter (labeled by result) to disambiguate "idle because no jobs" vs "actively mining".

use once_cell::sync::Lazy;
use prometheus::{
    opts, Encoder, Gauge, GaugeVec, IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Registry,
    TextEncoder,
};

#[cfg(feature = "http-exporter")]
use {
    anyhow::Result,
    std::net::SocketAddr,
    warp::{http::Response, Filter},
};

#[cfg(not(feature = "http-exporter"))]
use anyhow::Result;

// -------------------------------------------------------------------------------------
// Global Registry and Default Metrics
// -------------------------------------------------------------------------------------

static REGISTRY: Lazy<Registry> = Lazy::new(Registry::new);

// -------------------------------------------------------------------------------------
// High-level service metrics
// -------------------------------------------------------------------------------------

static ACTIVE_JOBS: Lazy<IntGauge> = Lazy::new(|| {
    let g = IntGauge::new(
        "miner_active_jobs",
        "Number of currently running mining jobs",
    )
    .expect("create miner_active_jobs");
    REGISTRY
        .register(Box::new(g.clone()))
        .expect("register miner_active_jobs");
    g
});

static ENGINE_BACKEND_INFO: Lazy<GaugeVec> = Lazy::new(|| {
    let g = GaugeVec::new(
        opts!(
            "miner_engine_backend",
            "Engine backend info (label-only gauge set to 1). Labels: engine, backend"
        ),
        &["engine", "backend"],
    )
    .expect("create miner_engine_backend");
    REGISTRY
        .register(Box::new(g.clone()))
        .expect("register miner_engine_backend");
    g
});

static EFFECTIVE_CPUS: Lazy<IntGauge> = Lazy::new(|| {
    let g = IntGauge::new(
        "miner_effective_cpus",
        "Detected logical CPU capacity available to this process (cpuset-aware)",
    )
    .expect("create miner_effective_cpus");
    REGISTRY
        .register(Box::new(g.clone()))
        .expect("register miner_effective_cpus");
    g
});

static MINE_REQUESTS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    let c = IntCounterVec::new(
        opts!(
            "miner_mine_requests_total",
            "Count of /mine requests by result"
        ),
        &["result"], // accepted, duplicate, invalid, error
    )
    .expect("create miner_mine_requests_total");
    REGISTRY
        .register(Box::new(c.clone()))
        .expect("register miner_mine_requests_total");
    c
});

static JOBS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    let c = IntCounterVec::new(
        opts!("miner_jobs_total", "Number of jobs by status"),
        &["status"],
    )
    .expect("create miner_jobs_total");
    REGISTRY
        .register(Box::new(c.clone()))
        .expect("register miner_jobs_total");
    c
});

static HASHES_TOTAL: Lazy<IntCounter> = Lazy::new(|| {
    let c = IntCounter::new("miner_hashes_total", "Total nonces tested")
        .expect("create miner_hashes_total");
    REGISTRY
        .register(Box::new(c.clone()))
        .expect("register miner_hashes_total");
    c
});

static HASH_RATE: Lazy<Gauge> = Lazy::new(|| {
    let g =
        Gauge::new("miner_hash_rate", "Estimated hash rate (nonces per second)").expect("create");
    REGISTRY
        .register(Box::new(g.clone()))
        .expect("register miner_hash_rate");
    g
});

static HTTP_REQUESTS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    let c = IntCounterVec::new(
        opts!(
            "miner_http_requests_total",
            "HTTP requests count by endpoint and status code"
        ),
        &["endpoint", "code"],
    )
    .expect("create miner_http_requests_total");
    REGISTRY
        .register(Box::new(c.clone()))
        .expect("register miner_http_requests_total");
    c
});

/// Access the global Prometheus registry used by this crate.
// Engine-aware, per-job, and per-thread labeled metrics
pub fn set_effective_cpus(n: i64) {
    EFFECTIVE_CPUS.set(n);
}

static JOB_HASHES_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    let c = IntCounterVec::new(
        opts!(
            "miner_job_hashes_total",
            "Total nonces tested per job and engine"
        ),
        &["engine", "job_id"],
    )
    .expect("create miner_job_hashes_total");
    REGISTRY
        .register(Box::new(c.clone()))
        .expect("register miner_job_hashes_total");
    c
});

static JOB_HASH_RATE: Lazy<GaugeVec> = Lazy::new(|| {
    let g = GaugeVec::new(
        opts!(
            "miner_job_hash_rate",
            "Estimated hash rate (nonces per second) per job and engine"
        ),
        &["engine", "job_id"],
    )
    .expect("create miner_job_hash_rate");
    REGISTRY
        .register(Box::new(g.clone()))
        .expect("register miner_job_hash_rate");
    g
});

static JOB_ESTIMATED_RATE: Lazy<GaugeVec> = Lazy::new(|| {
    let g = GaugeVec::new(
        opts!(
            "miner_job_estimated_rate",
            "Estimated work rate (nonces per second) per job and engine"
        ),
        &["engine", "job_id"],
    )
    .expect("create miner_job_estimated_rate");
    REGISTRY
        .register(Box::new(g.clone()))
        .expect("register miner_job_estimated_rate");
    g
});

static CANDIDATES_FOUND_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    let c = IntCounterVec::new(
        opts!(
            "miner_candidates_found_total",
            "Total candidates found per job and engine"
        ),
        &["engine", "job_id"],
    )
    .expect("create miner_candidates_found_total");
    REGISTRY
        .register(Box::new(c.clone()))
        .expect("register miner_candidates_found_total");
    c
});

static CANDIDATES_FALSE_POSITIVE_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    let c = IntCounterVec::new(
        opts!(
            "miner_candidates_false_positive_total",
            "Total false-positive candidates rejected by host re-verification per engine"
        ),
        &["engine"],
    )
    .expect("create miner_candidates_false_positive_total");
    REGISTRY
        .register(Box::new(c.clone()))
        .expect("register miner_candidates_false_positive_total");
    c
});

static SAMPLE_MISMATCH_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    let c = IntCounterVec::new(
        opts!(
            "miner_sample_mismatch_total",
            "Total decision parity mismatches between engine and host per engine"
        ),
        &["engine"],
    )
    .expect("create miner_sample_mismatch_total");
    REGISTRY
        .register(Box::new(c.clone()))
        .expect("register miner_sample_mismatch_total");
    c
});

static THREAD_HASHES_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    let c = IntCounterVec::new(
        opts!(
            "miner_thread_hashes_total",
            "Total nonces tested per thread, job, and engine"
        ),
        &["engine", "job_id", "thread_id"],
    )
    .expect("create miner_thread_hashes_total");
    REGISTRY
        .register(Box::new(c.clone()))
        .expect("register miner_thread_hashes_total");
    c
});

static THREAD_HASH_RATE: Lazy<GaugeVec> = Lazy::new(|| {
    let g = GaugeVec::new(
        opts!(
            "miner_thread_hash_rate",
            "Estimated hash rate (nonces per second) per thread, job, and engine"
        ),
        &["engine", "job_id", "thread_id"],
    )
    .expect("create miner_thread_hash_rate");
    REGISTRY
        .register(Box::new(g.clone()))
        .expect("register miner_thread_hash_rate");
    g
});

static JOBS_BY_ENGINE_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    let c = IntCounterVec::new(
        opts!(
            "miner_jobs_by_engine_total",
            "Number of jobs by engine and terminal status"
        ),
        &["engine", "status"],
    )
    .expect("create miner_jobs_by_engine_total");
    REGISTRY
        .register(Box::new(c.clone()))
        .expect("register miner_jobs_by_engine_total");
    c
});

static JOB_STATUS_GAUGE: Lazy<IntGaugeVec> = Lazy::new(|| {
    let g = IntGaugeVec::new(
        opts!(
            "miner_job_status",
            "Job status gauge (set to 1 for current status)"
        ),
        &["engine", "job_id", "status"],
    )
    .expect("create miner_job_status");
    REGISTRY
        .register(Box::new(g.clone()))
        .expect("register miner_job_status");
    g
});

pub fn default_registry() -> &'static Registry {
    &REGISTRY
}

// -------------------------------------------------------------------------------------
// Update Helpers (to be called from the service layer)
// -------------------------------------------------------------------------------------

/// Increment the total hashes counter by `n` (number of nonces tested).
// Labeled helpers for engine/job/thread metrics
//
// Service-level helpers
pub fn set_active_jobs(n: i64) {
    ACTIVE_JOBS.set(n);
}

pub fn set_engine_backend(engine: &str, backend: &str) {
    ENGINE_BACKEND_INFO
        .with_label_values(&[engine, backend])
        .set(1.0);
}

pub fn inc_mine_requests(result: &str) {
    MINE_REQUESTS_TOTAL.with_label_values(&[result]).inc();
}

pub fn inc_job_hashes(engine: &str, job_id: &str, n: u64) {
    JOB_HASHES_TOTAL
        .with_label_values(&[engine, job_id])
        .inc_by(n);
}

pub fn set_job_hash_rate(engine: &str, job_id: &str, rate: f64) {
    JOB_HASH_RATE.with_label_values(&[engine, job_id]).set(rate);
}

pub fn set_job_estimated_rate(engine: &str, job_id: &str, rate: f64) {
    JOB_ESTIMATED_RATE
        .with_label_values(&[engine, job_id])
        .set(rate);
}

pub fn inc_candidates_found(engine: &str, job_id: &str) {
    CANDIDATES_FOUND_TOTAL
        .with_label_values(&[engine, job_id])
        .inc();
}

pub fn inc_candidates_false_positive(engine: &str) {
    CANDIDATES_FALSE_POSITIVE_TOTAL
        .with_label_values(&[engine])
        .inc();
}

pub fn inc_sample_mismatch(engine: &str) {
    SAMPLE_MISMATCH_TOTAL.with_label_values(&[engine]).inc();
}

pub fn inc_thread_hashes(engine: &str, job_id: &str, thread_id: &str, n: u64) {
    THREAD_HASHES_TOTAL
        .with_label_values(&[engine, job_id, thread_id])
        .inc_by(n);
}

pub fn set_thread_hash_rate(engine: &str, job_id: &str, thread_id: &str, rate: f64) {
    THREAD_HASH_RATE
        .with_label_values(&[engine, job_id, thread_id])
        .set(rate);
}

pub fn inc_jobs_by_engine(engine: &str, status: &str) {
    JOBS_BY_ENGINE_TOTAL
        .with_label_values(&[engine, status])
        .inc();
}

pub fn set_job_status_gauge(engine: &str, job_id: &str, status: &str, value: i64) {
    JOB_STATUS_GAUGE
        .with_label_values(&[engine, job_id, status])
        .set(value);
}

pub fn inc_hashes(n: u64) {
    HASHES_TOTAL.inc_by(n);
}

/// Set the current estimated hash rate (nonces per second).
pub fn set_hash_rate(rate: f64) {
    HASH_RATE.set(rate);
}

/// Increment the jobs counter for a terminal status: completed | failed | cancelled.
pub fn inc_job_status(status: &str) {
    JOBS_TOTAL.with_label_values(&[status]).inc();
}

/// Increment HTTP request counters for an endpoint with a status code.
pub fn inc_http_request(endpoint: &str, code: u16) {
    HTTP_REQUESTS_TOTAL
        .with_label_values(&[endpoint, &code.to_string()])
        .inc();
}

// -------------------------------------------------------------------------------------
// Removal helpers for end-of-life series
// -------------------------------------------------------------------------------------

/// Remove the per-job hash rate series for a finished job.
pub fn remove_job_hash_rate(engine: &str, job_id: &str) {
    let _ = JOB_HASH_RATE.remove_label_values(&[engine, job_id]);
}

/// Remove the per-thread hash rate series for a finished thread.
pub fn remove_thread_hash_rate(engine: &str, job_id: &str, thread_id: &str) {
    let _ = THREAD_HASH_RATE.remove_label_values(&[engine, job_id, thread_id]);
}

// -------------------------------------------------------------------------------------
// HTTP Exporter (feature: http-exporter)
// -------------------------------------------------------------------------------------

/// Start the Prometheus HTTP exporter on 0.0.0.0:`port`.
///
/// Behavior:
/// - Spawns the exporter as a background task and returns immediately.
/// - Serves plaintext metrics at GET /metrics.
/// - If called multiple times, multiple servers may be created (call once).
pub async fn start_http_exporter(port: u16) -> Result<()> {
    // Encoder is created inside the handler to avoid capturing non-Clone state
    // Use REGISTRY.gather() directly in the handler

    // GET /metrics -> plaintext Prometheus format
    let metrics_route = warp::path("metrics").and(warp::get()).map(|| {
        let encoder = TextEncoder::new();
        let metric_families = REGISTRY.gather();
        let mut buffer = Vec::with_capacity(16 * 1024);
        encoder
            .encode(&metric_families, &mut buffer)
            .unwrap_or_default();

        Response::builder()
            .header("Content-Type", encoder.format_type())
            .body(String::from_utf8(buffer).unwrap_or_default())
    });

    let addr: SocketAddr = ([0, 0, 0, 0], port).into();
    tokio::spawn(async move {
        warp::serve(metrics_route).run(addr).await;
    });

    Ok(())
}

// If the exporter feature is not enabled, expose a no-op to keep call sites simple.
#[cfg(not(feature = "http-exporter"))]
pub async fn start_http_exporter(_port: u16) -> Result<()> {
    log::warn!(
        "metrics::start_http_exporter called but 'http-exporter' feature is disabled; ignoring"
    );
    Ok(())
}

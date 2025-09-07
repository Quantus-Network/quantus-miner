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

use once_cell::sync::Lazy;
use prometheus::{
    opts, Encoder, Gauge, IntCounter, IntCounterVec, Registry, TextEncoder,
};

#[cfg(feature = "http-exporter")]
use {
    anyhow::Result,
    serde::Serialize,
    std::net::SocketAddr,
    warp::{http::Response, Filter},
};

#[cfg(not(feature = "http-exporter"))]
use anyhow::Result;

// -------------------------------------------------------------------------------------
// Global Registry and Default Metrics
// -------------------------------------------------------------------------------------

static REGISTRY: Lazy<Registry> = Lazy::new(Registry::new);

static JOBS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    let c = IntCounterVec::new(opts!("miner_jobs_total", "Number of jobs by status"), &["status"])
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
pub fn default_registry() -> &'static Registry {
    &REGISTRY
}

// -------------------------------------------------------------------------------------
// Update Helpers (to be called from the service layer)
// -------------------------------------------------------------------------------------

/// Increment the total hashes counter by `n` (number of nonces tested).
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
// HTTP Exporter (feature: http-exporter)
// -------------------------------------------------------------------------------------

#[cfg(feature = "http-exporter")]
#[derive(Serialize)]
struct MetricsInfo {
    endpoint: &'static str,
}

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
        encoder.encode(&metric_families, &mut buffer).unwrap_or_default();

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

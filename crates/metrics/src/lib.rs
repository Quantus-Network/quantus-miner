#![deny(rust_2018_idioms)]
#![forbid(unsafe_code)]

//! Minimal metrics for the Quantus External Miner.
//!
//! Exposes a small set of Prometheus metrics for monitoring mining performance:
//! - `miner_hash_rate`: Total hash rate (CPU + GPU combined)
//! - `miner_cpu_hash_rate`: CPU-only hash rate
//! - `miner_gpu_hash_rate`: GPU-only hash rate
//! - `miner_hashes_total`: Total hashes computed (all time)
//! - `miner_active_jobs`: Currently running jobs (0 or 1)
//! - `miner_workers`: Total worker count
//! - `miner_cpu_workers`: Number of CPU workers
//! - `miner_gpu_devices`: Number of GPU devices
//! - `miner_effective_cpus`: Detected CPU cores
//!
//! Optionally runs a Warp-based HTTP endpoint (`/metrics`) when the
//! `http-exporter` feature is enabled.

use once_cell::sync::Lazy;
use prometheus::{IntCounter, IntGauge, Registry};
use std::sync::Mutex;
use std::time::Instant;

#[cfg(feature = "http-exporter")]
use {
    anyhow::Result,
    prometheus::{Encoder, TextEncoder},
    std::net::SocketAddr,
    warp::Filter,
};

#[cfg(not(feature = "http-exporter"))]
use anyhow::Result;

// ---------------------------------------------------------------------------
// Global Registry
// ---------------------------------------------------------------------------

static REGISTRY: Lazy<Registry> = Lazy::new(Registry::new);

// ---------------------------------------------------------------------------
// Hash Rate Metrics
// ---------------------------------------------------------------------------

static HASH_RATE: Lazy<IntGauge> = Lazy::new(|| {
    let g = IntGauge::new("miner_hash_rate", "Total hash rate in hashes per second")
        .expect("create miner_hash_rate");
    REGISTRY
        .register(Box::new(g.clone()))
        .expect("register miner_hash_rate");
    g
});

static CPU_HASH_RATE: Lazy<IntGauge> = Lazy::new(|| {
    let g = IntGauge::new("miner_cpu_hash_rate", "CPU hash rate in hashes per second")
        .expect("create miner_cpu_hash_rate");
    REGISTRY
        .register(Box::new(g.clone()))
        .expect("register miner_cpu_hash_rate");
    g
});

static GPU_HASH_RATE: Lazy<IntGauge> = Lazy::new(|| {
    let g = IntGauge::new("miner_gpu_hash_rate", "GPU hash rate in hashes per second")
        .expect("create miner_gpu_hash_rate");
    REGISTRY
        .register(Box::new(g.clone()))
        .expect("register miner_gpu_hash_rate");
    g
});

// ---------------------------------------------------------------------------
// Counter Metrics
// ---------------------------------------------------------------------------

static HASHES_TOTAL: Lazy<IntCounter> = Lazy::new(|| {
    let c = IntCounter::new("miner_hashes_total", "Total hashes computed")
        .expect("create miner_hashes_total");
    REGISTRY
        .register(Box::new(c.clone()))
        .expect("register miner_hashes_total");
    c
});

// ---------------------------------------------------------------------------
// Gauge Metrics
// ---------------------------------------------------------------------------

static ACTIVE_JOBS: Lazy<IntGauge> = Lazy::new(|| {
    let g = IntGauge::new("miner_active_jobs", "Number of currently running mining jobs")
        .expect("create miner_active_jobs");
    REGISTRY
        .register(Box::new(g.clone()))
        .expect("register miner_active_jobs");
    g
});

static WORKERS: Lazy<IntGauge> = Lazy::new(|| {
    let g = IntGauge::new("miner_workers", "Total number of worker threads")
        .expect("create miner_workers");
    REGISTRY
        .register(Box::new(g.clone()))
        .expect("register miner_workers");
    g
});

static CPU_WORKERS: Lazy<IntGauge> = Lazy::new(|| {
    let g = IntGauge::new("miner_cpu_workers", "Number of CPU worker threads")
        .expect("create miner_cpu_workers");
    REGISTRY
        .register(Box::new(g.clone()))
        .expect("register miner_cpu_workers");
    g
});

static GPU_DEVICES: Lazy<IntGauge> = Lazy::new(|| {
    let g = IntGauge::new("miner_gpu_devices", "Number of GPU devices")
        .expect("create miner_gpu_devices");
    REGISTRY
        .register(Box::new(g.clone()))
        .expect("register miner_gpu_devices");
    g
});

static EFFECTIVE_CPUS: Lazy<IntGauge> = Lazy::new(|| {
    let g = IntGauge::new(
        "miner_effective_cpus",
        "Detected logical CPU cores available to this process",
    )
    .expect("create miner_effective_cpus");
    REGISTRY
        .register(Box::new(g.clone()))
        .expect("register miner_effective_cpus");
    g
});

// ---------------------------------------------------------------------------
// Hash Rate Tracking
// ---------------------------------------------------------------------------

/// Tracks cumulative hashes to compute rolling hash rates.
struct HashRateTracker {
    /// Cumulative CPU hashes
    cpu_total: u64,
    /// Cumulative GPU hashes  
    gpu_total: u64,
    /// When tracking started (or was last reset)
    start_time: Instant,
}

impl HashRateTracker {
    fn new() -> Self {
        Self {
            cpu_total: 0,
            gpu_total: 0,
            start_time: Instant::now(),
        }
    }

    fn record_cpu(&mut self, hashes: u64) {
        self.cpu_total += hashes;
        self.update_rates();
    }

    fn record_gpu(&mut self, hashes: u64) {
        self.gpu_total += hashes;
        self.update_rates();
    }

    fn update_rates(&self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            let cpu_rate = (self.cpu_total as f64 / elapsed) as i64;
            let gpu_rate = (self.gpu_total as f64 / elapsed) as i64;
            CPU_HASH_RATE.set(cpu_rate);
            GPU_HASH_RATE.set(gpu_rate);
            HASH_RATE.set(cpu_rate + gpu_rate);
        }
    }
}

static HASH_TRACKER: Lazy<Mutex<HashRateTracker>> =
    Lazy::new(|| Mutex::new(HashRateTracker::new()));

// ---------------------------------------------------------------------------
// Public API - Hash Recording
// ---------------------------------------------------------------------------

/// Record CPU hashes and update hash rate metrics.
pub fn record_cpu_hashes(n: u64) {
    HASHES_TOTAL.inc_by(n);
    if let Ok(mut tracker) = HASH_TRACKER.lock() {
        tracker.record_cpu(n);
    }
}

/// Record GPU hashes and update hash rate metrics.
pub fn record_gpu_hashes(n: u64) {
    HASHES_TOTAL.inc_by(n);
    if let Ok(mut tracker) = HASH_TRACKER.lock() {
        tracker.record_gpu(n);
    }
}

// ---------------------------------------------------------------------------
// Public API - Hash Rates (for direct setting, kept for compatibility)
// ---------------------------------------------------------------------------

/// Set the total hash rate (CPU + GPU combined).
pub fn set_hash_rate(rate: i64) {
    HASH_RATE.set(rate);
}

/// Set the CPU-only hash rate.
pub fn set_cpu_hash_rate(rate: i64) {
    CPU_HASH_RATE.set(rate);
}

/// Set the GPU-only hash rate.
pub fn set_gpu_hash_rate(rate: i64) {
    GPU_HASH_RATE.set(rate);
}

// ---------------------------------------------------------------------------
// Public API - Counters (kept for compatibility)
// ---------------------------------------------------------------------------

/// Increment the total hashes counter (without updating rates).
/// Prefer `record_cpu_hashes` or `record_gpu_hashes` instead.
pub fn inc_hashes(n: u64) {
    HASHES_TOTAL.inc_by(n);
}

// ---------------------------------------------------------------------------
// Public API - Gauges
// ---------------------------------------------------------------------------

/// Set the number of active jobs (0 or 1).
pub fn set_active_jobs(n: i64) {
    ACTIVE_JOBS.set(n);
}

/// Set the total number of workers.
pub fn set_workers(n: i64) {
    WORKERS.set(n);
}

/// Set the number of CPU workers.
pub fn set_cpu_workers(n: i64) {
    CPU_WORKERS.set(n);
}

/// Set the number of GPU devices.
pub fn set_gpu_devices(n: i64) {
    GPU_DEVICES.set(n);
}

/// Set the detected CPU core count.
pub fn set_effective_cpus(n: i64) {
    EFFECTIVE_CPUS.set(n);
}

// ---------------------------------------------------------------------------
// HTTP Exporter
// ---------------------------------------------------------------------------

/// Start the Prometheus HTTP exporter on `0.0.0.0:port`.
///
/// Spawns the exporter as a background task and returns immediately.
/// Serves plaintext metrics at `GET /metrics`.
#[cfg(feature = "http-exporter")]
pub async fn start_http_exporter(port: u16) -> Result<()> {
    let metrics_route = warp::path("metrics").and(warp::get()).map(|| {
        let encoder = TextEncoder::new();
        let metric_families = REGISTRY.gather();
        let mut buffer = Vec::with_capacity(4096);
        encoder
            .encode(&metric_families, &mut buffer)
            .unwrap_or_default();

        warp::http::Response::builder()
            .header("Content-Type", encoder.format_type())
            .body(String::from_utf8(buffer).unwrap_or_default())
    });

    let addr: SocketAddr = ([0, 0, 0, 0], port).into();
    tokio::spawn(async move {
        warp::serve(metrics_route).run(addr).await;
    });

    Ok(())
}

/// No-op when HTTP exporter feature is disabled.
#[cfg(not(feature = "http-exporter"))]
pub async fn start_http_exporter(_port: u16) -> Result<()> {
    log::warn!(
        "metrics::start_http_exporter called but 'http-exporter' feature is disabled; ignoring"
    );
    Ok(())
}

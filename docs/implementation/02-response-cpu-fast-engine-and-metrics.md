# Response Summary: CPU Fast Engine and Metrics Integration (Extended)

This document summarizes the implementation of an optimized CPU mining engine and the integration of Prometheus metrics into the miner service, including engine-/job-/thread-aware metrics and per-thread rate smoothing. It corresponds to the prompt captured in `02-prompt-cpu-fast-engine-and-metrics.md`.

## Overview

- Added a new optimized CPU engine (`cpu-fast`) that leverages precomputation and incremental updates to drastically reduce per-nonce work.
- Integrated Prometheus metrics (optional at runtime) for:
  - Job terminal states (completed/failed/cancelled),
  - Total number of nonces tested,
  - Aggregate hash rate (nonces/sec) across running jobs,
  - Engine-aware per-job and per-thread totals and rates (via labels).
- Implemented per-thread hash rate gauges computed via delta-rate in the service and smoothed using EMA with a code constant.
- Preserved the node-facing HTTP API and maintained separation of concerns via a clean engine abstraction.

## Changes Implemented

- New engine implementation in `crates/engine-cpu/src/lib.rs`:
  - `FastCpuEngine` implements `MinerEngine` using pow-core’s optimized helpers.
  - Keeps cancellation and range semantics identical to baseline.
- CLI engine selection in `crates/miner-cli/src/main.rs`:
  - `--engine` now accepts `cpu-baseline` and `cpu-fast`.
  - Default is `cpu-fast`.
- Service updates in `crates/miner-service/src/lib.rs`:
  - `EngineSelection` supports `CpuFast`.
  - When `CpuFast` is selected, the service instantiates `FastCpuEngine`.
- Metrics integration:
  - Runtime toggle: exporter starts only when `--metrics-port` is provided.
  - Counters and gauges in `crates/metrics/src/lib.rs` and called from the service:
    - Global:
      - `miner_jobs_total{status="completed|failed|cancelled"}`
      - `miner_hashes_total`
      - `miner_hash_rate`
    - Engine-/job-aware:
      - `miner_job_hashes_total{engine,job_id}`
      - `miner_job_hash_rate{engine,job_id}`
      - `miner_job_status{engine,job_id,status}` (IntGauge set to 1 for current status)
      - `miner_jobs_by_engine_total{engine,status}`
    - Engine-/job-/thread-aware:
      - `miner_thread_hashes_total{engine,job_id,thread_id}`
      - `miner_thread_hash_rate{engine,job_id,thread_id}`

## Fast CPU Engine: Algorithm Details

File: `crates/engine-cpu/src/lib.rs`
- `FastCpuEngine::prepare_context`:
  - Builds a `pow_core::JobContext` that precomputes `(m, n)` and the `target` value from the header and threshold.
- `FastCpuEngine::search_range`:
  - Initializes `y0` for the starting nonce using `pow_core::init_worker_y0(ctx, start)`.
  - Loops over the inclusive nonce range:
    - Uses `pow_core::distance_from_y(ctx, y)` to compute the current distance.
    - Checks validity via `pow_core::is_valid_distance(ctx, distance)`.
    - If not valid, advances one step with `pow_core::step_mul(ctx, y)` (O(1) modular multiplication).
    - Supports prompt cancellation via an atomic flag.
  - Returns `Found(candidate)`, `Exhausted`, or `Cancelled` with an accumulated `hash_count`.

Correctness:
- The incremental path (y <- y*m mod n) is equivalent to computing `m^(h + nonce)` via exponentiation, but avoids the heavy per-nonce modular exponentiation.
- We validate per-nonce by computing distance from `y` and comparing it to the threshold.

## Service Integration

File: `crates/miner-service/src/lib.rs`
- Engine selection extended:
  - `EngineSelection::CpuFast` added.
  - `run(ServiceConfig)` branches to instantiate `FastCpuEngine` when selected.
- Mining loop unchanged:
  - Orchestration retains responsibility for job lifecycle, range partitioning, cancellation, and result aggregation.
  - The engine’s interface abstracts the compute details.

## Metrics Integration

Files:
- `crates/metrics/src/lib.rs` (Prometheus registry and HTTP exporter)
- `crates/miner-service/src/lib.rs` (hook points for updates)

Key behavior:
- Metrics are disabled by default and only activated when `--metrics-port` is provided in the CLI.
- When active:
  - Global metrics:
    - `miner_jobs_total{status="completed"}` increments on job completion.
    - `miner_jobs_total{status="failed"}` increments when a job fails.
    - `miner_jobs_total{status="cancelled"}` increments when cancelled.
    - `miner_hashes_total` increments by `hash_count` deltas from worker threads.
    - `miner_hash_rate` is periodically set from aggregate job progress.
  - Engine-/job metrics:
    - `miner_job_hashes_total{engine,job_id}` increments by deltas.
    - `miner_job_hash_rate{engine,job_id}` set from per-job aggregate progress.
    - `miner_job_status{engine,job_id,status}` gauges kept mutually exclusive (1 for current state).
    - `miner_jobs_by_engine_total{engine,status}` increments on terminal transitions.
  - Engine-/job-/thread metrics:
    - `miner_thread_hashes_total{engine,job_id,thread_id}` increments by per-thread deltas.
    - `miner_thread_hash_rate{engine,job_id,thread_id}` set via delta-rate with EMA smoothing.

Exporter:
- Serves `/metrics` on the specified port when enabled.

## Per-thread Delta Rate and EMA Smoothing

- Each `MiningJob` tracks:
  - `thread_last_update: HashMap<usize, Instant>`
  - `thread_rate_ema: HashMap<usize, f64>`
- On each `ThreadResult`:
  - Compute `dt = now - last_update(thread_id)`; if `dt > 0` and `hash_count > 0`, compute `instant_rate = hash_count / dt`.
  - Smooth with EMA: `ema = alpha * instant_rate + (1 - alpha) * prev_ema`.
  - Set `miner_thread_hash_rate{engine,job_id,thread_id} = ema`.
  - Update `thread_last_update` and `thread_rate_ema`.
- On thread completion, remove its tracking entries.
- The EMA coefficient is a code constant in the service:
  - `THREAD_RATE_EMA_ALPHA: f64 = 0.2`
  - We’ll revisit this value once we collect real-world data.

## Backward Compatibility

- The HTTP API exposed to the node is unchanged:
  - Endpoints: `POST /mine`, `GET /result/{job_id}`, `POST /cancel/{job_id}`.
  - Types sourced from `resonance-miner-api` remain intact.
- All metrics additions are internal and optional.

## Build and Run

- Build (release):
  - `cargo build -p miner-cli --release`
- Run (fast engine, no metrics):
  - `cargo run -p miner-cli -- --port 9833 --engine cpu-fast`
- Run with metrics:
  - `cargo run -p miner-cli -- --port 9833 --engine cpu-fast --metrics-port 9900`
  - Scrape: `http://localhost:9900/metrics`

CLI defaults:
- `--engine` defaults to `cpu-fast`.
- Metrics remain off unless `--metrics-port` is supplied.

## Verification

- Compilation in release mode completes successfully after minor fixes.
- Runtime behavior mirrors baseline engine semantics with improved performance potential.
- Metrics endpoint activates only when requested and otherwise remains inert.
- Per-thread hash rate gauges reflect smoothed rates, reducing jitter.

## Notes and Next Steps

- Integrate Montgomery multiplication into `pow-core` and switch the fast engine to use it for even higher throughput.
- Optional: add per-job/engine labels to result responses for correlation (kept internal to metrics for now).
- Optional: promote the `MinerEngine` trait to a small shared crate (e.g., `engine-api`) for GPU engine implementations.
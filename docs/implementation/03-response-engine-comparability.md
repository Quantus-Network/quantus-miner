# Response Summary (03): Engine `Found` hash_count Included for Accurate Metrics, Logging/CLI Updates, and Dashboards

This document summarizes the implementation that augments the mining engine’s result reporting so that the final “Found” event carries the per-thread `hash_count`. This enables accurate global, per-job, and per-thread metrics (including rates), improving Grafana comparisons between engines (e.g., `cpu-baseline` vs `cpu-fast`).

## Overview

- Extended the engine result type so `Found` includes `{ candidate, hash_count }`.
- Updated both CPU engines to return the final `hash_count` on `Found`.
- Modified the service to forward and accumulate this count:
  - Global totals (`miner_hashes_total`)
  - Per-job totals (`miner_job_hashes_total{engine,job_id}`)
  - Per-thread totals (`miner_thread_hashes_total{engine,job_id,thread_id}`)
  - Per-thread hash-rate gauge via delta-rate with EMA smoothing
- No changes to the node-facing HTTP API.

## Changes Implemented

### 1) Engine API

- File: `crates/engine-cpu/src/lib.rs`
- Updated `EngineStatus`:
  - Before: `Found(Candidate)`
  - Now: `Found { candidate: Candidate, hash_count: u64 }`
- Other variants (`Exhausted`, `Cancelled`, `Running`) retain their `hash_count` shape for consistency.

### 2) Engines

- Files: `crates/engine-cpu/src/lib.rs`
- `BaselineCpuEngine`:
  - On discovering a solution, returns `EngineStatus::Found { candidate, hash_count }` with the accumulated count for the current range.
- `FastCpuEngine`:
  - Same behavior as baseline but using the incremental path (init once with `init_worker_y0`, then `step_mul` per nonce).

### 3) Service

- File: `crates/miner-service/src/lib.rs`
- `mine_range_with_engine`:
  - Updated to extract `hash_count` from `EngineStatus::Found` and include it in the `ThreadResult`.
- `update_from_results`:
  - Accumulates `hash_count` into:
    - `total_hash_count`
    - Global/engine-aware per-job/per-thread counters
  - Per-thread delta-rate calculation with EMA smoothing now uses the `hash_count` from `Found` as well.
- All metric updates are engine-aware and job-/thread-labeled where appropriate.

### 4) Metrics

- No new metrics introduced; existing labeled series are now fully accurate for winning threads:
  - Global:
    - `miner_jobs_total{status}`
    - `miner_hashes_total`
    - `miner_hash_rate`
  - Engine-/job-aware:
    - `miner_job_hashes_total{engine,job_id}`
    - `miner_job_hash_rate{engine,job_id}`
    - `miner_job_status{engine,job_id,status}` (IntGauge)
    - `miner_jobs_by_engine_total{engine,status}`
  - Engine-/job-/thread-aware:
    - `miner_thread_hashes_total{engine,job_id,thread_id}`
    - `miner_thread_hash_rate{engine,job_id,thread_id}` (EMA-smoothed)

## Compatibility

- The public HTTP API (endpoints and payloads) remains unchanged.
- Engine abstraction change is internal to the workspace; service updated accordingly.
- Works with both `cpu-baseline` and `cpu-fast`, enabling apples-to-apples comparisons using the `engine` label in Grafana.

## Build and Run

- Build (release):
  - `cargo build -p miner-cli --release`
- Run (baseline, no metrics):
  - `cargo run -p miner-cli -- --port 9833 --engine cpu-baseline`
- Run (fast engine, with metrics):
  - `cargo run -p miner-cli -- --port 9833 --engine cpu-fast --metrics-port 9900`
  - Scrape: `http://localhost:9900/metrics`

## Verification

- Compile-time: Verified in release mode after updating the service to consume the new `Found` shape.
- Runtime: The winning thread’s final work is now counted in totals and informs the per-thread rate EMA, reducing undercounting and improving rate fidelity.

## Notes and Next Steps

- With winning-thread `hash_count` included, Grafana dashboards can fairly compare engines across global/job/thread metrics.
- Next logical optimization steps:
  - Add correctness tests that compare baseline vs fast engine outcomes on small ranges (golden tests).
  - Add benchmark harness to quantify nonces/sec improvements.
  - Implement Montgomery multiplication in `pow-core` and switch `cpu-fast` to it for further speedups.
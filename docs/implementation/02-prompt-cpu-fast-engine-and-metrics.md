# Prompt Summary: CPU Fast Engine and Metrics Integration (Extended with Engine-/Job-/Thread-aware Metrics and Smoothing)

## Context

Following the prior repository restructuring into a Cargo workspace with clear separation of concerns (CLI, service, engines, and math core), you requested the next implementation steps focusing on performance and observability:

- Introduce an optimized CPU mining engine leveraging the new `pow-core` incremental helpers.
- Integrate Prometheus metrics into the service layer with a runtime toggle based on `--metrics-port`.
- Extend metrics to be engine-, job-, and thread-aware with labeled counters/gauges.
- Add per-thread hash rate gauges computed via delta-rate in the service, with EMA smoothing and a code-level constant for alpha.

This work must preserve API compatibility with the node and fit into the existing engine abstraction.

## Objectives

1. Add a new `cpu-fast` engine in `engine-cpu`:
   - Implement the `MinerEngine` trait using the optimized, incremental path from `pow-core`:
     - Initialize `y0` for the range start using `init_worker_y0`.
     - Iterate nonces using `step_mul` per step (O(1) modular multiplication) instead of per-nonce exponentiation.
     - Compute distances via `distance_from_y` and evaluate against the threshold with `is_valid_distance`.
   - Ensure cancellation and range handling semantics remain identical to the baseline engine.
   - Keep the public HTTP API unchanged.

2. Hook up foundational metrics in `miner-service`:
   - Add counters for job terminal states: `completed`, `failed`, `cancelled`.
   - Track total number of nonces evaluated (hashes total).
   - Compute and expose a basic aggregate hash rate (nonces/second) across running jobs.
   - Respect the runtime toggle: metrics are enabled only when `--metrics-port` is provided; otherwise metrics are disabled entirely.

3. Extend metrics with engine-aware per-job and per-thread labels:
   - Per-job totals and rates labeled by `{engine, job_id}`.
   - Per-thread totals and rates labeled by `{engine, job_id, thread_id}`.
   - A gauge series to reflect current job status labeled by `{engine, job_id, status}`.
   - A counter for jobs by engine and terminal status `{engine, status}`.

4. Implement per-thread delta-rate calculation with EMA smoothing:
   - Compute per-thread hash rate as `delta_hashes / delta_time` on each worker update.
   - Smooth the per-thread rate using EMA with a code constant (e.g., `THREAD_RATE_EMA_ALPHA = 0.2`).

## Requirements and Constraints

- Maintain full compatibility with the node’s HTTP API and `resonance-miner-api` types.
- Keep orchestration and compute concerns separate via the `MinerEngine` trait.
- Metrics must impose no runtime cost or bindings when `--metrics-port` is not provided.
- Preserve existing job lifecycle and cancellation behavior.
- Ensure code compiles cleanly in release mode.
- Avoid protocol or public API changes; all metrics additions are internal and optional.

## Acceptance Criteria

- A new `FastCpuEngine` exists in `engine-cpu` implementing `MinerEngine` and using:
  - `pow-core::init_worker_y0` at the start of the range;
  - `pow-core::step_mul` for each subsequent nonce in the range;
  - `pow-core::distance_from_y` and `pow-core::is_valid_distance` for validation.
- The service can select this engine via CLI (e.g., `--engine cpu-fast`) without breaking other engines.
- When metrics are enabled (via `--metrics-port`), the service:
  - Increments a jobs counter by status on completion/failure/cancellation.
  - Increments total hashes by the amount of work reported from workers.
  - Exposes:
    - A global hash rate gauge computed from aggregate hashes/elapsed time across running jobs.
    - Per-job hash rate gauges labeled `{engine, job_id}`.
    - Per-thread hash rate gauges labeled `{engine, job_id, thread_id}` computed via delta-rate in the service.
  - Exposes labeled totals:
    - `miner_job_hashes_total{engine,job_id}`
    - `miner_thread_hashes_total{engine,job_id,thread_id}`
  - Tracks job status via:
    - `miner_job_status{engine,job_id,status}` IntGauge set to 1 for the current status
    - `miner_jobs_by_engine_total{engine,status}` counters
- No changes to the node-facing HTTP endpoints or payload formats.
- Successful `cargo build --release`.

## Deliverables (Code-Level)

- `engine-cpu`:
  - `FastCpuEngine` implementing the incremental mining loop.
- `miner-service`:
  - Engine selection supporting `cpu-fast`.
  - Metrics integration points for job status, total hashes, aggregate and per-job hash rates.
  - Per-thread delta-rate computation and EMA smoothing with a code constant (`THREAD_RATE_EMA_ALPHA`).
- `metrics` crate:
  - Labeled metrics (engine/job/thread) and helper functions for increments/sets.
  - Optional HTTP exporter gated by the `--metrics-port` runtime toggle.

## Out of Scope (for this prompt)

- Montgomery multiplication integration (future optimization).
- GPU engines (CUDA/OpenCL) implementation.
- Persisted metrics storage or external observability systems (Grafana dashboards can be configured separately).

## Notes

- The fast engine enables substantial performance gains by replacing per-nonce modular exponentiation with per-nonce modular multiplication after a single initialization exponentiation per worker.
- Metrics integration provides observability for benchmarking and tuning while remaining fully optional at runtime.
- Per-thread hash rate smoothing reduces jitter and produces more stable telemetry; we’ll revisit `THREAD_RATE_EMA_ALPHA` after collecting real-world data.
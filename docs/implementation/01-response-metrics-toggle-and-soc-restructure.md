# Response Summary: Metrics Toggle and Separation-of-Concerns (SoC) Restructure

This document summarizes the implementation work performed to restructure the repository for better separation of concerns and to add a runtime metrics toggle suitable for Prometheus scraping. It is tied to the prompt captured in `01-prompt-metrics-toggle-and-soc-restructure.md`.

## Objectives Addressed

- Convert this repository into a Cargo workspace with clear boundaries between protocol, orchestration, and compute.
- Bring the QPoW math functionality locally (compatibility layer plus new optimized structure), to enable focused optimization work.
- Maintain full compatibility with the node-facing HTTP API and request/response formats.
- Add a metrics exporter that is enabled only when a metrics port is provided via a CLI flag.

## High-Level Architecture

The repository was reorganized into a workspace with the following crates:

- crates/miner-cli
  - CLI binary. Parses flags, initializes logging, and runs the service.
  - Flags: `--port`, `--num-cores`, `--metrics-port`, `--engine`.
- crates/miner-service
  - Service layer: HTTP API (same endpoints), job orchestration, concurrency, and engine selection.
  - Converts HTTP requests into jobs; uses an engine to scan nonce ranges and aggregates results.
- crates/pow-core
  - Local QPoW math core. Provides:
    - `compat` API to mirror the external `qpow-math` behavior (for compatibility).
    - New `JobContext` with precomputed constants for future optimizations.
    - Incremental helpers to replace per-nonce exponentiation with per-nonce modular multiplication.
- crates/engine-cpu
  - Defines the `MinerEngine` trait (abstraction boundary between orchestration and compute).
  - Baseline CPU engine implementation (reference path).
- crates/engine-gpu-cuda (placeholder)
  - Scaffold for a CUDA backend; no device code yet.
- crates/engine-gpu-opencl (placeholder)
  - Scaffold for an OpenCL backend; no device code yet.
- crates/metrics
  - Prometheus registry and optional HTTP exporter for `/metrics`.

The old root `src/lib.rs` and `src/main.rs` are deprecated. New entrypoint is `crates/miner-cli`.

## Key Implementation Details

### 1) Workspace Setup
- The root `Cargo.toml` now defines a workspace:
  - members: `miner-cli`, `miner-service`, `pow-core`, `engine-cpu`, `engine-gpu-cuda`, `engine-gpu-opencl`, `metrics`.
  - common dependency versions via `[workspace.dependencies]`.
  - release profile tuned for performance (LTO, opt-level=3, etc.).

### 2) Engine Abstraction
- Introduced `MinerEngine` trait in `engine-cpu`, used by the service layer:
  - `fn prepare_context(&self, header_hash: [u8; 32], threshold: U512) -> JobContext`
  - `fn search_range(&self, ctx: &JobContext, range: Range, cancel: &AtomicBool) -> EngineStatus`
- Implemented `BaselineCpuEngine` as the reference engine (linear scan using `pow-core` reference path).
- The service uses only this trait; engine choice is configured by the CLI.

### 3) pow-core (Local QPoW Math Core)
- Created a local `pow-core` crate to:
  - Provide a `compat` module mirroring the external `qpow-math`’s API (`is_valid_nonce`, `get_nonce_distance`, etc.).
  - Introduce a `JobContext` (header, threshold, derived `m`, `n`, and `target`).
  - Provide incremental helpers:
    - `init_worker_y0(ctx, start_nonce)` → one-time exponentiation per worker.
    - `step_mul(ctx, y)` → advance by one nonce with a single modular multiplication.
    - `distance_from_y(ctx, y)` and `distance_for_nonce(ctx, nonce)`.
- This structure supports the planned optimization where per-nonce mod exponentiation is replaced by a per-nonce modular multiplication.

### 4) miner-service
- Rewrote the service to:
  - Manage jobs with `MiningService` (jobs map, start/stop/cancel, cleanup).
  - Spawn worker threads partitioning nonce ranges evenly.
  - Prepare a single `JobContext` per job and pass ranges to the engine workers.
  - Aggregate results via channels and update job status (Running/Completed/Failed/Cancelled).
  - Expose unchanged HTTP endpoints:
    - `POST /mine`
    - `GET /result/{job_id}`
    - `POST /cancel/{job_id}`

### 5) CLI (miner-cli)
- New binary that:
  - Parses `--port`, `--num-cores`, `--metrics-port`, `--engine`.
  - Sets up logging and runs the service with the chosen engine.
- Examples:
  - `cargo run -p miner-cli -- --port 9833`
  - `cargo run -p miner-cli -- --port 9833 --metrics-port 9900`
  - `cargo run -p miner-cli -- --num-cores 4`

### 6) Metrics Toggle and Exporter
- Metrics are a runtime toggle:
  - If `--metrics-port` is provided, the service starts an HTTP exporter at `/metrics` on the given port.
  - If `--metrics-port` is absent, metrics are disabled entirely (no exporter is started).
- `crates/metrics`:
  - Provides a global Prometheus registry and useful counters/gauges.
  - Exposes `start_http_exporter(port)` that is a no-op when the exporter feature is not enabled.
- The `miner-service` crate enables metrics by default (feature) so runtime toggling via `--metrics-port` Just Works.

## Compatibility

- The node-facing HTTP API is preserved (same endpoints and data structures).
- `resonance-miner-api` remains the source of truth for request/response types.
- Engine internals and math core changes are internal; orchestration encapsulates the compute layer.

## Deprecations

- Root `src/lib.rs` and `src/main.rs` are deprecated; they now print guidance to use the new CLI:
  - `cargo run -p miner-cli -- [args...]`
- Future code changes should target the workspace crates, not the old root.

## Documentation and Audit Trail

- Added `docs/implementation/` for structured prompts and responses:
  - `01-prompt-metrics-toggle-and-soc-restructure.md`
  - `01-response-metrics-toggle-and-soc-restructure.md` (this document)
- This process will continue for future implementation prompts to maintain a thorough audit trail.

## Build and Run

- Build (release):
  - `cargo build -p miner-cli --release`
- Run (examples):
  - `cargo run -p miner-cli -- --port 9833`
  - `cargo run -p miner-cli -- --port 9833 --metrics-port 9900`
  - `cargo run -p miner-cli -- --num-cores 4`

## Next Steps

- Add a “cpu-fast” engine in `engine-cpu` that uses `pow-core`’s incremental path (`init_worker_y0` + `step_mul`).
- Hook up metrics updates in the service (job counters, hash totals, hash rate) now that the toggle is in place.
- Optionally promote the `MinerEngine` trait into a neutral crate (e.g., `engine-api`) to remove the minor coupling to `engine-cpu`.
- Implement Montgomery multiplication in `pow-core` and integrate it into the CPU fast path.
- Add optional GPU engines (CUDA/OpenCL) behind flags, using the same engine interface.

## Notes

- This summary does not make claims about build/test status in your environment. Use the commands above to build/run locally.
- The `tests/` folder may contain references to the previous single-crate structure; those tests should be migrated to the new service crate’s API in a follow-up change.

# CPU Mining Algorithm and Job Lifecycle

This document explains how the external miner performs CPU mining for the Resonance (Quantus) network, how jobs are created and tracked, and how results are produced and reported over the HTTP API.

The implementation details referenced below correspond to the crate’s runtime behavior as implemented in:
- `src/lib.rs` (algorithm, job management, HTTP handlers)
- `src/main.rs` (CLI/configuration, server bootstrap)
- `api/openapi.yaml` and `EXTERNAL_MINER_PROTOCOL.md` (public API and protocol context)

## Overview

At a high level, the miner:
1. Receives a mining job from a node via `POST /mine`.
2. Validates the request and initializes a `MiningJob`.
3. Partitions the requested nonce search range across N CPU threads.
4. Each thread linearly scans its assigned nonce range, evaluating each candidate via `qpow_math::is_valid_nonce`.
5. The first valid nonce (or best result by distance) triggers an early stop via a shared cancel flag.
6. The miner aggregates per-thread progress and reports status and results via `GET /result/{job_id}`.
7. A job can be aborted at any time by `POST /cancel/{job_id}`.
8. Finished or stale jobs are cleaned up after a short retention period.

The implementation uses:
- `U512` for the 512-bit nonce space.
- `crossbeam_channel` for intra-job result/progress reporting.
- `std::thread` for CPU-bound work distribution across cores.
- `tokio` for asynchronous HTTP handling and the mining state loop.
- `qpow_math::{is_valid_nonce, get_nonce_distance}` for QPoW validity and scoring.

## Core Data Types

- `MiningState`
  - Holds all active jobs in `jobs: Arc<Mutex<HashMap<String, MiningJob>>>`.
  - Tracks `num_cores` used for CPU mining.
  - Periodically polls and advances job state in a background loop (`start_mining_loop`).

- `MiningJob`
  - Inputs: `header_hash: [u8; 32]`, `distance_threshold: U512`, `nonce_start: U512`, `nonce_end: U512`.
  - State: `status`, `start_time`, `total_hash_count`, `best_result`, `cancel_flag`, `result_receiver`, `thread_handles`, `completed_threads`.
  - Behavior:
    - `start_mining(num_cores)`: spawns worker threads and sets up a results channel.
    - `cancel()`: sets cancel flag and joins all worker threads.
    - `update_from_results()`: consumes worker messages, aggregates progress, picks the best result, and advances `status`.

- `ThreadResult`
  - Per-thread message with `hash_count` delta, optional `MiningJobResult`, and `completed` flag.

- `MiningJobResult`
  - The best candidate found so far: `nonce: U512`, `work: [u8; 64]`, `distance: U512`.
  - Note: `work` is the 64-byte big-endian representation of the winning nonce.

- `QPoWSeal`
  - Encodes the sealing payload returned to the node. In this crate, it includes `nonce: [u8; 64]` (the `work` value).

## Inputs and Validation

The miner accepts jobs via `POST /mine` with fields defined by `resonance_miner_api::MiningRequest`. In code, the following fields are validated:
- `job_id`: non-empty string (UUID recommended).
- `mining_hash`: 64-char hex string (32 bytes).
- `distance_threshold`: decimal string convertible to `U512`.
- `nonce_start`, `nonce_end`: 128-char hex strings (each 64 bytes), with `nonce_start <= nonce_end`.

If validation fails, the miner returns a 400 with an error message. If a duplicate `job_id` is submitted, it returns a 409.

Note: The YAML protocol uses `difficulty` terminology, while the implementation consumes `distance_threshold` (decimal string). In QPoW terms here:
- `distance_threshold` is the acceptance threshold for `is_valid_nonce`.
- `get_nonce_distance` returns a score (smaller is better) used to select the best candidate when multiple valid nonces are observed near-simultaneously.

## Work Distribution Across CPU Threads

A job’s nonce space is partitioned evenly across `num_cores` worker threads:

- Compute `total_range = (nonce_end - nonce_start) + 1`.
- Compute `range_per_core = total_range / num_cores`.
- Compute `remainder = total_range % num_cores`.
- Assign each thread a contiguous sub-range:
  - Thread `i` gets `[nonce_start + i * range_per_core, (start + range_per_core - 1)]`.
  - The final thread receives the `remainder` to ensure full coverage.
  - Ranges are clamped not to exceed `nonce_end`.

This partitioning ensures:
- Near-equal work per core.
- No overlaps between thread ranges.
- Full coverage of the requested inclusive interval.

## The Inner Mining Loop

Each thread runs `mine_range(...)`:
- Iterate `current_nonce` from `start` to `end` inclusive.
- For each nonce:
  - Convert `current_nonce` to a 64-byte big-endian array (`U512::to_big_endian()`), referred to as `nonce_bytes`.
  - Increment a local `hash_count` (number of candidates checked).
  - Call `qpow_math::is_valid_nonce(header_hash, nonce_bytes, distance_threshold)`.
    - If valid:
      - Compute `distance = qpow_math::get_nonce_distance(header_hash, nonce_bytes)`.
      - Send a `ThreadResult` carrying a `MiningJobResult { nonce, work: nonce_bytes, distance }` through the `crossbeam_channel` to the coordinator.
      - Reset the thread-local `hash_count` to 0 (since it was emitted with the result).
  - Periodically check the shared `cancel_flag`. If set, exit early.
    - The implementation checks the flag each loop iteration, but additionally includes a cutoff check every 4096 iterations to break sooner once cancellation is signaled under high-throughput conditions.

At thread termination (either completing its range or due to cancellation), the thread emits a final `ThreadResult` with `completed = true` and any remaining `hash_count` delta.

## Coordinating Results and Early Termination

`MiningJob::update_from_results()` is called by the mining state loop to coordinate progress:

- Drain all available `ThreadResult` messages:
  - Accumulate `hash_count` into `total_hash_count`.
  - When `completed = true`, increment `completed_threads`.
  - If a `MiningJobResult` is present:
    - Compare against `best_result` by `distance` (smaller distance is better).
    - If it improves the best-so-far:
      - Update `best_result`.
      - Set `cancel_flag = true` to stop other threads promptly (first-best-wins policy).
- Update the job status:
  - If any `best_result` exists: `status = Completed`.
  - Else, if all threads are finished and none found a valid nonce: `status = Failed`.
  - Otherwise: `status` remains `Running`.

This pattern yields:
- Minimal time-to-first-solution via early cancellation.
- A consistent best-result selection when multiple threads race to produce solutions.

## Job Lifecycle

1. Submission (node → miner)
   - `POST /mine` → `handle_mine_request`
   - `validate_mining_request` ensures inputs are structurally valid.
   - `MiningState::add_job` inserts a `MiningJob`, calls `start_mining(num_cores)`, and acknowledges with `accepted`.

2. Execution (miner internal)
   - `start_mining` spawns worker threads and attaches a bounded `crossbeam_channel` sender/receiver pair.
   - Threads run `mine_range`, push `ThreadResult` messages (progress, solutions, completion signals).
   - `MiningState::start_mining_loop` runs a background task:
     - Periodically locks the jobs map and calls `update_from_results()` on each running job.
     - Logs job completion and aggregates metrics.
     - Retains finished jobs briefly for retrieval; cleans up stale entries after ~5 minutes.

3. Cancellation (node → miner)
   - `POST /cancel/{job_id}` → `handle_cancel_request`
   - `MiningState::cancel_job` sets `cancel_flag` and joins all threads.
   - The job status becomes `Cancelled`. The job remains visible for a short retention period.

4. Retrieval (node → miner)
   - `GET /result/{job_id}` → `handle_result_request`
   - Returns `status ∈ {running, completed, failed, cancelled}` plus:
     - `nonce` (hex `U512`) and `work` (`[u8; 64]` hex) when `completed`.
     - `hash_count` (total candidates checked) and `elapsed_time` (seconds) in all cases.
   - If `job_id` does not exist: returns `not_found`.

5. Cleanup (miner internal)
   - The mining loop periodically removes jobs that are not `Running` and are older than a short retention window to keep memory usage bounded.

## Concurrency and Memory Model

- Threads coordinate via:
  - `Arc<AtomicBool>` cancel flag (checked with relaxed ordering).
  - `crossbeam_channel` for non-blocking result and progress delivery.
- `MiningState.jobs` is `Arc<Mutex<HashMap<...>>>` (Tokio mutex) to serialize job map updates and queries from async contexts.
- Per-thread handles are joined during explicit cancellation or when a job is dropped/removed.
- The result channel is bounded (`num_cores * 2`) to avoid unbounded memory usage under bursty result conditions (extremely low thresholds).

## Performance Characteristics

- Linear scan with O(1) work per candidate (dominated by `is_valid_nonce` and `get_nonce_distance`).
- Load scales with `num_cores` (default: all logical CPUs via `num_cpus::get()`).
- Cost centers:
  - Big-endian conversion of `U512` to `[u8; 64]` per candidate.
  - The `is_valid_nonce` and distance computations in `qpow_math`.
  - Channel operations and atomic flag checks.
- Early-cancel strategy minimizes wasted work after a solution is found.
- Hash rate tracking is coarse-grained across threads (accumulated via periodic messages and final completion reports).

## Edge Cases and Guarantees

- Duplicate job IDs are rejected.
- Nonce ranges are inclusive; the miner never searches outside `[nonce_start, nonce_end]`.
- If no valid nonce is found in the range, the job ends with `Failed`.
- On `Cancelled`, threads stop as soon as they observe the flag; they still flush final `hash_count` so the last reported progress is accurate.
- The best result is the smallest `distance` observed; however, the miner typically cancels on the first winning candidate, so ties are rare in practice.
- Jobs are retained briefly after completion/cancellation to allow the node to fetch results; they are then cleaned up to bound resource use.

## API Mapping

- Submit job: `POST /mine` → `accepted` or `error`.
- Poll result: `GET /result/{job_id}` → `running`, `completed`, `failed`, `cancelled`, or `not_found`.
- Cancel job: `POST /cancel/{job_id}` → `cancelled` or `not_found`.

Returned fields:
- `nonce`: hex string representation of `U512` (no 0x).
- `work`: 128-hex-char representation of `[u8; 64]` (no 0x). In this miner, it is the big-endian form of the winning `nonce`.
- `hash_count`: total number of candidates tested for this job.
- `elapsed_time`: seconds since `start_time`.

## Configuration

- `--port` / `MINER_PORT` (default 9833)
- `--num-cores` / `MINER_CORES` (default: all available logical CPUs)

## Potential Future Improvements

- Adaptive partitioning and dynamic work stealing for uneven ranges or heterogeneous cores.
- SIMD or instruction-level optimizations inside `qpow_math` hot paths.
- Batch evaluation of multiple nonces per iteration to amortize overhead of conversions and checks.
- Optional GPU offloading with the same external protocol.
- Rate metrics per thread and moving averages to refine progress estimates.
- Structured telemetry for better observability under load.

---

In summary, the miner is a CPU-bound, multi-threaded exhaustive search over a 512-bit nonce range, guided by the QPoW validity predicate and a distance-based scoring. It is designed for low-latency solution discovery via early cancellation, with simple and robust job lifecycle management exposed through a small HTTP surface.
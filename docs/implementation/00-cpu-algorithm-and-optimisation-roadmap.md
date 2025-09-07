# CPU Mining Algorithm and Optimisation Roadmap

This document serves as the authoritative reference for:
- The CPU mining algorithm and miner job lifecycle (as originally implemented in the baseline miner).
- A staged roadmap for performance optimisation (CPU/GPU) and the test/metrics infrastructure supporting this refactor.

Context:
- The “CPU mining algorithm” section reflects the baseline implementation before any optimisation work.
- The “Optimisation roadmap” section captures the goals and scope for the refactor, guiding incremental improvements.

The implementation reflects the repository’s workspace structure:
- crates/miner-service: HTTP API, job orchestration, scheduling.
- crates/engine-cpu: CPU engines (baseline and fast) behind a trait.
- crates/pow-core: QPoW math core (compat API + precompute/fast-path helpers).
- crates/metrics: Prometheus metrics and optional HTTP exporter.

It preserves compatibility with the node-facing HTTP API defined by `resonance-miner-api`.

---

## Table of Contents

1. Overview
2. Core Data Types and Concepts
3. Job Lifecycle
4. Concurrency and Memory Model
5. Performance Characteristics
6. API Mapping (Node ↔ Miner)
7. CPU Optimisation Roadmap
   - Phase 1 (Precompute + Incremental)
   - Phase 2 (Montgomery Arithmetic)
   - Phase 3 (Micro-optimisations)
   - Phase 4 (Cleaner Math API Surface)
8. GPU Offload Roadmap
9. Testing, Verification, and Metrics
10. Rollout and Risk Management
11. Concrete Task Checklist
12. Open Questions
13. Expected Outcomes

---

## 1) Overview

At a high level, the miner:
1. Receives a mining job from a node via `POST /mine` (header hash, threshold, nonce range).
2. Validates the request and initializes a job.
3. Partitions the nonce range across N worker threads (or device workgroups).
4. Each worker linearly scans its assigned sub-range, evaluating nonces under QPoW rules.
5. The first acceptable solution triggers an early stop via a shared cancel flag.
6. The miner aggregates progress and reports results via `GET /result/{job_id}`.
7. Jobs can be cancelled via `POST /cancel/{job_id}`; stale jobs are periodically cleaned up.

Key dependencies:
- U512 for the 512-bit nonce space.
- crossbeam-channel for per-thread results/progress.
- std::thread for CPU engines (Tokio handles HTTP and the job loop).
- pow-core for QPoW math: validity and distance functions; precompute and fast path helpers.

---

## 2) Core Data Types and Concepts

- MiningService (crates/miner-service)
  - Owns the job registry (Arc<Mutex<HashMap<job_id, MiningJob>>>).
  - Configures the engine backend (CPU baseline/fast; GPU future).
  - Runs the mining loop to poll/update job status.

- MiningJob (crates/miner-service)
  - Inputs: `header_hash: [u8; 32]`, `distance_threshold: U512`, `nonce_start: U512`, `nonce_end: U512`.
  - State: `status`, `start_time`, `total_hash_count`, `best_result`, `cancel_flag`, `result_receiver`, `thread_handles`, per-thread stats.
  - Behavior:
    - `start_mining(engine, num_cores)`: partitions the range, spawns workers.
    - `cancel()`: flips cancel flag; joins threads; transitions to Cancelled.
    - `update_from_results()`: drains per-thread messages; aggregates counters; picks best result; advances status.

- MinerEngine trait (crates/engine-cpu)
  - Abstracts the compute backend (baseline or optimised CPU; GPU in future).
  - `prepare_context(header, threshold) -> JobContext`
  - `search_range(ctx, range, cancel) -> EngineStatus` (synchronous for now)

- JobContext (crates/pow-core)
  - Precomputed per-job constants: `(m, n)`, `target`, `threshold`, `header_int`.
  - Supports optimised incremental evaluation via:
    - `init_worker_y0(ctx, start_nonce) -> y0` (one-time modular exponentiation)
    - `step_mul(ctx, y) -> y * m mod n` (O(1) per nonce)
    - `distance_from_y(ctx, y)` and `distance_for_nonce(ctx, nonce)`

- ThreadResult (crates/miner-service)
  - Per-thread message with fields:
    - `thread_id`, `hash_count` (delta), optional `result`, and `completed` flag.

- MiningJobResult (crates/miner-service)
  - The winning candidate: `nonce: U512`, `work: [u8; 64]` (big-endian nonce), `distance: U512`.

- EngineStatus (crates/engine-cpu)
  - `Found { candidate, hash_count } | Exhausted { hash_count } | Cancelled { hash_count }`
  - Carries hash_count for accurate metrics, especially for the winning thread.

- QPoWSeal (concept)
  - Encodes the sealing payload returned to the node; includes the winning `nonce` bytes.

---

## 3) Job Lifecycle

1) Submission (node → miner)
- `POST /mine` → validation → construct `MiningJob`.
- `MiningService::add_job` inserts the job, prepares context, starts workers.

2) Execution (internal)
- `start_mining`:
  - Computes total range and partitions into contiguous, inclusive sub-ranges.
  - Spawns threads that call `mine_range_with_engine`, which delegates to the selected `engine.search_range`.
- Worker threads:
  - Emit `ThreadResult` on completion (with `hash_count`) and with `result` if a solution is found.

3) Orchestration
- The mining loop periodically:
  - Drains thread messages, aggregates `hash_count`, selects the best result by smallest `distance`.
  - Sets `status`:
    - Completed if a solution exists.
    - Failed if all threads exhausted with no solution.
    - Stays Running otherwise.
  - Cleans up stale jobs after a retention period (e.g., 5 minutes).

4) Cancellation (node → miner)
- `POST /cancel/{job_id}` → flip cancel flag → join threads → `status = Cancelled`.

5) Retrieval (node → miner)
- `GET /result/{job_id}` returns:
  - status ∈ {running, completed, failed, cancelled}
  - `nonce` (hex U512) and `work` ([u8; 64] hex) when completed
  - `hash_count` total and `elapsed_time` in all cases

6) Cleanup
- Jobs not Running and older than retention are removed to bound memory use.

---

## 4) Concurrency and Memory Model

- Shared state:
  - Jobs map guarded by a Tokio Mutex.
  - Per-job cancel flag (Arc<AtomicBool>, relaxed ordering).
- Communication:
  - crossbeam-channel for per-thread results and completion statuses.
- Threads:
  - Joined on cancellation or when a job is explicitly removed.
- Bounded channels:
  - Sized to `num_cores * 2` to avoid unbounded memory during bursts.

---

## 5) Performance Characteristics

- Baseline engine:
  - Per-nonce modular exponentiation dominates CPU time.
- Fast engine:
  - Replaces per-nonce exponentiation with:
    - One initial exponentiation per worker (`y0 = m^(h + start_nonce)`),
    - Then O(1) modular multiplication per nonce (`y = y * m mod n`).
  - Computes `nonce_element = SHA3_512(y)` and `distance = target XOR nonce_element`.
  - Preserves exact correctness with dramatically reduced per-nonce cost.
- Hot spots:
  - Modular arithmetic (multiplication and exponentiation).
  - SHA3/Keccak per 64-byte input.
  - Big-endian conversion and loop overhead.
- Factors:
  - Work scales with `num_cores`.
  - Early cancel reduces waste after the first solution.

---

## 6) API Mapping (Node ↔ Miner)

- `POST /mine` → `accepted | error`
- `GET /result/{job_id}` → `running | completed | failed | cancelled | not_found`
- `POST /cancel/{job_id}` → `cancelled | not_found`

Returned fields (when applicable):
- `nonce`: U512 hex (no 0x).
- `work`: 128-hex chars (nonce bytes).
- `hash_count`: total nonces tested for the job.
- `elapsed_time`: seconds.

---

## 7) CPU Optimisation Roadmap

Goal: reduce per-nonce cost from O(log exponent) to O(1) and then improve constants.

Phase 1: Precompute + Incremental Evaluation
- Compute `(m, n)` and `target` once per job (CPU).
- For each worker/thread, compute `y0 = m^(h + start_nonce) mod n` once (sliding-window exponentiation).
- Replace per-nonce `mod_pow` with `y = (y * m) mod n`.
- Compute `nonce_element = sha3_512(y)` and `distance = target XOR nonce_element`; compare with `threshold`.
- Benefits:
  - Identical correctness with massive speedup (orders of magnitude).
- Status:
  - Architected and implemented via `pow-core::JobContext`, `init_worker_y0`, `step_mul`, `distance_from_y`.
  - Exposed through `engine-cpu::FastCpuEngine`.

Phase 2: Fast Modular Arithmetic (Montgomery)
- Rationale: per-nonce modular multiplication is now the hot path; fixed `n` per job enables Montgomery form.
- Tasks:
  - Implement 512-bit Montgomery multiplication (e.g., 8×64-bit limbs) or swap to a fixed-limb bigint library (crypto-bigint/fiat).
  - Precompute Montgomery params: `R`, `R^2 mod n`, `n'`.
  - Keep `y` and `m` in Montgomery domain; reduce for SHA3 if required by consensus.
- Expected:
  - >2× speedup over Phase 1 (hardware-dependent).
- Status:
  - Planned.

Phase 3: Micro-optimisations
- Use fast SHA3/Keccak implementation with SIMD for 64-byte input.
- Loop unrolling; batch a few iterations to reduce branch overhead.
- Batch hash_count updates; reduce crossbeam traffic.
- Thread affinity and NUMA awareness on multi-socket systems.
- Build flags (release/LTO/CGU/panic) and tuning.
- Status:
  - Planned (some flags applied at workspace level).

Phase 4: Cleaner Math API Surface (Optional but Recommended)
- `pow-core`:
  - Keep a `compat` API mirroring the original.
  - Provide fast-path APIs as first-class.
  - Potentially route legacy functions to the fast path internally (with care).
- Status:
  - Partially implemented (compat + fast-path APIs exist; compatibility preserved).

---

## 8) GPU Offload Roadmap

Objective: Offload the per-nonce inner loop to GPU for massive parallelism; keep orchestration and job setup on CPU.

CPU-side per job:
- `get_random_rsa(header)` and Miller–Rabin loop to select `n` (done once).
- Precompute `target`, `threshold`, and (if used) Montgomery parameters.
- Orchestrate ranges and collect results; maintain HTTP API.

GPU kernel per worker:
- Option A: compute `y0 = m^(h + start_nonce)` in the kernel; Option B: provide precomputed `y0`.
- For each nonce:
  - `y = montgomery_mul(y, m, n, n')`
  - `nonce_element = sha3_512(y)` (convert out of Montgomery domain if necessary)
  - `distance = target XOR nonce_element`; compare to `threshold`
  - If valid, set a global flag and report solution; early-exit other threads.

Implementation plan:
- G1 (Prototype):
  - Choose backend (CUDA via cust/rustacuda or OpenCL via ocl).
  - Implement 512-bit Montgomery mul/reduction; integrate a fast Keccak.
  - Kernel inputs: `m`, `n`, Montgomery params, `target`, `threshold`, per-thread `start_nonce`, `range_len`.
  - Host integration: coarse-grained range partitioning, device transfers, early-cancel.
- G2 (Optimise):
  - Occupancy tuning, register pressure, block size.
  - Batch multiple steps per thread; use constant memory for shared parameters.
  - Minimise global memory traffic; efficient found-flag handling.
- G3 (Integrate/Fallback):
  - Feature flags; auto-detect; clean cancellation semantics; multi-GPU support.
  - Unified JobContext across CPU/GPU paths.

---

## 9) Testing, Verification, and Metrics

Correctness
- Golden tests: fixed header/threshold and small ranges; ensure baseline and fast engines yield identical winners/failures and distances.
- Randomised: compare across many random jobs for small ranges.
- GPU (future): cross-check GPU vs CPU fast path.

Benchmarks
- Measure nonces/sec per engine; time-to-first-solution across thresholds.
- Resource utilisation (CPU%, memory, GPU occupancy).

Metrics (Prometheus)
- Global:
  - `miner_jobs_total{status}`
  - `miner_hashes_total`
  - `miner_hash_rate`
- Engine/job-aware:
  - `miner_job_hashes_total{engine,job_id}`
  - `miner_job_hash_rate{engine,job_id}`
  - `miner_job_status{engine,job_id,status}` (IntGauge)
  - `miner_jobs_by_engine_total{engine,status}`
- Engine/job/thread-aware:
  - `miner_thread_hashes_total{engine,job_id,thread_id}`
  - `miner_thread_hash_rate{engine,job_id,thread_id}` (EMA smoothed via service)
- Exporter: `/metrics` is enabled only if a metrics port is configured; otherwise metrics are disabled.

Dashboards (Grafana)
- Operator-focused (“Miner Overview”).
- Engine-focused (“Engines Comparison”).

---

## 10) Rollout and Risk Management

Staged Release
- Keep HTTP API unchanged; engines behind a trait.
- Gate new backends behind flags; keep a baseline engine for reference.
- Ship the CPU fast path first; validate; then iterate with Montgomery and/or GPU.

Risk Mitigations
- Big-int correctness:
  - Property tests; cross-check with `num-bigint` reference.
- Montgomery domain boundaries:
  - Clear conversions; isolated code; unit tests.
- SHA3/Keccak differences:
  - Use vetted implementations; vector test suites.
- Early cancel races:
  - Atomic flags; careful polling; stress tests.

---

## 11) Concrete Task Checklist

CPU — Phase 1 (High impact, low complexity)
- [ ] Add per-job precompute: `(m, n)`, `target`, `threshold`.
- [ ] Implement `init_worker_y0(header, m, n, start_nonce)` (sliding-window exponentiation).
- [ ] Replace per-nonce `mod_pow` with `y = (y * m) mod n`.
- [ ] Compute `nonce_element = sha3_512(y)`; `distance = target XOR nonce_element`.
- [ ] Keep miner API identical; validate correctness and benchmark.

CPU — Phase 2 (Montgomery)
- [ ] Introduce Montgomery params: `R`, `R^2 mod n`, `n'`.
- [ ] Implement 512-bit Montgomery mul/reduction.
- [ ] Keep `y` and `m` in Montgomery domain; convert for SHA3 if required.
- [ ] Benchmark vs Phase 1; ensure correctness.

CPU — Phase 3 (Micro-tuning)
- [ ] Faster SHA3 path (SIMD).
- [ ] Loop unrolling and batching.
- [ ] Thread affinity and NUMA tuning.
- [ ] Build flags: LTO, opt-level, codegen-units, panic strategy.
- [ ] Measure and document gains.

Math API Surface
- [ ] Add/solidify `JobContext` and helpers for precompute, init, step, distance.
- [ ] Keep legacy API functional while enabling fast paths.

GPU — Phases G1/G2/G3
- [ ] Choose CUDA/OpenCL; scaffolding.
- [ ] Implement Montgomery mul; integrate SHA3 in kernel.
- [ ] Kernel for inner loop with early-exit signal.
- [ ] Host integration: ranges, transfers, cancellation.
- [ ] Occupancy/memory tuning; multi-GPU; feature flags.
- [ ] Cross-validate vs CPU fast path; benchmarks.

Metrics and Testing
- [ ] Deterministic golden tests.
- [ ] Randomised cross-checks.
- [ ] Benchmarks and telemetry (dashboards).

Note: Some Phase 1 items may already be implemented depending on the current branch state (e.g., `pow-core::JobContext`, `init_worker_y0`, `step_mul`, and `engine-cpu::FastCpuEngine` exist). Treat this checklist as a living plan; tick off items as verified by tests/benchmarks.

---

## 12) Open Questions

- qpow-math parity:
  - Confirm any semantic differences vs upstream (e.g., return shapes, logging).
- SHA3 input domain:
  - Ensure consensus requires SHA3 of the standard (non-Montgomery) result of `m^(h + nonce) mod n`; apply correct domain conversion if Montgomery is used.
- Threshold semantics:
  - Confirm endian/formatting alignment for `distance` vs `threshold` across engines.

---

## 13) Expected Outcomes

- CPU fast path: large speedup by eliminating per-nonce exponentiation.
- Montgomery arithmetic: further significant gains on CPU.
- GPU offload: orders-of-magnitude throughput improvements where available.
- Robust telemetry: engine/job/thread-level metrics for accurate comparisons and operational insight.
- Safe rollout: stable HTTP API and staged engine evolution behind a clean abstraction.

---
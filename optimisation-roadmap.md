# Quantus External Miner — Optimisation Plan and GPU Offload Roadmap

Context: This miner searches for a valid nonce (U512) for a given 32-byte header hash using QPoW rules defined in `qpow-math`. The current implementation linearly scans nonce ranges across CPU threads and calls `is_valid_nonce` per candidate, which internally performs expensive big-integer operations.

This document lays out a methodical, staged plan to:
- Greatly accelerate the CPU miner (cheap, immediate wins).
- Prepare APIs and structure to enable a high-performance GPU backend.
- Define verification, metrics, and rollout steps to keep changes safe.

Scope validated from code:
- Miner: `quantus-external-miner/src/lib.rs` and `src/main.rs`
- Math: `qpow-math/src/lib.rs` (modular arithmetic, SHA3, Miller–Rabin, etc.)

Note: The miner currently appears to call `qpow_math::is_valid_nonce(header, nonce, threshold)` as a boolean predicate. In the qpow-math version inspected, `is_valid_nonce` returns `(bool, U512)` (bool, distance). We’ll need to align the exact version and/or add new helper APIs to enable the fast path described below.

-------------------------------------------------------------------------------

## 1) Key Observations and Bottlenecks

Per nonce, `get_nonce_distance(header, nonce)` does:
- Compute `(m, n) = get_random_rsa(header)` — constant per job.
- `target = hash_to_group_bigint_sha(header, m, n, 0)` — constant per job.
- `nonce_element = hash_to_group_bigint_sha(header, m, n, nonce)` =>
  - `hash_to_group_bigint`: `mod_pow(m, h + nonce, n)` (BigUint square-and-multiply)
  - `sha3_512` of the result
- `distance = target XOR nonce_element`.

Core bottleneck:
- The modular exponentiation `m^(h + nonce) mod n` dominates cost per nonce. However, for sequential nonces, exponents differ by +1, enabling an O(1) incremental step:
  - If `y_k = m^(h + nonce_k) mod n`, then `y_{k+1} = y_k * m mod n`.
  - This replaces a full exponentiation per nonce with a single modular multiplication per nonce after one initial exponentiation per thread.

Other observations:
- `m`, `n`, `target` are invariant per job.
- Initial `y0 = m^(h + start_nonce)` must be computed once per worker (either CPU or GPU).
- SHA3-512 on a fixed 64-byte input per nonce is not free but much cheaper than exponentiation.
- Modular multiplication can be further accelerated using Montgomery arithmetic with a fixed modulus `n`.

-------------------------------------------------------------------------------

## 2) CPU Optimisation Roadmap (Phased)

Goal: Reduce per-nonce work from O(log exponent) to O(1) and improve constants.

Phase 1: Precompute job constants and incremental evaluation
- Tasks:
  - Compute `(m, n)` and `target` once per job (on CPU).
  - For each worker thread, compute `y0 = m^(h + start_nonce) mod n` once using an optimized exponentiation (sliding window).
  - Replace `mod_pow` per nonce with single modular multiplication: `y = (y * m) mod n`.
  - Compute `nonce_element = sha3_512(y)` then `distance = target XOR nonce_element`; compare to `threshold`.
- Acceptance:
  - Identical correctness to baseline (same winning nonce for same job/range/threshold).
  - Measured hash rate (nonces/sec) improves by at least 10x–100x vs baseline on a typical CPU.

Phase 2: Fast modular arithmetic (Montgomery multiplication)
- Rationale: Modular multiplication is now the hot path. Fixed modulus `n` per job allows Montgomery form.
- Tasks:
  - Switch big-integer backend for 512-bit ops to a fixed-width limb implementation (e.g., `crypto-bigint`) or implement custom 8×64-bit Montgomery mul/reduction.
  - Precompute Montgomery parameters once per job:
    - `R = 2^(word_size*limbs) mod n`, `R^2 mod n`, and `n'` (Montgomery inverse).
  - Keep `y`, `m` in Montgomery domain:
    - Initial transforms: `m_hat = m * R^2 mod n`, `y0_hat = mod_pow(m, h + start_nonce, n) in Montgomery` or transform after standard pow.
    - Per nonce: `y_hat = montgomery_mul(y_hat, m_hat, n, n')`.
    - Before SHA3, either leave in normal domain (one extra Montgomery reduction per iteration) or compute SHA3 over the Montgomery representation if the consensus math requires the normal domain only (likely the latter).
- Acceptance:
  - >2x speedup over Phase 1 on the same hardware (target depends on implementation).
  - No correctness regressions.

Phase 3: Micro-optimisations and systems tuning
- Tasks:
  - Use a SHA3/Keccak implementation with SIMD acceleration and optimized 64-byte input path (e.g., `tiny-keccak` with SIMD features).
  - Unroll inner loop (process 2–4 nonces per iteration) to reduce branch overhead.
  - Batch `hash_count` updates; send fewer messages over `crossbeam_channel`.
  - Adjust cancellation checks to a tuned interval that balances latency and throughput (current 4096 iterations is a start).
  - CPU pinning (set thread affinity per worker), consider NUMA locality when scaling across sockets.
  - Ensure release build, LTO, `opt-level = 3`, `codegen-units = 1`, `panic = abort` for maximum performance.
- Acceptance:
  - Additional measurable improvement (10–30%).
  - Stable latency to cancel after solution found.

Phase 4: Cleaner qpow-math API surface (optional but recommended)
- Tasks:
  - Introduce a `JobContext` in qpow-math:
    - Fields: `m, n, target, threshold`, Montgomery params if enabled.
  - Provide helpers:
    - `init_worker(context, start_nonce) -> y0` (fast exponentiation)
    - `step_once(y) -> y * m mod n` (Montgomery mul if enabled)
    - `distance(context, y) -> U512`
    - `is_valid_distance(distance, threshold) -> bool`
  - Keep current API for compatibility; route it through fast path internally where possible.
- Acceptance:
  - Miner uses new APIs and is simpler/readable.
  - No public protocol changes.

-------------------------------------------------------------------------------

## 3) GPU Offload Roadmap

Objective: Offload the per-nonce inner loop to GPU for massive parallelism. The CPU keeps orchestration, job setup, and RSA/prime work.

What to keep on CPU (per job):
- `get_random_rsa(header)` including Miller–Rabin loop to find a suitable `n` (prime checks are non-trivial on GPU; executed once per job on CPU is fine).
- Precompute `target`, `threshold`, and Montgomery parameters for `n`.
- Orchestrate job ranges, collect results, and serve HTTP.

What to offload to GPU (per thread/block):
- Initial worker state `y0 = m^(h + start_nonce) mod n` (either computed on GPU with windowed exp or provided by CPU).
- Inner loop across assigned nonce stride:
  - `y = montgomery_mul(y, m, n, n')`
  - `nonce_element = sha3_512(y)`  // normal domain required for consensus
  - `distance = target XOR nonce_element`
  - Compare to `threshold`
  - Signal solution via global atomic flag/queue.

Implementation plan (CUDA or OpenCL; wgpu possible but less mature for big-int perf):

Phase G1: Prototype GPU kernel
- Tasks:
  - Choose backend: CUDA (rustacuda/cust) or OpenCL (ocl) for portability.
  - Implement 512-bit Montgomery multiplication using 8×64-bit limbs with carry handling and reduction.
  - Integrate a known fast Keccak/SHA3 kernel for 64-byte input.
  - Implement a simple kernel:
    - Input buffers: `m`, `n`, Montgomery params, `target`, `threshold`, per-thread `start_nonce`, `range_len` (or stride).
    - Output: solution buffer/flag.
  - Host code:
    - Partition nonce ranges at a coarse granularity and launch kernels.
    - Poll or stream results; early-cancel on solution.
- Acceptance:
  - Correctness parity with CPU fast path on test vectors.
  - Measurable speedup over CPU on a mid-range GPU.

Phase G2: Optimise GPU throughput
- Tasks:
  - Tune occupancy: thread/block sizes, register pressure.
  - Batch multiple nonces per thread to amortize control overhead.
  - Reduce global memory traffic; keep per-thread state in registers/shared memory.
  - Use constant memory for `m`, `n`, Montgomery params, and `target`.
  - Implement an efficient global “found” flag with minimal warp divergence.
  - Consider multi-GPU support.
- Acceptance:
  - Achieve targeted kh/s or Mh/s based on hardware class.
  - Stable early-exit behavior with minimal wasted work after solution found.

Phase G3: Integration and fallback
- Tasks:
  - Feature gate GPU backend (e.g., `--backend=cpu|cuda|opencl|auto`).
  - Auto-detect GPU and fall back to CPU if unavailable.
  - Safe shutdown and cancellation semantics mirrored with CPU backend.
  - Share common `JobContext` and per-worker init code where possible.
- Acceptance:
  - Identical behavior and API surface to CPU backend.
  - Operationally safe with clear logs/metrics.

-------------------------------------------------------------------------------

## 4) Testing, Verification, and Metrics

Correctness and determinism
- Golden tests:
  - For fixed `header`, `threshold`, and small nonce ranges, verify both CPU baseline and fast path yield identical winners (or failure) and distances.
  - Repeat for edge thresholds (`0`, `U512::MAX`, borderline cases).
- Randomized tests:
  - Generate random jobs and verify CPU fast path vs original slow path on small ranges.
  - Cross-check GPU results against CPU fast path for representative samples.

Performance benchmarks
- Add benchmark harness:
  - `kh/s` (thousands of candidates per second) per thread and aggregate.
  - Time-to-first-solution for a set of synthetic jobs at varied thresholds (controlling success rate).
  - Resource usage stats (CPU %, memory, GPU occupancy if available).
- Metrics & telemetry:
  - Emit structured logs or Prometheus-style metrics for hash rate, solution latency, and cancel latency.
  - Per-backend metrics to compare CPU vs GPU.

-------------------------------------------------------------------------------

## 5) Rollout and Risk Management

Staged rollout
- Keep the current API stable and public protocol unchanged.
- Guard new paths behind flags (e.g., `--fast-cpu`, `--gpu-backend=cuda|opencl`).
- Ship CPU fast path first; validate in dev/test networks.
- Add GPU backend as opt-in; run shadow trials (compare GPU vs CPU results invisibly on small samples).

Risk areas and mitigations
- Big-int arithmetic bugs (carry/overflow):
  - Extensive unit tests and cross-validation with `num-bigint` reference.
- Montgomery domain misuse:
  - Clear domain conversions; isolate code paths; property tests.
- SHA3 kernel differences:
  - Use vetted implementations; verify against test vectors.
- Early-cancel races:
  - Use atomic flags; document memory model; fuzz cancellation at random intervals.

-------------------------------------------------------------------------------

## 6) Concrete Task Checklist

CPU — Phase 1 (High impact, low complexity)
- [ ] Add per-job precompute: `(m, n)`, `target`, `threshold`.
- [ ] Implement `init_worker_y0(header, m, n, start_nonce)` with sliding-window exponentiation.
- [ ] Replace per-nonce `mod_pow` with `y = (y * m) mod n`.
- [ ] Compute `nonce_element = sha3_512(y)`; `distance = target XOR nonce_element`.
- [ ] Keep miner API identical; validate correctness and benchmark.

CPU — Phase 2 (Montgomery)
- [ ] Introduce Montgomery parameters for `n`: `R`, `R^2 mod n`, `n'`.
- [ ] Implement 512-bit Montgomery mul/reduction (8×64-bit limbs).
- [ ] Keep `y` and `m` in Montgomery domain; convert for SHA3 as required.
- [ ] Benchmark vs Phase 1; ensure correctness.

CPU — Phase 3 (Micro-tuning)
- [ ] Faster SHA3 path (SIMD).
- [ ] Loop unrolling and batching.
- [ ] Thread affinity and NUMA tuning.
- [ ] Build flags: LTO, opt-level, codegen-units, panic strategy.
- [ ] Measure gains; document configs.

qpow-math API
- [ ] Add `JobContext` and helper functions for precompute, init, step, distance.
- [ ] Implement fast paths internally; keep legacy API functional.

GPU — Phase G1/G2/G3
- [ ] Choose CUDA/OpenCL; setup scaffolding.
- [ ] Implement Montgomery mul on GPU; integrate SHA3.
- [ ] Write kernel for inner loop with early-exit.
- [ ] Host integration: ranges, transfers, solution reporting, cancellation.
- [ ] Optimise occupancy and memory.
- [ ] Feature flags and fallback.
- [ ] Cross-validate with CPU; benchmark.

Metrics and Testing
- [ ] Add deterministic test suites (golden cases).
- [ ] Add randomized property tests.
- [ ] Add benchmark suite and telemetry endpoints/logs.

-------------------------------------------------------------------------------

## 7) Open Questions

- Exact qpow-math version alignment: does `is_valid_nonce` return `(bool, distance)` or only `bool` in the miner-pinned commit? We may need small refactors to expose distance cleanly.
- Domain of SHA3 input: Confirm consensus expects SHA3 of the standard (non-Montgomery) result of `m^(h + nonce) mod n`. If so, ensure correct domain conversion on CPU/GPU fast paths.
- Threshold semantics: Ensure CPU/GPU compare the same representation and endian order for `distance` vs `threshold`.

-------------------------------------------------------------------------------

## 8) Expected Outcomes

- CPU fast path should deliver a major speedup (order(s) of magnitude), by replacing exponentiation per nonce with a single modular multiplication.
- Montgomery arithmetic and micro-optimisations further improve throughput.
- GPU backend unlocks additional parallelism with the same external protocol and miner API.
- Instrumentation, tests, and staged rollout make improvements safe to adopt.

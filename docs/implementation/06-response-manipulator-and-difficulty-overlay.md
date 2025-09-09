# 06 — Response Summary: Manipulator engine (per‑block throttle), throttle controls & resume, and difficulty overlay

This iteration delivered a new throttling engine to help unwind difficulty overshoot, operational controls to tune and resume that throttle, and a dashboard panel that shows a single, robust difficulty line per chain.

---

## What we implemented

### 1) New CPU engine: `cpu-chain-manipulator` (per‑block throttling)

- High-performance start:
  - Uses the same incremental fast path as `cpu-fast`:
    - `init_worker_y0` (one-time mod pow at range start)
    - `step_mul` (O(1) modular multiply per nonce)
    - `distance_from_y` (distance using precomputed context)
  - This gives “cpu‑fast‑like” early hashrate to break stalls and begin producing blocks.

- Per‑block throttle (linear ramp with optional cap):
  - A `solved_blocks` counter increments only when a block is solved (FOUND).
  - After each batch of `step_batch` nonces (default 10,000), the engine sleeps:
    - `sleep_ns = base_delay_ns * solved_blocks`
    - Defaults: `base_delay_ns = 500_000` ns (0.5 ms), `step_batch = 10_000`, no cap.
  - Effect: the next block is slower than the previous, allowing chain difficulty to trend down.

- Engine selection:
  - `--engine cpu-chain-manipulator`

### 2) Throttle controls and “resume from last state”

- New CLI/env flags (all optional):
  - `--manip-solved-blocks <u64>` (`MINER_MANIP_SOLVED_BLOCKS`)
    - Initialize `solved_blocks` on startup (resume where we left off after a restart).
  - `--manip-base-delay-ns <u64>` (`MINER_MANIP_BASE_DELAY_NS`)
    - Base sleep per batch (ns) (default 500,000).
  - `--manip-step-batch <u64>` (`MINER_MANIP_STEP_BATCH`)
    - Nonce attempts between sleeps (default 10,000).
  - `--manip-throttle-cap <u64>` (`MINER_MANIP_THROTTLE_CAP`)
    - Cap the `solved_blocks` index so throttling doesn’t grow unbounded.

- Service wiring:
  - The service applies overrides when constructing the engine and, if provided, stores the starting throttle index into the engine.

### 3) Throttle state debug logs (for Graylog)

- At job start (context prepared):
  - `manipulator throttle start: solved_blocks=<N>, sleep_ns_per_batch=<S>, step_batch=<B>`

- On block found (increment):
  - `manipulator throttle increment: solved_blocks=<N>, (next sleep_ns_per_batch=<S>, cap=<cap>)`

- These appear when `RUST_LOG` includes `miner=debug`.

### 4) Difficulty overlay (Instances Comparison dashboard)

- New panel “Difficulty (by chain)” under “Global Hash Rate by Instance.”
- Robust PromQL for a single line per chain:
  ```
  quantile by (chain) (0.5, avg_over_time(qpow_metrics{data_group="difficulty"}[5m]))
  ```
  - Median across nodes for each chain, based on 5-minute averages → tolerates outliers (e.g., forked nodes).
  - Avoids axis conflicts by using its own panel.

---

## Operator guidance

- Recovery from difficulty overshoot:
  - Run `cpu-chain-manipulator` on one canary; start with low `--workers` (e.g., 1).
  - Let the engine produce a block at full speed, then progressively throttle per block.
  - If restarting, use `--manip-solved-blocks N` to resume throttle state (avoid restarting at full speed).
  - Tune with:
    - `--manip-base-delay-ns` (increase to throttle more per block),
    - `--manip-step-batch` (decrease to sleep more frequently),
    - `--manip-throttle-cap` (limit throttle at a chosen level).

- Dashboards:
  - Instances Comparison:
    - Hash rate (5m trends) should ramp down over blocks for the manipulator node.
    - Difficulty (by chain) should trend down over time (single robust line per chain).
    - Use “Active Jobs” and “Accepted Mine Requests” panels to correlate bursts with mining activity and job issuance.

---

## Validation

- Grafana shows:
  - `cpu-chain-manipulator` starts with high hashrate and then decreases per block as expected.
  - Difficulty panel renders exactly one robust series per chain, smoothing out ephemeral outliers.
- Graylog shows:
  - Throttle start and increment debug lines with the expected values.
- CLI resume works:
  - Launching with `--manip-solved-blocks <N>` picks up throttle from the specified block count.

---

## Notes and next steps

- Optional enhancements:
  - Exponential or hybrid throttle (e.g., `sleep_ns = base * 2^(blocks/k)`, with cap).
  - Persistent throttle state (e.g., write recent `solved_blocks` to a small local file).
  - “Hold” mode: keep throttle constant once block cadence is near target for K blocks.

- Prometheus/Grafana:
  - If desired, add a “Chain” dashboard variable to filter difficulty by chain (query: `label_values(qpow_metrics{data_group="difficulty"}, chain)`).
  - Current panel already aggregates per chain; filtering is not required for correct operation.

---

## Files touched (high level)

- Engine (new behavior and debug):
  - `crates/engine-cpu/src/lib.rs` (ChainManipulatorEngine with fast path + per-block throttle, logs, and tunables)

- Service & CLI (controls and resume):
  - `crates/miner-service/src/lib.rs` (engine selection, parameter wiring, resume)
  - `crates/miner-cli/src/main.rs` (new flags and env vars)

- Dashboard:
  - `docs/grafana/instances-comparison-dashboard.json` (Difficulty panel)

This completes the per‑block manipulator, operator controls, and difficulty visualization so you can safely and transparently drive difficulty back into a healthy range.
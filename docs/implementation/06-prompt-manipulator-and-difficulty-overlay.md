# 06 — Prompt Summary: Manipulator engine (per‑block throttle), throttle controls/restore, and difficulty overlay

This iteration focused on (a) adding a new CPU engine that can manipulate effective hashrate per block to help recover from difficulty overshoot, (b) giving operators control over that throttle and a way to resume it after restarts, and (c) making difficulty visible in Grafana as a single robust line per chain.

## Objectives

- Introduce a new CPU engine (“cpu‑chain‑manipulator”) that:
  - Starts fast (cpu‑fast‑like throughput) to resume stalled chains or seed initial blocks.
  - Then throttles down per block solved, so chain difficulty can trend down without stalling the network.
- Provide throttle controls (via CLI/env) and a “resume from previous throttle” mechanism so restarts don’t reset to full speed.
- Add explicit throttle state debug logs for operational visibility.
- Surface chain difficulty in Grafana as one robust line per chain, ignoring forked/outlier nodes.

## Prompts (condensed)

- We need an engine that begins at cpu‑fast performance and then deliberately slows with each solved block (“artificial bottleneck”) to reduce chain difficulty over time.
- Throttling should be per block (not per job), so the ramp correlates with block production.
- Add throttle logs so we can see current throttle state at job start and when it increments.
- Add CLI params so we can:
  - Resume throttle “where we left off” after a restart.
  - Tune base delay per batch, batch size, and cap the throttle.
- Add a small “Difficulty (by chain)” panel to Instances Comparison without axis conflicts; show a single difficulty line per chain and suppress outlier nodes temporarily on a fork.

## What we did

### 1) New engine: cpu‑chain‑manipulator (per‑block throttle)
- Behavior:
  - Uses the same incremental fast path as cpu‑fast (init_worker_y0 + step_mul + distance_from_y), so it starts with cpu‑fast‑like hashrate.
  - A solved‑block counter (job_index) increments on Found (block solved), not on job start.
  - Sleep per batch (step_batch nonces) scales linearly with job_index (per‑block throttle), with an optional cap.
  - Defaults:
    - base_delay_ns = 500,000 (0.5 ms) per batch
    - step_batch = 10,000 nonces
    - throttle_cap = None (unbounded)
- Selection:
  - CLI: `--engine cpu-chain-manipulator`
  - Engines wired through service enum/switch.

### 2) Throttle controls and resume
- CLI/Env:
  - `--manip-solved-blocks` (`MINER_MANIP_SOLVED_BLOCKS`): initialize solved‑block counter on startup (resume where left off).
  - `--manip-base-delay-ns` (`MINER_MANIP_BASE_DELAY_NS`): override base sleep per batch (ns).
  - `--manip-step-batch` (`MINER_MANIP_STEP_BATCH`): override batch size (nonces between sleeps).
  - `--manip-throttle-cap` (`MINER_MANIP_THROTTLE_CAP`): cap solved‑block throttle index.
- Service applies overrides when constructing the engine.

### 3) Throttle state debug logs
- At job start (prepare_context):
  - `manipulator throttle start: solved_blocks=<N>, sleep_ns_per_batch=<S>, step_batch=<B>`
- On block found:
  - `manipulator throttle increment: solved_blocks=<N>, (next sleep_ns_per_batch=<S>, cap=<cap>)`

### 4) Difficulty overlay (instances dashboard)
- New timeseries panel: “Difficulty (by chain),” separate from hashrate panels (no axis conflict).
- Query (PromQL):
  - `quantile by (chain) (0.5, avg_over_time(qpow_metrics{data_group="difficulty"}[5m]))`
  - Renders a single robust line per chain (median across instances, with 5m smoothing).
  - Outlier/forked nodes’ difficulty reports are suppressed by the median aggregation.

## Operator notes

- To recover from difficulty overshoot:
  - Start manipulator on one canary with modest workers (e.g., `--workers 1`) and let per‑block throttle reduce effective hashrate as blocks are found.
  - Tune delay/batch/cap with CLI flags if needed.
  - Use `--manip-solved-blocks N` after a restart to resume throttle state without starting at full speed again.
- Logs:
  - Enable miner=debug to see throttle start/increment lines; correlate with Grafana hashrate and difficulty lines.
- Dashboards (Instances Comparison):
  - Confirm difficulty trends down over time while per‑instance hashrate ramps down (5m windows).
  - Use Active Jobs and Accepted Requests panels to correlate bursts with mining activity and issuance.

## Validation

- Grafana now shows cpu‑chain‑manipulator starting with cpu‑fast‑like rates and then trending down.
- Difficulty panel shows a single, smoothed line per chain (resilient to node outliers).
- Restarting a manipulator with `--manip-solved-blocks` continues throttling from the last known state.

## Next

- (Optional) Add exponential throttle mode and/or persistent throttle state (e.g., write to a small state file).
- (Optional) Add a “hold” mode: maintain current throttle once block cadence stabilizes near target.
- (Optional) Add a chain variable to filter the difficulty panel if you’d like per‑chain focus.

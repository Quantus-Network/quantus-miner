# 04 — Response Summary: Metrics Freshness, Chunking Knob, Remove-on-End, and Dashboards

This document summarises what was implemented since iteration 03 to improve metric accuracy, responsiveness, and usability for engine/instance comparisons.

## Overview

We addressed four main areas:
1. Metrics freshness: emit periodic progress updates from workers so job/global hash rates update in near‑real time.
2. Accuracy and artifact reduction: switch to remove‑on‑end for rate series (instead of writing zeros) to avoid scrape-timing spikes.
3. Operator control: add a time‑based chunking knob to tune the heartbeat cadence without code changes.
4. Observability and UX: extend Prometheus metrics and update Grafana dashboards to show meaningful trends and to disambiguate “idle because no jobs” vs “actively mining”.

---

## Code Changes

### 1) Periodic progress updates (chunking)

- Mining worker now splits its assigned nonce range into time‑targeted “chunks.”
- After each chunk completes, the worker sends a `ThreadResult` with `hash_count` even if it didn’t find a solution.
- Default chunk size derived from a time target:
  - `derived_chunk ≈ max(5_000, est_ops_per_sec * progress_chunk_ms / 1000)`
  - Using a conservative `est_ops_per_sec = 100_000` as an initial constant.
- Effect: Prometheus/Grafana receive frequent progress updates so per‑job/global rates are responsive while work is ongoing.

### 2) Per-thread delta-rate with EMA smoothing, per‑job/global derived from threads

- On each `ThreadResult`, compute per‑thread instantaneous rate:
  - `instant_rate = hash_count / Δt`, then smooth:
  - `ema = alpha * instant_rate + (1 - alpha) * prev_ema` with `THREAD_RATE_EMA_ALPHA = 0.2`.
- Publish `miner_thread_hash_rate{engine,job_id,thread_id} = ema` (gauge).
- Per‑job rate = sum of thread EMAs → `miner_job_hash_rate{engine,job_id}` (gauge).
- Global rate = sum of per‑job rates across running jobs → `miner_hash_rate` (gauge).

### 3) Remove-on-end for rate series

- When a thread completes or a job transitions to Completed/Failed/Cancelled:
  - Remove the corresponding gauge series instead of writing zeros:
    - `remove_thread_hash_rate(engine, job_id, thread_id)`
    - `remove_job_hash_rate(engine, job_id)`
- Rationale: avoids scrape timing artifacts (spurious zeros) and yields cleaner trend rollups with range functions.

### 4) Time‑based chunking knob

- CLI/env:
  - `--progress-chunk-ms <ms>` (env: `MINER_PROGRESS_CHUNK_MS`)
  - Default (if omitted): 2000ms
- Use ~1000–2000ms at a 2s Prometheus scrape interval for smooth charts and low overhead.

### 5) Service-level metrics

New Prometheus series:
- `miner_active_jobs` (gauge): number of currently running jobs in the miner.
- `miner_mine_requests_total{result}` (counter): counts `/mine` requests by `result`:
  - `accepted`, `duplicate`, `invalid`, `error`.

Wiring:
- On job accept: `active_jobs + 1`, `mine_requests_total{result="accepted"} + 1`
- On job cancel/remove: `active_jobs - 1`
- On invalid request: `mine_requests_total{result="invalid"} + 1`
- On duplicate or other add-job error: `result="duplicate"` or `result="error"` accordingly.

---

## Dashboard Updates (Grafana JSONs in `docs/grafana/`)

We maintained a clean, value‑only label style (e.g., `a1.i.res.fm`, `cpu-baseline`, `job_id`, `thread_id`), and adopted short range window functions to show trends rather than purely instantaneous views.

### 1) Miner Overview — `miner-dashboard.json`

- Global Hash Rate (over time):
  - `sum(max_over_time(miner_job_hash_rate[1m]))`
- Jobs by Engine and Status:
  - `sum by (engine, status) (increase(miner_jobs_by_engine_total[1m]))`
- Thread Hash Rate:
  - `max_over_time(miner_thread_hash_rate[1m])`
- Total Hashes / Per-Job Hashes:
  - `increase(...[1m])`
- Job Status:
  - `max_over_time(miner_job_status[1m])`

### 2) Engines Comparison — `engines-comparison-dashboard.json`

- Hash Rate by Engine (trend):
  - `sum by (engine) (max_over_time(miner_job_hash_rate{engine=~"$engine"}[1m]))`
- Jobs by Engine and Status:
  - `sum by (engine, status) (increase(miner_jobs_by_engine_total{engine=~"$engine"}[1m]))`
- Per-Engine Total Hashes:
  - `sum by (engine) (increase(miner_job_hashes_total{engine=~"$engine"}[1m]))`
- Per-Job / Per-Thread (trend):
  - `max_over_time(miner_job_hash_rate{...}[1m])`
  - `max_over_time(miner_thread_hash_rate{...}[1m])`

### 3) Instances Comparison — `instances-comparison-dashboard.json`

- Added a `job` filter variable (multi‑select) alongside `instance` and `engine`, and applied `job=~"$job"` across queries to filter by scrape job (e.g., chain).
- Global Hash Rate by Instance (trend):
  - `sum by (instance) (max_over_time(miner_job_hash_rate{job=~"$job", instance=~"$instance"}[1m]))`
- Per‑Instance Job Hash Rate (trend):
  - `sum by (instance) (max_over_time(miner_job_hash_rate{job=~"$job", instance=~"$instance", engine=~"$engine"}[1m]))`
- Current vs 1h Peak (per instance) — converted to a table with delta:
  - Current (1m max):
    - `sum by (instance) (max_over_time(miner_job_hash_rate{job=~"$job", instance=~"$instance", engine=~"$engine"}[1m]))`
  - Peak 1h (1m resolution subquery):
    - `sum by (instance) (max_over_time(miner_job_hash_rate{job=~"$job", instance=~"$instance", engine=~"$engine"}[1h:1m]))`
  - Delta:
    - Calculated as `current / peak_1h` via a table transformation (`calculateField`).
  - Columns: `instance | current_ops | peak_1h_ops | delta_current_over_peak`
- 24h Peak Hash Rate (selected instances):
  - `sum by (instance) (max_over_time(miner_job_hash_rate{job=~"$job", instance=~"$instance"}[24h:1m]))`
- New panels to disambiguate idle vs active:
  - Active Jobs (by instance):
    - `sum by (instance) (miner_active_jobs{job=~"$job", instance=~"$instance"})`
  - Mine Requests (5m increase; by instance & result):
    - `sum by (instance, result) (increase(miner_mine_requests_total{job=~"$job", instance=~"$instance"}[5m]))`

---

## Prometheus Settings

- Keep global defaults (e.g., `scrape_interval: 15s`), but override miner jobs to 2s:
  - `scrape_interval: 2s`, `scrape_timeout: 1.5s`
- For short-lived performance experiments, consider 1s; for load-sensitive prod, 5s works with chunking still providing reasonable freshness.
- Align `--progress-chunk-ms` with your scrape interval (e.g., chunk ~1000–2000ms for a 2s scrape).

---

## Operational Results

- Job/global rates are now responsive while work is active and disappear cleanly after completion.
- Trend panels (using `max_over_time` / `increase`) provide a meaningful “recent performance” view for engine/instance comparisons.
- New service-level metrics (`miner_active_jobs`, `miner_mine_requests_total`) make it obvious when a miner is idle because no jobs are being issued (e.g., on a quiet chain).
- Label formatting is consistent and readable across dashboards.

---

## Next Actions (Optional Enhancements)

- Auto‑tune chunk size based on observed per‑thread throughput in the first chunks.
- Expose EMA alpha (smoothing) as a knob for advanced tuning.
- Add field overrides to render the delta column as a percentage with thresholds.
- Provide baseline alerts/recording rules, e.g.:
  - “No active_jobs for X minutes” (node not issuing /mine).
  - “Hash rate below expected baseline for Y minutes.”

---

## Quick Reference

- CLI:
  - `--engine cpu-baseline|cpu-fast`
  - `--progress-chunk-ms <ms>` (env: `MINER_PROGRESS_CHUNK_MS`)
  - `--metrics-port <port>`
  - `--cores <n>` (alias: `--num-cores`)
- New metrics:
  - `miner_active_jobs` (gauge)
  - `miner_mine_requests_total{result}` (counter)
- Rate series handling:
  - Per-thread and per-job rate series are removed on end, avoiding zero artifacts.
- Dashboards:
  - `miner-dashboard.json` — Operator overview
  - `engines-comparison-dashboard.json` — Engine comparability
  - `instances-comparison-dashboard.json` — Per-instance analysis with `job` filter

These changes yield accurate, timely, and comparative metrics for evaluating engine performance and instance health across your testnets and (eventually) production.
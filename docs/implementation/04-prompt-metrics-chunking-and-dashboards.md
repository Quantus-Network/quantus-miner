# 04 — Metrics Freshness, Chunking Knob, Remove-on-End, and Dashboards

This document summarizes the prompts and responses since the last update (03), focused on making the miner’s metrics accurate, timely, and useful in Grafana for engine comparability and per-instance analysis.

Scope covered:
- Fixing miner hash-rate metrics (job/global) so they reflect active work and show trend meaningfully
- Emitting periodic progress updates (chunking) and adding a time-based chunking knob
- Switching to remove-on-end for rate gauges to avoid scrape-timing artifacts
- Extending dashboards with cleaner labels, trend queries, 24h/1h peaks, current vs peak visuals, and a job filter
- Adding service-level metrics to distinguish “idle because no jobs” vs “actively mining”
- Prometheus scrape interval guidance

---

## 1) What we observed

- On resonance (busy), job/global hash rates appeared; on heisenberg (quiet), they often showed 0.
- miner_thread_hash_rate was present (thanks to per-thread EMA), but miner_job_hash_rate/miner_hash_rate often flatlined at 0.
- “Instances Comparison” dashboard had labels rendered as Prometheus label objects (e.g., `{instance="a1.i.res.fm"}`) and some panels were not informative.
- We needed trend-oriented panels for engine comparability, not just instantaneous gauges.

Root causes:
- Engines reported results only at the end of large ranges, so totals/elapsed were stale most of the time.
- Job/global rates were computed as totals/elapsed and reset aggressively on job completion, leading to zeros or absent data.
- Thread EMA rates weren’t cleared on completion — this made thread charts look alive even when jobs weren’t running.
- Some dashboards summed the “instant” miner_hash_rate instead of a trend-friendly aggregate.

---

## 2) Decisions

- Emit periodic sub-range progress from each worker (chunking) for real-time metrics.
- Compute per-thread delta-rate (hash_count / Δt) and smooth with EMA; derive per-job and global rates from the sum of thread EMAs.
- Switch from “write zero on end” to “remove-on-end” for rate series to avoid scrape-timing artifacts and produce clean rollups.
- Add a time-based knob for progress chunking: `--progress-chunk-ms` (`MINER_PROGRESS_CHUNK_MS`).
- Adopt trend-oriented queries in Grafana (max_over_time/increase) to show recent performance (current vs. 1h peak, etc.).
- Add service-level metrics:
  - `miner_active_jobs` (gauge)
  - `miner_mine_requests_total{result}` (counter: accepted, duplicate, invalid, error)
- Add a job filter to dashboards to select a specific scrape job per chain (e.g., heisenberg vs. resonance).

---

## 3) Implementation highlights

A) Periodic progress updates (chunking)
- The mining loop (`mine_range_with_engine`) now splits the assigned range into sub-ranges (“chunks”).
- After each chunk, a `ThreadResult` is emitted with `hash_count` even if no solution is found.
- Default chunk size is derived from `--progress-chunk-ms`:
  - `derived_chunk = max(5_000, est_ops_per_sec * progress_chunk_ms / 1000)`, with `est_ops_per_sec = 100_000` as a conservative starting point.
- This gives a steady cadence of per-thread updates and fresher job/global rates.

B) Rates derived from thread EMAs
- For each `ThreadResult`, compute instantaneous rate = `hash_count / Δt`, then smooth with EMA (`THREAD_RATE_EMA_ALPHA = 0.2`).
- Publish `miner_thread_hash_rate{engine,job_id,thread_id}` = EMA.
- Per-job rate = sum of thread EMAs per job → `miner_job_hash_rate{engine,job_id}`.
- Global rate = sum of per-job rates across running jobs → `miner_hash_rate`.

C) Remove-on-end semantics
- When a thread ends or a job transitions to Completed/Failed/Cancelled:
  - Remove thread rate series: `remove_thread_hash_rate(engine, job_id, thread_id)`.
  - Remove job rate series: `remove_job_hash_rate(engine, job_id)`.
- Rationale: this avoids scrape-race zero spikes and produces cleaner trend rollups.

D) Metrics added
- `miner_active_jobs` (gauge)
  - Increment on job accept, decrement on job cancel/remove. Reflects #running jobs “now”.
- `miner_mine_requests_total{result}` (counter)
  - Increment per /mine request outcome: `accepted`, `duplicate`, `invalid`, `error`.
- Effect: makes it obvious when a miner is idle because the node isn’t issuing /mine (active_jobs = 0, requests not increasing) vs actively mining.

E) CLI knob for chunking
- `--progress-chunk-ms` (env: `MINER_PROGRESS_CHUNK_MS`) sets target milliseconds for per-thread progress updates.
- Default: 2000 ms (if not provided).
- Use ~1000–2000 ms with `scrape_interval: 2s` for responsive charts.

---

## 4) Prometheus/Grafana updates

A) Scrape interval guidance
- Global: keep 15s.
- Miner jobs: override to 2s (or 1s for short experiments; 5s to reduce Prometheus load).
- Example per-job override:
  ```
  scrape_configs:
    - job_name: 'quantus-miner-resonance'
      scrape_interval: 2s
      scrape_timeout: 1.5s
      scheme: https
      metrics_path: /metrics/miner
      static_configs:
        - targets: [a1.i.res.fm, a2.i.res.fm, a3.i.res.fm]
    - job_name: 'quantus-miner-heisenberg'
      scrape_interval: 2s
      scrape_timeout: 1.5s
      scheme: https
      metrics_path: /metrics/miner
      static_configs:
        - targets: [a1.t.res.fm, a2.t.res.fm, a3.t.res.fm]
  ```

B) Dashboards
- Three JSONs under `docs/grafana/`:
  - `miner-dashboard.json` (operator overview)
  - `engines-comparison-dashboard.json` (baseline vs fast comparison)
  - `instances-comparison-dashboard.json` (per-instance analysis)
- Label display cleanup:
  - Legends use `{{engine}}, {{job_id}}, {{thread_id}}`, etc. (no `{label="value"}` surfacing).
- Trend-oriented queries:
  - Use `max_over_time(...[1m])` for rates and `increase(...[1m])` for counters so panels show recent performance rather than just instantaneous snapshots.
- Peak vs Current visuals:
  - Instances comparison now includes:
    - A 24h peak stat (computed via PromQL subqueries: `[24h:1m]`) for trend-friendly peak potential.
    - A per-instance table (current vs 1h peak + delta), with filters for job/instance/engine:
      - Current: `sum by (instance) (max_over_time(miner_job_hash_rate{...}[1m]))`
      - Peak 1h: `sum by (instance) (max_over_time(miner_job_hash_rate{...}[1h:1m]))`
      - Delta column: `current / peak_1h`
- New panels:
  - Active Jobs by instance: `miner_active_jobs`
  - Mine Requests 5m increase by instance & result: `increase(miner_mine_requests_total{...}[5m])`

---

## 5) Operational guidance

- Scrape:
  - Per-job: 2s scrape interval (1s for deep dives).
- Chunking:
  - Start with `--progress-chunk-ms 2000` (or 1000 for more frequent updates).
- Engines:
  - Compare `cpu-baseline` vs `cpu-fast` by setting different instances to different engines and using the Engines Comparison dashboard.
- Trend windows:
  - Dashboards use `[1m]` windows for charts to reduce “zero gaps” while preserving short-time responsiveness.
  - Peak windows can be adjusted (e.g., `[6h:1m]` or `[24h:1m]`) depending on what “recent performance” means for your use case.

---

## 6) Next steps

- Auto-tune chunk size:
  - Estimate per-thread throughput in the first few chunks and adapt `derived_chunk` dynamically rather than relying on the static `est_ops_per_sec`.
- Better per-thread trend smoothing:
  - Expose alpha as a CLI/env knob if operators want to tune EMA smoothing.
- Dashboard polish:
  - Add a percent unit & color thresholds for the delta column; optionally include sparkline mini-charts.
- Alerts/recording rules:
  - Record per-engine/per-instance hash rate aggregates and set basic alerts (e.g., “no jobs for X minutes” or “hash rate below expected baseline for Y minutes”).

---

## 7) Quick reference

- CLI:
  - `--engine cpu-baseline|cpu-fast`
  - `--progress-chunk-ms <ms>`
  - `--metrics-port <port>`
  - `--cores <n>` (alias: `--num-cores`)
- Service metrics added:
  - `miner_active_jobs` (gauge)
  - `miner_mine_requests_total{result}` (counter)
- Remove-on-end:
  - Thread/job rate gauges removed (series deleted) when completed/failed/cancelled.
- Grafana JSONs:
  - `docs/grafana/miner-dashboard.json`
  - `docs/grafana/engines-comparison-dashboard.json`
  - `docs/grafana/instances-comparison-dashboard.json`

These changes make the miner’s metrics fresher, more accurate, and easy to compare across engines and instances, while avoiding scrape artifacts and providing the context needed to explain “0 ops/s” situations on quiet networks.
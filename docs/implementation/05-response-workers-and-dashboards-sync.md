# 05 — Response Summary: Workers rename, cpuset‑aware defaults, dashboards re‑sync, active jobs fix, and cleanup

This document captures what we implemented since iteration 04, why we did it, and how operators should expect the system to behave.

## Overview

We aligned the codebase and dashboards with how the miner actually consumes CPU resources and emits metrics. We:
- Replaced the ambiguous “cores” naming with “workers” (logical CPU–bound threads)
- Defaulted worker count safely using cpusets (when present), leaving ~50% headroom by default
- Fixed the “active jobs” metric to always reflect ground truth
- Resynchronized the Engines and Instances dashboards with the current metric semantics (remove‑on‑end, chunked progress, trend windows)
- Cleaned up unused code in the metrics crate
- Added a light README note about the CLI change

These changes improve operator clarity, default safety, and visualization fidelity without changing public protocol behavior.

---

## What changed

### 1) “cores” → “workers” (logical CPU threads)
- CLI:
  - Introduced `--workers` (env: `MINER_WORKERS`)
  - Removed support for `--cores` and `--num-cores` (normal clap errors handle unknown flags)
- Service config and internals:
  - `ServiceConfig.num_cores` → `ServiceConfig.workers`
  - `MiningService.num_cores` → `MiningService.workers`
  - `start_mining(..., workers, ...)` and logs updated accordingly
- Rationale:
  - Accurately communicates that we map one worker to one logical CPU, not a physical core

### 2) Safe defaults via cpuset detection
- Worker default is derived from the effective CPU pool:
  - Prefer cgroup v2: `/sys/fs/cgroup/cpuset.cpus.effective`
  - Fallback to cgroup v1: `/sys/fs/cgroup/cpuset/cpuset.cpus`
  - Fallback to `num_cpus::get()` if cpuset not present
- Default `workers` chooses a value that leaves ~half the effective logical CPUs available to other processes (and never consumes 100%):
  - Intuition: avoid starving the node/OS by default
- Clamp user value to `[1, effective_cpus]`
- Logs clearly state detected effective CPUs and chosen worker count

### 3) “Active jobs” metric now set from ground truth each loop
- Previous event‑based increments could drift (e.g., duplicate `/mine`, rapid transitions)
- We now recompute `miner_active_jobs` on every mining loop tick:
  - Count jobs with `status == Running`
  - Set the gauge from that count
- Result: the metric is always accurate “right now,” and correlates with bursts on the dashboards

### 4) Dashboards re‑synced to metrics model and usage
- Instances Comparison
  - Added “Job(s)” filter and applied it to all queries
  - Switched trend windows from 1m to 5m in hash‑rate panels:
    - Retains recent bursts longer on quiet nets without inventing current activity
  - “Active Jobs (by instance, timeseries)” panel for visual correlation with bursts
  - “Accepted Mine Requests by Instance (10m increase)” timeseries to visualize issuance cadence
  - “Per‑Instance Hash Rate (current vs 1h peak)” converted to a table:
    - Current (1m max), Peak (1h max @1m resolution), and Delta (current/peak)
- Engines Comparison
  - Added “Job(s)” filter and applied across panels
  - Switched to 5m trend windows for engine hash‑rate panels; 1m for job counters
  - Fixed queries/legends and descriptions to match remove‑on‑end and chunked progress
- Legends everywhere use value‑only labels (e.g., `cpu-baseline`, `a1.t.res.fm`, `job_id`)—no `{label="value"}` objects

### 5) README update (light touch)
- Added a short note that `--workers` replaces `--num-cores` to better reflect intent
- Examples updated (`MINER_WORKERS`, `--workers`)
- Avoided editorializing; focused on clarity for maintainers

### 6) Metrics crate cleanup
- Removed unused `MetricsInfo` struct and the corresponding import
- Release build is clean

---

## Why these changes

- “Workers” removes conceptual ambiguity and aligns with what the miner actually does (spawn CPU‑bound threads mapped to logical CPUs).
- Cpuset‑aware defaults prevent the miner from consuming all CPU resources by default, which is especially important on dedicated hosts running both node and miner—and on modern distros where cpusets partition compute.
- The active jobs fix ensures the metric is dependable and correlates precisely with bursts of hash‑rate on dashboards.
- The dashboards now faithfully represent an accurate, “trend‑friendly” view under remove‑on‑end semantics and chunked progress: they retain recent bursts (5m windows) without faking activity between jobs.
- Cleanup removes dead code paths and warnings.

---

## Operator impact

- CLI
  - Use `--workers N` instead of `--num-cores` or `--cores`
  - `MINER_WORKERS` replaces `MINER_CORES`
- Defaults
  - If `--workers` is omitted, the miner detects the effective CPU pool and chooses a default that leaves ~half available to other processes (and never uses all)
- Dedicated systems (systemd)
  - Continue to pin the node and miner to disjoint CPU sets (e.g., `AllowedCPUs=` / `CPUAffinity=`), give the miner a sensible `--workers` ≤ its cpuset size
  - Optionally adjust `CPUWeight`/Nice so the node wins under contention
- Dashboards
  - Instances:
    - Bursts of work are more visible (5m windows) and clearly tied to “Active Jobs” and “Accepted Mine Requests”
    - The per‑instance table shows current vs 1h peak, plus delta, for “recent capacity” context
  - Engines:
    - Now apples‑to‑apples for `cpu-baseline` vs `cpu-fast` under the new model

---

## Validation

- Built in release mode without warnings
- Verified:
  - `--workers` respected and clamped to cpuset
  - Logs show effective logical CPUs and chosen worker count
  - `miner_active_jobs` rises during bursts, falls to zero between jobs
  - Instances/Engines dashboards render expected trend behavior with 5m windows
  - Metrics crate no longer warns about unused items

---

## Migration notes

- Replace uses of `--num-cores` with `--workers`
- Replace `MINER_CORES` with `MINER_WORKERS`
- If you rely on systemd cpusets, expect the miner to default to a conservative worker count (leaving ~50% headroom) unless you specify `--workers`
- Existing Prometheus/Grafana setups work as before; new panels/fixes require importing updated JSONs from `docs/grafana/`

---

## Next steps (optional)

- Auto‑tune chunk sizes based on initial observed per‑thread throughput
- Expose EMA alpha and chunk target as CLI knobs for advanced tuning
- Add a percent format + thresholds to the delta column in the per‑instance table
- Provide recording/alerting rules for:
  - “No active jobs” for N minutes
  - “Hash rate below expected baseline” for M minutes
- Detect the cgroup cpuset at runtime and use it to set the default `--workers` only if the user didn’t specify one (already implemented), and optionally log the actual cpus list for transparency

---

## Files touched (high‑level)

- CLI / service config:
  - `crates/miner-cli/src/main.rs` (workers flag, env var updates, logs)
  - `crates/miner-service/src/lib.rs` (workers rename, cpuset detection, defaults, active jobs fix)
- Metrics crate:
  - `crates/metrics/src/lib.rs` (remove `MetricsInfo`, import cleanup)
- Dashboards:
  - `docs/grafana/instances-comparison-dashboard.json`
  - `docs/grafana/engines-comparison-dashboard.json`
- Documentation:
  - `README.md` (light note on `--workers` replacing `--num-cores`)

These changes align terminology with implementation, make default behavior safe, and ensure dashboards present accurate, comprehensible trends for engine and instance comparisons.
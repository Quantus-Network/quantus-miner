# 05 — Prompt Summary: Workers rename, cpuset‑aware defaults, dashboards sync, active jobs fix, and cleanup

This document summarizes the prompts and decisions since iteration 04. The focus was to align naming and defaults with real behavior, improve default safety on shared/dedicated hosts, bring dashboards fully in sync with the new metrics model, and fix an inconsistency in the “active jobs” metric.

---

## Goals from the prompts

- Clarify that the miner’s CPU setting is about worker threads (logical CPUs), not physical cores, and avoid user confusion.
- Provide safer defaults so a miner does not starve other processes on a host by default (especially on shared/dedicated systems).
- Leverage systemd/cgroup cpusets when available.
- Bring the Engines/Instances dashboards fully up to date with the latest metric semantics (remove‑on‑end, chunked progress).
- Make it easy to correlate work bursts with job activity and issuance.
- Clean up any now‑unused artifacts in the metrics code.

---

## Key prompts and decisions

1) Rename “cores” to “workers”
- Replace the misleading “num_cores” terminology (and CLI flags) with “workers,” explicitly meaning worker threads (logical CPUs).
- Drop support for the legacy `--cores` and `--num-cores` flags; only support `--workers`.
- Keep README changes light: state that `--workers` replaces `--num-cores` to better reflect intent (no editorializing).

2) Default worker count from cpuset (and leave headroom)
- Detect the effective cpuset first (cgroup v2: `/sys/fs/cgroup/cpuset.cpus.effective`; v1 fallback).
- If no cpuset is present, fallback to `num_cpus::get()` (logical CPUs).
- Default `workers` so that the miner leaves roughly half of the effective logical CPUs available to the rest of the system (and never uses all).
- Clamp any user‑provided `--workers` value to the effective cpuset size (≥1, ≤effective).

3) Dashboards alignment (without “inventing” activity)
- Instances Comparison dashboard:
  - Use 5m trend windows (`max_over_time`/`increase`) for hash‑rate series to retain recent bursts longer on quiet networks.
  - Add job filter (Prometheus `job` label) and apply it to all queries.
  - Add timeseries for “Active Jobs (by instance)” and “Accepted Mine Requests (10m increase)” to correlate bursts with job activity and issuance cadence.
  - Convert the per‑instance “current vs peak” panel into a table with:
    - Current (1m max), Peak (1h max with 1m resolution), and Delta (current/peak) columns.
- Engines Comparison dashboard:
  - Add job filter and apply consistently.
  - Use 5m windows for engine hash‑rate panels, 1m for job counters.
  - Fix legends and textual descriptions; avoid label object forms in the UI.
- Keep exactness: dashboards use windows to show “recent performance,” but do not fake current activity.

4) “Active jobs” metric fix
- Previous event‑based increments were brittle. Now we set `miner_active_jobs` every mining loop by counting `JobStatus::Running` directly from the jobs map. This guarantees accurate current values.

5) Cleanup
- Remove the unused `MetricsInfo` struct and associated import from the metrics crate (warnings gone).
- Keep “value‑only” legend formats across dashboards (no `{label="value"}`).

---

## Outcomes

- Users now set miner capacity with `--workers` (logical CPUs), which is unambiguous and aligns with the implementation.
- The miner auto‑detects cpusets and picks a default number of workers that leaves headroom for the node/OS by default (safer on dedicated/shared hosts).
- Dashboards are synchronized with the metric model (remove‑on‑end, chunked progress, trend windows).
  - Instances: 5m windows, job filter, per‑instance current vs 1h peak table with delta, timeseries for active jobs and accepted requests.
  - Engines: 5m windows, job filter, corrected legends and queries.
- “Active Jobs” and “Accepted Mine Requests” panels make it clear whether idle graphs are due to no jobs issued or a lack of work.
- Metrics codebase is cleaned of unused artifacts.

---

## Operational guidance captured by the prompts

- On dedicated systems with systemd:
  - Use `AllowedCPUs=`/`CPUAffinity=` to partition CPUs between node and miner.
  - Set `--workers` to match (or slightly below) the miner’s cpuset size.
  - Optionally lower miner priority (Nice/CPUWeight) relative to the node.
- On quiet testnets:
  - Expect bursty activity—miners only hash during job windows. Use trend windows (e.g., 5m) to see recent performance without inventing current activity.
- For developer laptops:
  - Keep `--workers` modest (e.g., ≤ half of logical CPUs) or pin the miner away from UI CPUs.

---

## What changed (high‑level)

- CLI/Config:
  - `--workers` (env: `MINER_WORKERS`) replaces legacy `--num-cores`.
- Defaults:
  - cpuset‑aware detection of effective CPUs; choose workers to leave ~50% headroom by default.
- Metrics:
  - `miner_active_jobs` set from ground truth each loop; `miner_mine_requests_total` wired already (03/04).
  - Remove‑on‑end semantics for rate series retained (no zero artifacts).
- Dashboards:
  - Instances: 5m trend windows; job filter; active jobs/accepted requests timeseries; per‑instance current vs 1h peak table with delta.
  - Engines: 5m trend windows; job filter; consistent queries/legends.
- Cleanup:
  - Unused `MetricsInfo` removed; imports tidied.

---

## Remaining nice‑to‑haves (not required for this iteration)

- Auto‑tune chunk sizes based on initial observed thread throughput (instead of a constant estimate).
- Expose EMA alpha and progress chunk target via CLI for advanced tuning.
- Provide recording/alerting rules (e.g., no active jobs for N minutes, rate below baseline).
- Optional panel additions:
  - Delta column as a percentage with thresholds.
  - Per‑engine accepted requests timeseries.

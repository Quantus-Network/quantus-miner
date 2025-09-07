# Prompt Summary (03): Accurate Metrics for Engine Comparisons, Logging Defaults, CLI Flag Rename, and Dashboards

## Context

To compare the performance of different mining engines (e.g., `cpu-baseline` vs `cpu-fast`) in Grafana using Prometheus metrics, we need precise per-job and per-thread hash counts. Currently, when a worker thread finds a solution, the engine reports only the `Candidate` without the associated `hash_count`. This causes undercounting in totals/rates, especially for the winning thread.

Additionally, we want a better out-of-the-box operator experience and evaluation tooling:
- Default the logging level to `info` when `RUST_LOG` is not set.
- Rename the CLI flag `--num-cores` to `--cores` but keep the internal field `num_cores`, and preserve `--num-cores` as an alias for backward compatibility.
- Provide two Grafana dashboards:
  - A “Miner Overview” dashboard for operators, focusing on the fastest implementation.
  - An “Engines Comparison” dashboard for performance evaluations across implementations.

We want to extend the engine’s result reporting so that the final “Found” event carries the `hash_count` accumulated by that thread. The miner-service will consume this information to update metrics correctly (global, per-job, per-thread), enabling accurate comparison between engines.

We also want sane defaults for logging, a consistent CLI, and packaged dashboards to support evaluation and production usage.

## Objectives

- Extend the engine result type to carry per-thread `hash_count` when a solution is found.
- Update both CPU engines (`cpu-baseline` and `cpu-fast`) to return the `hash_count` on `Found`.
- Update the miner-service to:
  - Include `hash_count` from the `Found` status in totals and rate calculations.
  - Maintain engine-aware, per-job, and per-thread metrics parity.
- Preserve the node-facing HTTP API and service behavior.
- Keep engine selection and metrics toggle behavior unchanged.
- Default RUST_LOG to `info` if not set, so operators get useful logs without extra env configuration.
- Rename CLI flag to `--cores` (keep `--num-cores` as an alias) while keeping the code field as `num_cores`.
- Add two Grafana dashboards under `docs/grafana/`:
  - `miner-dashboard.json` for operators running the fastest engine.
  - `engines-comparison-dashboard.json` for evaluating implementations.

## Requirements

- Engine API:
  - Change `EngineStatus::Found(Candidate)` to carry `{ candidate: Candidate, hash_count: u64 }`.
  - Maintain the existing `Exhausted { hash_count }` and `Cancelled { hash_count }` patterns for consistency.
- Engines:
  - `BaselineCpuEngine` and `FastCpuEngine` must return accurate `hash_count` on `Found`.
- Service:
  - In `update_from_results`, when a `ThreadResult` is built from an engine’s `Found`, carry the returned `hash_count` and accumulate it in:
    - Global totals: `miner_hashes_total`
    - Per-job totals: `miner_job_hashes_total{engine,job_id}`
    - Per-thread totals: `miner_thread_hashes_total{engine,job_id,thread_id}`
  - Ensure per-thread delta-rate and EMA smoothing logic still runs using the `hash_count` returned by `Found`.
- Metrics:
  - No new metrics required; existing labeled counters/gauges should be used.
  - Accuracy improves because the winning thread’s final work is now captured.
- Logging:
  - If `RUST_LOG` is unset, default to `info` at startup.
- CLI:
  - Rename `--num-cores` to `--cores`, keeping `--num-cores` as an alias and `num_cores` as the field name.
- Dashboards:
  - Add `docs/grafana/miner-dashboard.json` (operator view).
  - Add `docs/grafana/engines-comparison-dashboard.json` (evaluation view).

## Constraints and Compatibility

- The public HTTP API (endpoints/payloads) must remain unchanged.
- The engine abstraction is internal to the repo, so the type change is acceptable and should be updated wherever consumed (service only).
- Maintain compatibility between `cpu-baseline` and `cpu-fast` engines for apples-to-apples comparison.
- Logging default must not override an explicitly set `RUST_LOG`.
- CLI change must preserve backward compatibility (`--num-cores` alias).
- Dashboards must be optional artifacts (no runtime impact).

## Acceptance Criteria

- `EngineStatus::Found` carries `{ candidate, hash_count }`.
- Both engines report the correct `hash_count` when returning `Found`.
- Miner-service uses this `hash_count` to update:
  - `miner_hashes_total`
  - `miner_job_hashes_total{engine,job_id}`
  - `miner_thread_hashes_total{engine,job_id,thread_id}`
  - Per-thread delta-rate → `miner_thread_hash_rate{engine,job_id,thread_id}` (EMA-smoothed).
- Build success in release mode.
- No changes to node-facing HTTP behavior.
- Grafana can now accurately compare `cpu-baseline` vs `cpu-fast` with improved fidelity for the winning thread.
- When `RUST_LOG` is unset, logs default to `info`.
- `--cores` is available (with `--num-cores` as an alias) and maps to `num_cores`.
- Two dashboards exist under `docs/grafana/` for operators and engine evaluation.

## Deliverables

- Engine API update:
  - Enum change for `EngineStatus::Found` to include `hash_count`.
- Engine implementations:
  - Baseline and fast CPU engines updated to return `hash_count` on `Found`.
- Service updates:
  - `mine_range_with_engine` and `update_from_results` modified to forward and accumulate the `hash_count` from `Found`.
  - Metrics update hooks remain correct and comprehensive (global, per-job, per-thread, per-thread EMA rate).
- CLI/logging updates:
  - Default `RUST_LOG` to `info` when unset.
  - Expose `--cores` flag (keep `--num-cores` alias) while retaining `num_cores` as the code field.
- Dashboards:
  - `docs/grafana/miner-dashboard.json` (operator-focused).
  - `docs/grafana/engines-comparison-dashboard.json` (engine evaluation).
- Documentation:
  - Capture this prompt and a separate response summary as iteration “03” in `docs/implementation/`.

## Out of Scope

- Additional metrics beyond those already implemented.
- Changes to the HTTP API or external protocol.
- Performance optimizations (e.g., Montgomery arithmetic) not directly related to the `hash_count` reporting tweak.
- Further CLI renames or logging customization (beyond defaults).
- Grafana provisioning/export automation.

## Notes

- This tweak corrects the main remaining accuracy gap for the winning thread in metrics.
- With this change, Grafana dashboards can make fair and accurate comparisons between different engines using the `engine` label, across global/job/thread-level telemetry.
- Default logging reduces operational friction; CLI alias retains backward compatibility during the transition.
- Packaged dashboards (overview and engines comparison) provide out-of-the-box visibility for operators and evaluators.

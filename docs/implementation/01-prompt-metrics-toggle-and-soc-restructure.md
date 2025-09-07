# Prompt Summary: Metrics Toggle and Separation-of-Concerns (SoC) Restructure

## Context
You requested guidance and implementation steps to:
- Restructure the external miner repository for better separation of concerns and long-term maintainability.
- Bring the functionality currently provided by `qpow-math` into this codebase so it can be optimized and refactored locally.
- Maintain compatibility with the existing HTTP API used by the node.
- Introduce a metrics/observability toggle to support Prometheus scraping, controllable by a `--metrics-port` parameter.

## Objectives
- Establish a clear architecture that decouples:
  - HTTP protocol handling and job orchestration.
  - The mining engine abstraction (CPU/GPU) behind a trait/interface.
  - The QPoW math core (local fork/superset of `qpow-math`) to enable aggressive optimization.
- Keep the node-facing API stable and unchanged.
- Provide a runtime toggle for metrics such that:
  - When `--metrics-port` is present, a Prometheus endpoint is exposed on that port.
  - When `--metrics-port` is absent, metrics are disabled entirely (no exporter, no overhead).

## Constraints and Compatibility
- No breaking changes to the node-visible HTTP API.
- All optimizations and refactors must be internal to the miner.
- The miner should be prepared for future GPU offloading without coupling protocol code to device-specific implementations.

## Requirements for Metrics Toggle
- A Prometheus exporter is enabled only when `--metrics-port` is provided.
- If the parameter is not provided, the miner runs with metrics fully disabled.
- The metrics stack should be optional and isolated to avoid side-effects when disabled.

## Requested Deliverables (from the prompt)
- A repository restructure that:
  - Separates CLI, service/orchestration, engines (CPU/GPU), math core, and metrics into distinct crates within a Cargo workspace.
  - Vendors or recreates `qpow-math` functionality in a local crate to facilitate optimization (e.g., `pow-core`).
  - Provides a baseline CPU engine and a clean engine interface for future optimized CPU and GPU engines.
- A CLI that accepts `--metrics-port` and toggles the Prometheus exporter accordingly, while preserving all existing node-facing behavior and API endpoints.
/*!
Deprecated root library entrypoint.

This repository has been restructured into a Cargo workspace to provide clear
separation of concerns and to enable aggressive optimization of the mining core.

Use the new workspace crates instead of the old root library:

- crates/miner-cli
  - The CLI binary for running the external miner service.
  - Examples:
      - cargo run -p miner-cli -- --port 9833
      - cargo run -p miner-cli -- --port 9833 --metrics-port 9900
      - cargo run -p miner-cli -- --num-cores 4

- crates/miner-service
  - The service layer: HTTP API (compatible with the node), job orchestration,
    and engine abstraction (CPU/GPU backends).

- crates/pow-core
  - The QPoW math core (local fork/superset of qpow-math) with a compatibility
    API and new optimized paths for future refactoring (e.g., precomputation,
    incremental evaluation, Montgomery multiplication).

- crates/engine-cpu
  - CPU mining engine(s) implementing the unified engine trait. Contains the
    baseline/reference engine and will host the optimized incremental/Montgomery
    engine.

- crates/engine-gpu-cuda (optional, scaffold)
  - Placeholder for a CUDA-based GPU engine.

- crates/engine-gpu-opencl (optional, scaffold)
  - Placeholder for an OpenCL-based GPU engine.

- crates/metrics (optional)
  - Prometheus metrics registry and optional HTTP exporter.
  - Metrics are toggled by the presence of the `--metrics-port` CLI parameter
    (when omitted, metrics are disabled entirely).

Notes:
- The public HTTP API remains compatible with the node.
- The old root library is intentionally left without exports to avoid conflicts.
- Build the binary via the new CLI package:
    cargo build -p miner-cli --release
- Run the service:
    cargo run -p miner-cli -- --port 9833 [--metrics-port 9900] [--num-cores N]
*/

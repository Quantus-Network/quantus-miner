# GPU miner bench (provider-agnostic)

Spin up a Quantus node + native GPU miner on any rented NVIDIA GPU host (Vast.ai,
RunPod, Lambda, etc.), sample hashrate / utilization / GPU specs, and append a
row to [`results.csv`](results.csv) so we can answer “what hardware should we
recommend?”

No cloud APIs. You supply `cloud_provider` and `cost_per_hour`.

## Prerequisites

On the GPU host:

- NVIDIA driver + `nvidia-smi`
- Docker (for the node image)
- Rust toolchain (to build the miner), **or** a prebuilt binary via `MINER_BIN`
- This `quantus-miner` checkout (scripts live in `gpu-bench/`)

## Quick start

```bash
cd gpu-bench
cp .env.example .env
# Set REWARDS_INNER_HASH (from: docker run --rm ghcr.io/quantus-network/quantus-node:latest key quantus --scheme wormhole)

./setup.sh
./record.sh --provider vast.ai --cost-per-hour 0.35

# When done
./setup.sh stop
```

### Hardware-only compare (no chain sync)

If you only need hashrate vs cost for the spreadsheet, skip the full stack:

```bash
./record.sh --benchmark --provider runpod --cost-per-hour 0.42 --duration 60
```

This runs `quantus-miner benchmark` (builds the miner if needed) and samples
`nvidia-smi` utilization during the run.

## Spreadsheet columns

| Column | Source |
|--------|--------|
| `timestamp` | UTC ISO time |
| `cloud_provider` | `--provider` |
| `gpu_model` | `nvidia-smi` name |
| `vram_mb` | total VRAM |
| `sm_count` | CUDA SM / multiprocessor count |
| `driver_version` | NVIDIA driver |
| `hashrate` | avg GPU H/s (`miner_gpu_hash_rate` live, or benchmark average) |
| `gpu_utilization_pct` | avg `utilization.gpu` over the sample |
| `cost_per_hour` | `--cost-per-hour` (USD) |
| `efficiency` | `hashrate / cost_per_hour` |
| `sample_seconds` | duration used |
| `notes` | optional `--notes` |

## Scripts

| Script | Purpose |
|--------|---------|
| [`setup.sh`](setup.sh) | Start node (Docker) + native GPU miner; `./setup.sh stop` tears down |
| [`record.sh`](record.sh) | Sample metrics and append one CSV row (`--live` or `--benchmark`) |

## Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `REWARDS_INNER_HASH` | (required for setup) | Wormhole preimage for rewards |
| `QUANTUS_MINER_DIR` | repo root (parent of `gpu-bench`) | Miner source tree |
| `MINER_BIN` | built `target/release/quantus-miner` | Skip cargo build if set |
| `GPU_DEVICES` | `1` | GPUs for miner / benchmark |
| `METRICS_PORT` | `9900` | Miner Prometheus port |
| `NODE_VERSION` | `latest` | `ghcr.io/quantus-network/quantus-node` tag |

## Notes

- The public `quantus-miner` Docker image is CPU-only. This toolkit builds the
  miner natively so WGPU can use host GPU drivers.
- Node and miner run on the **same host**; QUIC stays on `127.0.0.1:9833/udp`
  (do not publish that port to the public internet on multi-host setups).
- Mining during live bench uses `--cpu-workers 0` so hashrate reflects the GPU.

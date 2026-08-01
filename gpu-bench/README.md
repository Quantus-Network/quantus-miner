# GPU miner bench (provider-agnostic)

Spin up a Quantus node + native GPU miner on any rented NVIDIA GPU host (Clore,
RunPod, Vast.ai, etc.), sample hashrate / utilization / GPU specs, and append a row to
[`results.csv`](results.csv) so we can answer “what hardware should we recommend?”

No cloud APIs. You supply `cloud_provider` and `cost_per_hour`.

**Default path is native binaries on the same machine** (no Docker / miner-stack).
That works on RunPod, Vast, and similar GPU rentals.

## Prerequisites

On the GPU host:

- NVIDIA driver + `nvidia-smi`
- `curl` + `tar` (to fetch `quantus-node` from GitHub Releases)
- Rust toolchain (to build the miner), **or** a prebuilt binary via `MINER_BIN`
- This `quantus-miner` checkout

## Quick start (same-host mining)

```bash
cd gpu-bench
cp .env.example .env

./setup.sh wormhole          # downloads node if needed; save inner_hash
# paste inner_hash into .env as REWARDS_INNER_HASH

./setup.sh                   # native node + GPU miner on this host
./record.sh --provider clore --cost-per-hour 0.35    # or --provider runpod, vast, ...

./setup.sh stop
```

Pass whatever marketplace you rented from as `--provider` (e.g. `clore`,
`runpod`, `vast`) plus the machine's total `--cost-per-hour` in USD — the
scripts have no provider-specific logic.

`setup.sh` will:

1. Resolve `quantus-node` (`QUANTUS_NODE_BIN` → `PATH` → sibling `chain` build → GitHub release download into `.run/bin/`)
2. Generate a node key under `.run/node-keys/`
3. Start the node with `--miner-listen-port 9833`
4. Build/start the GPU miner against `127.0.0.1:9833` (`--cpu-workers 0`)

Optional Docker node (not recommended on RunPod): `./setup.sh start --docker`

### Hardware-only compare (no chain sync)

```bash
./record.sh --benchmark --provider runpod --cost-per-hour 0.42 --duration 60
```

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
| [`setup.sh`](setup.sh) | Native node + GPU miner on this host; `stop` / `status` |
| [`record.sh`](record.sh) | Sample metrics and append one CSV row (`--live` or `--benchmark`) |

## Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `REWARDS_INNER_HASH` | (required for setup) | Wormhole preimage for rewards |
| `QUANTUS_NODE_BIN` | download to `.run/bin/quantus-node` | Skip release download |
| `QUANTUS_MINER_DIR` | repo root | Miner source tree |
| `MINER_BIN` | built `target/release/quantus-miner` | Skip cargo build |
| `GPU_DEVICES` | `1` | GPUs for miner / benchmark |
| `METRICS_PORT` | `9900` | Miner Prometheus port |

## Notes

- Prefer this over `chain/miner-stack` on cloud GPUs: Compose + the public miner
  image are CPU-oriented and awkward on nested-Docker hosts.
- Node and miner share the host; QUIC is `127.0.0.1:9833/udp`.
- Give the volume enough disk for chain sync (tens of GB).
- Mining during live bench uses `--cpu-workers 0` so hashrate reflects the GPU.

# GPU miner bench (provider-agnostic)

Rent an NVIDIA GPU, build `quantus-miner`, run **`benchmark` across batch sizes**
(no node), and append rows to [`results.csv`](results.csv) for hardware
comparison. You supply `--provider` and `--cost-per-hour`.

## Miner binary (git build by default)

`remote-run.sh` / the RunPod sweep **clone + `cargo build -p miner-cli --release`**
from a git branch (default `illuzen/gpu-bench`) so you can iterate without
cutting a GitHub release. **Push the branch before sweeping.**

```bash
export MINER_BRANCH=illuzen/gpu-bench   # default
# escape hatch:
# export MINER_SOURCE=release
```

Container disk defaults to **50GB** for cargo `target/`. No `quantus-node`
download — hardware benches use `quantus-miner benchmark` only.

## On-pod one-shot

```bash
./remote-run.sh --provider runpod --cost-per-hour 0.69 --duration 30
# default batch sizes: 1M 4M 8M 16M (one CSV row each, notes include batch=N)
./remote-run.sh --cost-per-hour 0.39 --batch-sizes "1000000 16777216"
```

Or call record directly if the miner is already built:

```bash
MINER_BIN=./bin/quantus-miner ./record.sh --benchmark \
  --provider runpod --cost-per-hour 0.42 \
  --batch-sizes "1000000 4194304 16777216" --duration 30
```

Local full stack (node + serve) is still available via `./setup.sh --dev` +
`./record.sh --live` when you need end-to-end mining, not just hardware H/s.

## RunPod API sweep

Uses the **REST API** (not MCP): create Pod → SSH → `remote-run.sh` → scp CSV → delete.

### 1. Prerequisites (laptop)

- `RUNPOD_API_KEY` from [RunPod settings](https://www.runpod.io/console/user/settings)
- SSH public key added in RunPod **Settings → SSH Public Keys**
- `curl`, `ssh`, `scp`, `python3`

### 2. Image / template tips

Default image: `runpod/base:1.1.0-cuda1281-ubuntu2404` with
`NVIDIA_DRIVER_CAPABILITIES=all` (Vulkan/WGPU). Container disk ~50GB for git
builds. Expose TCP 22 for SSH.

### 3. Debug one Pod interactively

```bash
export RUNPOD_API_KEY=...
chmod +x runpod-shell.sh remote-run.sh record.sh runpod-sweep.sh batch-tune.sh

./runpod-shell.sh "NVIDIA L4"
./runpod-shell.sh --ssh
# on pod:
#   cd /workspace/quantus-gpu-bench
#   ./remote-run.sh --cost-per-hour … --duration 30

./runpod-shell.sh --delete
```

### 4. Run the sweep

```bash
export RUNPOD_API_KEY=...
# optional: export BATCH_SIZES="1000000 16777216"
# optional: export DURATION=30
# optional: export CLOUD_TYPE=SECURE

./runpod-sweep.sh --gpus-file gpus.example.txt
./runpod-sweep.sh --gpus-file gpus.all.txt
```

Successful rows append to [`results.csv`](results.csv) (one row per batch size
per Pod). Commit new rows so others can reuse them.

### What each Pod does

1. Build `quantus-miner` from git (`MINER_BRANCH`)  
2. Ensure NVIDIA Vulkan ICD (not Mesa llvmpipe)  
3. Smoke-test `benchmark` (5s)  
4. Sweep `--gpu-batch-size` values → append CSV rows (`notes` includes `batch=N`)  
5. Tear down

## Knobs that matter for differently shaped GPUs

| Knob | Where | Notes |
|------|--------|--------|
| `--gpu-batch-size` | CLI / sweep | **Main runtime knob.** Nonces per dispatch; interacts with tier workgroup hints (`nonces_per_thread` vs occupancy). |
| GPU tier table | `crates/engine-gpu/src/gpu_tiers.rs` | Per-name `workgroup_divisor` + `min_workgroups`. Not a CLI flag — edit + rebuild to retune a class of cards. |
| `--gpu-devices` | CLI | How many GPUs; bench defaults to GPU-only when this is set. |
| `--allow-integrated` | CLI | Include iGPUs when a discrete GPU is present. |
| `--gpu-throttle-ms` | `serve` only | Delay between batches; not used in `benchmark`. |
| threads/workgroup | hardcoded `256` | Not exposed. |

If util stays low across 1M→16M, dispatch shape may not be the bottleneck (driver, kernel, or tier mis-detect). Check miner logs for `tier:` / `workgroups:`.

## Collaborative dataset

[`results.csv`](results.csv) is the shared hardware comparison table. Prefer the
row with the best hashrate (or hash_per_dollar) per GPU when ranking hardware.

## Spreadsheet columns

| Column | Source |
|--------|--------|
| `cloud_provider` | `runpod` / flag |
| `gpu_model`, `vram_mb`, `sm_count` | `nvidia-smi` |
| `hashrate` | `benchmark` Total hashes / Total time |
| `gpu_utilization_pct` | avg `utilization.gpu` during the run |
| `cost_per_hour` | Pod `costPerHr` (sweep) or your flag |
| `cost_per_sec` | `cost_per_hour / 3600` |
| `hash_per_dollar` | `hashrate / cost_per_sec` |
| `notes` | includes `batch=N` for sweep rows |

## Scripts

| Script | Role |
|--------|------|
| [`remote-run.sh`](remote-run.sh) | On-box: build miner, Vulkan, batch-size benchmark → CSV |
| [`record.sh`](record.sh) | `--benchmark` (or `--live`) → CSV row(s) |
| [`batch-tune.sh`](batch-tune.sh) | Quick util/hashrate table without cost columns |
| [`runpod-sweep.sh`](runpod-sweep.sh) | RunPod REST API multi-GPU loop |
| [`runpod-shell.sh`](runpod-shell.sh) | One Pod + SSH; keep alive for manual debug |
| [`setup.sh`](setup.sh) | Local native node+miner (`--dev` or Planck) |
| [`gpus.all.txt`](gpus.all.txt) | Full RunPod NVIDIA `gpuTypeId` list |

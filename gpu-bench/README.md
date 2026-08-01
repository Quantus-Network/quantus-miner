# GPU miner bench (provider-agnostic)

Spin up a Quantus `--dev` node + GPU miner on a rented NVIDIA host, scrape
Prometheus hashrate into [`results.csv`](results.csv), and compare hardware.

## Miner binary (git build by default)

`remote-run.sh` / the RunPod sweep **clone + `cargo build -p miner-cli --release`**
from a git branch (default `illuzen/gpu-bench`) so you can iterate without
cutting a GitHub release. Push the branch before sweeping.

```bash
export MINER_BRANCH=illuzen/gpu-bench   # default
# escape hatch:
# export MINER_SOURCE=release
```

`quantus-node` still comes from the latest
[chain release](https://github.com/Quantus-Network/chain/releases/latest)
linux x86_64 tarball. Container disk defaults to **50GB** for cargo `target/`.

## Same-host (manual)

```bash
cd gpu-bench
./setup.sh --dev          # native --dev node + GPU miner (no rewards hash)
./record.sh --provider runpod --cost-per-hour 0.69
./setup.sh stop
```

On-pod one-shot (downloads binaries itself):

```bash
./remote-run.sh --provider runpod --cost-per-hour 0.69 --duration 60
```

## RunPod API sweep

Uses the **REST API** (not MCP): create Pod → SSH → `remote-run.sh` → scp CSV → delete.

### 1. Prerequisites (laptop)

- `RUNPOD_API_KEY` from [RunPod settings](https://www.runpod.io/console/user/settings)
- SSH public key added in RunPod **Settings → SSH Public Keys**
- `curl`, `ssh`, `scp`, `python3`

### 2. How to make a RunPod template (optional but nice)

You don’t *need* a custom template — the sweep defaults to a slim RunPod CUDA
base image (`runpod/base:…-cuda…`) with port `22/tcp`. **Don’t use PyTorch**
images unless you need Torch; they’re much larger and unused here. NVIDIA
drivers are on the host; the image just needs CUDA userspace + RunPod’s
`/start.sh` (SSH).

**Console**

1. [Templates → New Template](https://www.console.runpod.io/user/templates)
2. Image: e.g. `runpod/base:1.1.0-cuda1281-ubuntu2404` (use **24.04** — node needs GLIBC ≥ 2.38)
3. Env: `NVIDIA_DRIVER_CAPABILITIES=all` (required for Vulkan/WGPU; compute-only → llvmpipe only)
4. Container disk ~50GB if building miner from git (cargo `target/`); volume 0 unless you need persistence
5. Expose **TCP 22** (SSH). `runpod/base` / `runpod/pytorch` already start `sshd`.
6. Save → copy the template id → `export TEMPLATE_ID=...`

**API**

```bash
curl -sS -X POST https://rest.runpod.io/v1/templates \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "quantus-gpu-bench",
    "imageName": "runpod/base:1.1.0-cuda1281-ubuntu2404",
    "category": "NVIDIA",
    "containerDiskInGb": 50,
    "volumeInGb": 0,
    "volumeMountPath": "/workspace",
    "ports": ["22/tcp", "9900/http"],
    "env": { "NVIDIA_DRIVER_CAPABILITIES": "all" },
    "isPublic": false
  }'
```

### 3. Debug one Pod interactively (recommended first)

Creates a Pod, uploads scripts, prints SSH, and **leaves it running** so you
can fix Vulkan / miner commands without pay-per-recreate:

```bash
export RUNPOD_API_KEY=...
chmod +x runpod-shell.sh remote-run.sh record.sh runpod-sweep.sh

./runpod-shell.sh "NVIDIA L4"
# ssh … (printed), or: ./runpod-shell.sh --ssh

# edit remote-run.sh locally, then:
./runpod-shell.sh --upload
# on pod: bash -x ./remote-run.sh --cost-per-hour … --duration 30

./runpod-shell.sh --delete   # stop billing when done
```

### 4. Run the sweep

```bash
export RUNPOD_API_KEY=...
# optional: export TEMPLATE_ID=...
# optional: export CLOUD_TYPE=SECURE   # default COMMUNITY
# optional: export SSH_KEY=~/.ssh/id_ed25519

chmod +x remote-run.sh record.sh runpod-sweep.sh setup.sh

./runpod-sweep.sh --gpus-file gpus.example.txt
# full NVIDIA catalog:
./runpod-sweep.sh --gpus-file gpus.all.txt
# or:
./runpod-sweep.sh "NVIDIA GeForce RTX 4090" "NVIDIA GeForce RTX 3090"
```

Successful rows append to the shared dataset [`results.csv`](results.csv)
(commit new rows so others can reuse them). Per-pod temps go under
`sweep-out/` (gitignored).

On failure the Pod is deleted unless `KEEP_ON_FAILURE=1`.

If Community has no capacity for a GPU, the sweep retries on **Secure**
(`FALLBACK_SECURE=1` by default). If a Pod is RUNNING but never gets a public
IP (common on Community), it falls back to proxy SSH
(`podId@ssh.runpod.io`) and pipes files instead of `scp`.

### What each Pod does

1. Download `quantus-node` + `quantus-miner` release binaries  
2. `quantus-node --dev --miner-listen-port 9833` (local chain, no Planck sync)  
3. `quantus-miner serve --gpu-devices 1 --cpu-workers 0` → `:9900`  
4. Sample Prometheus (`miner_gpu_hash_rate`) + `nvidia-smi` into CSV  
5. Tear down

## Collaborative dataset

[`results.csv`](results.csv) is the shared hardware comparison table. After a
sweep or `record.sh` run, commit any new rows you want others to see.

## Spreadsheet columns

| Column | Source |
|--------|--------|
| `cloud_provider` | `runpod` / flag |
| `gpu_model`, `vram_mb`, `sm_count` | `nvidia-smi` (not the RunPod catalog) |
| `hashrate` | avg `miner_gpu_hash_rate` from `:9900/metrics` |
| `gpu_utilization_pct` | avg `utilization.gpu` |
| `cost_per_hour` | Pod `costPerHr` from API (sweep) or your flag |
| `cost_per_sec` | `cost_per_hour / 3600` |
| `hash_per_dollar` | `hashrate / cost_per_sec` (hashes per $) |

## Scripts

| Script | Role |
|--------|------|
| [`setup.sh`](setup.sh) | Local native node+miner (`--dev` or Planck) |
| [`remote-run.sh`](remote-run.sh) | On-box: fetch bins, `--dev`, mine, `record.sh` |
| [`record.sh`](record.sh) | Prometheus / nvidia-smi → CSV row |
| [`runpod-sweep.sh`](runpod-sweep.sh) | RunPod REST API multi-GPU loop |
| [`runpod-shell.sh`](runpod-shell.sh) | One Pod + SSH; keep alive for manual debug |
| [`gpus.all.txt`](gpus.all.txt) | Full RunPod NVIDIA `gpuTypeId` list |

## Notes

- `--dev` is for **hardware comparison**, not mainnet rewards.
- Prefer this over `chain/miner-stack` on cloud GPUs (Compose / nested Docker).
- Catalog “vCPU” is host CPU, not CUDA SMs — SM count comes from inside the Pod.

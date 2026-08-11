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

Container disk defaults to **20GB** (cargo `target/` during miner build; no
node/chain data). Override with `CONTAINER_DISK_GB` if needed.

## On-pod one-shot

```bash
./remote-run.sh --provider runpod --cost-per-hour 0.69 --duration 30
# default: batch sizes 256K 512K 1M 4M, --job-interval 2 (simulated NewJob)
./remote-run.sh --cost-per-hour 0.39 --job-interval 0   # sustained peak H/s
./remote-run.sh --cost-per-hour 0.39 --difficulty max   # cancel-only churn
```

Or call record directly if the miner is already built:

```bash
MINER_BIN=./bin/quantus-miner ./record.sh --benchmark \
  --provider runpod --cost-per-hour 0.42 \
  --job-interval 2 --batch-sizes "262144 524288 1000000" --duration 30
```

`benchmark --job-interval N` rotates a random header about every N seconds (like
node `NewJob`), with `--job-jitter` ±fraction (default 0.2 → sleep in
`[0.8N, 1.2N]`) so cancels don't alias to a fixed batch phase. Default difficulty
is `max` (cancel-only preemption, like mainnet). Pass `--difficulty <dec>` to
also exercise Found→idle. Reports wind_up / busy / wind_down ms per GPU.

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
`NVIDIA_DRIVER_CAPABILITIES=all` (Vulkan/WGPU). Container disk ~20GB for git
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

## Clore API sweep

Same idea as the RunPod sweep, adapted to Clore.ai's rental model:
create order → SSH → `remote-run.sh` → scp CSV → cancel. Needs `CLORE_API_KEY`
(dashboard → settings → API) or a token in `~/.config/clore/token`, a USD
balance on the account, and your SSH key (sent with each order — no dashboard
key setup needed).

```bash
export CLORE_API_KEY=...

./clore-sweep.sh --list "RTX 4090"        # print offers, rent nothing
./clore-sweep.sh "RTX 4090" "RTX 5080"    # benchmark, append to results.csv
GPU_COUNT=2 ./clore-sweep.sh "RTX 3090"   # 2x rig: all-GPU + isolated 1-GPU pass
./clore-sweep.sh --server 98539 --price-per-day 0.5   # exact offer
```

Clore-specific behavior baked in:

- **Prices are per day** (`cost_per_hour` = price/24), paid from the account's
  `USD-Blockchain` balance; every order costs a ~$0.10 creation fee.
- Offers are filtered by `MIN_RELIABILITY` (0.97), `MIN_RATING` (4), and
  optional `MAX_PRICE_PER_DAY`, then tried cheapest-first with
  `HOST_RETRIES` fallbacks — hot GPUs get rented out from under you.
- **Vulkan host screening:** some Clore hosts mount only the CUDA compute
  userspace; Vulkan is unfixable there (an exact-version driver `.run`
  extract still fails `vkCreateInstance`). A 30-second probe for
  `libGLX_nvidia.so.0` right after SSH skips those hosts for the next offer,
  costing the creation fee instead of a doomed 15-minute build.
- Orders are **always cancelled** on exit (`KEEP_ON_FAILURE=1` to debug).
- Clore is where the consumer cards live (30/40/50-series, multi-GPU rigs)
  at a fraction of RunPod prices — it fills the family gaps the RunPod
  sweep can't.

## Akash Console API sweep

Same end state (SSH → `remote-run.sh` → CSV), but via the **Akash Console
managed-wallet API**: post an SDL asking for a GPU model → providers bid →
lease the cheapest → inject SSH → bench → close. Needs `AKASH_API_KEY`
([console.akash.network](https://console.akash.network) → Settings → API Keys)
or `~/.config/akash/api_key`, and a funded Console balance (credit card).

```bash
export AKASH_API_KEY=...

./akash-sweep.sh --list rtx4090          # temp deploy, print bids, close
./akash-sweep.sh rtx5080 "RTX 4090"      # normalize → rtx5080 / rtx4090
./akash-sweep.sh --list --gpus-file gpus.all.txt   # map RunPod names → Akash ids
./akash-sweep.sh --gpus-file gpus.all.txt          # full bench sweep (slow + $)
DEPOSIT_USD=8 ./akash-sweep.sh h100
```

`gpus.all.txt` is RunPod-oriented: names are normalized (`NVIDIA GeForce RTX 4090` →
`rtx4090`, `A100-SXM4-80GB` → `a100`). Many lines get **zero bids** on Akash —
`--list` first, or trim the file. Each no-bid attempt still burns a small deposit.

Akash-specific notes:

- There is **no offer catalog** — discovery is “deploy SDL → read bids”.
  `--list` does that without leasing.
- SDL pricing denom defaults to **`uact`** (Console managed wallet). Bid
  price is per block; the script converts to `$/hr` (×`AKT_USD` or USDC).
- SDL asks for `nvidia` + `model: <name>` (e.g. `rtx4090`). Inventory skews
  AI/datacenter; consumer 50-series may be sparse vs Clore.
- Default image is `nvidia/cuda:12.6.3-devel-ubuntu24.04` with an SSH
  entrypoint; providers must expose the NVIDIA runtime.
- Deployments are closed on EXIT (`KEEP_ON_FAILURE=1` to debug).

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
Rows with utilization under 50% are dropped (unreliable samples).

To fit `hashrate ~ sm_count (+ vram_mb)` (not H/$, which is price-driven):

```bash
./fit-hashrate.py
```

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
| `ideal_hash_per_dollar` | `hash_per_dollar / (util/100)` — H/$ scaled to 100% util |
| `wind_up_ms` | mean setup→first-batch (job-sim only; empty if sustained/`--live`) |
| `busy_ms` | mean GPU batch wall time per job (job-sim only) |
| `wind_down_ms` | mean cancel-seen→return (job-sim only) |
| `notes` | includes `batch=N`, `job_interval=…` for sweep rows |

## Scripts

| Script | Role |
|--------|------|
| [`remote-run.sh`](remote-run.sh) | On-box: build miner, Vulkan, batch-size benchmark → CSV |
| [`record.sh`](record.sh) | `--benchmark` (or `--live`) → CSV row(s) |
| [`batch-tune.sh`](batch-tune.sh) | Quick util/hashrate table without cost columns |
| [`runpod-sweep.sh`](runpod-sweep.sh) | RunPod REST API multi-GPU loop |
| [`runpod-shell.sh`](runpod-shell.sh) | One Pod + SSH; keep alive for manual debug |
| [`clore-sweep.sh`](clore-sweep.sh) | Clore.ai marketplace rent → bench → cancel |
| [`akash-sweep.sh`](akash-sweep.sh) | Akash Console API deploy → bid → lease → bench |
| [`fit-hashrate.py`](fit-hashrate.py) | OLS `hashrate ~ sm_count (+ vram)` |
| [`setup.sh`](setup.sh) | Local native node+miner (`--dev` or Planck) |
| [`gpus.all.txt`](gpus.all.txt) | Full RunPod NVIDIA `gpuTypeId` list |

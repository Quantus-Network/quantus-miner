# engine-gpu-cuda (CUDA backend) – G1 bring‑up

This crate provides the CUDA GPU backend for the Quantus External Miner. It currently implements “G1” bring‑up: the per‑nonce modular multiply loop runs on the GPU (512‑bit Montgomery CIOS), while SHA3‑512 and the threshold check run on the host CPU. This allows correctness and plumbing to be validated before we move SHA3 and early‑exit onto the device in G2.

The backend is feature‑gated. When built with `--features cuda`, the crate’s build script compiles the CUDA kernel and embeds device images into the binary (CUBIN preferred, PTX as fallback). At runtime the engine selects an embedded image and launches the kernel to produce normalized `y` values per iteration.

---

## Status

- G1 (current):
  - Device: 512‑bit Montgomery multiply (CIOS using 64×64→128 via `__umul64hi`), maintaining `ŷ` in Montgomery domain and converting to normal domain for output.
  - Host: SHA3‑512 and threshold compare (and orchestration).
  - Correctness: parity against CPU small‑range tests.
  - Performance: primarily limited by PCIe copy‑back and host SHA3. See “Tuning” below.

- G2 (next):
  - Device: SHA3‑512 (Keccak‑f[1600], 24 rounds) optimized for 64‑byte input.
  - Device: threshold compare + global early‑exit flag (atomic) + tiny candidate write.
  - Device: move constants to `__constant__` memory.
  - Host: poll early‑exit; no large copy‑backs (only candidate or counters).
  - Result: removes PCIe and host‑SHA3 bottlenecks; enables real GPU‑bound throughput. Selection will be enabled via `MINER_CUDA_MODE=g2` once available.

---

## Build

The build script (for this crate only) compiles `crates/engine-gpu-cuda/src/kernels/qpow_kernel.cu` with `nvcc` and embeds both:
- CUBIN (native SASS for a specific SM, preferred at runtime).
- PTX (forward‑compatible target, used as a fallback when necessary).

Header preflight: the build verifies CUDA headers (e.g., `cuda_runtime.h`) exist. If it cannot embed either PTX or CUBIN, the build fails fast to prevent emitting a GPU‑broken binary.

Supported build modes:
- Native build (host has a CUDA toolkit): `cargo build -p miner-cli --features cuda --release`
- Containerized build (recommended for CI): build inside NVIDIA’s CUDA devel images; extract the binary (see repo workflow and Containerfile).

Notes:
- The kernel is compiled for one SM arch per build. Choose your SM via `CUDA_ARCH=sm_86|sm_89|sm_120` (see “Env knobs – build‑time”).
- For driver/toolkit compatibility (e.g., CUDA 12.9 driver), build the device images with a matching toolkit (e.g., 12.9).

---

## Runtime selection and embeds

At startup, the engine prefers the embedded CUBIN; if absent it falls back to the embedded PTX. You can override with `MINER_CUDA_IMAGE=cubin|ptx`. To attempt the G2 path (device SHA3 + early-exit), set `MINER_CUDA_MODE=g2`; if the G2 kernel isn’t embedded/available for the current device, the engine will fall back to G1 automatically. You’ll see logs like:
- `CUDA: using CUBIN (embedded)`
- `CUDA: using PTX source = embedded`
- (If neither exists, the engine logs the absence and delegates to CPU fast engine.)

When a job runs, the engine prints its launch configuration and per‑launch outcomes:
- `CUDA launch config: block_dim=…, threads=…, iters=…`
- `CUDA launch: grid_dim=…, block_dim=…, threads=…, iters=…`
- `CUDA kernel and sync OK`
- `CUDA copy-back OK: elems=…`

---

## Env knobs – runtime (G1)

These knobs affect GPU launch shape and how much work is returned to the host (and thus how much SHA3 the CPU must perform per launch).

- `MINER_CUDA_BLOCK_DIM` (default `256`)
  - Threads per block (`blockDim.x`). Use a multiple of 32 (warp size). 256 is a good default.
- `MINER_CUDA_THREADS`
  - Total threads (grid workload). Grid dimension is `grid_dim = ceil(threads / block_dim)`. Target at least “#SMs × 1–2 blocks” for decent occupancy (e.g., RTX 3060 has 28 SMs → 28 or 32 blocks).
- `MINER_CUDA_ITERS`
  - Iterations per thread. Higher values produce larger output buffers and more host SHA3 work per launch.
- `MINER_CUDA_IMAGE` = `cubin` | `ptx` (optional)
  - Overrides the embedded image choice (debugging/testing). Default is to prefer CUBIN.
- `MINER_CUDA_HASH_THREADS` (optional)
  - Number of host SHA3 worker threads to use to consume GPU output. Defaults to available parallelism.
- `MINER_CUDA_PINNED` = `1|true` (optional)
  - Use pinned (page-locked) host buffers and asynchronous device-to-host copies for G1 copy-back to reduce PCIe latency.
- `MINER_CUDA_MODE` = `g2` (optional)
  - Attempt G2 kernel (device SHA3-512 + threshold compare + early-exit). Falls back to G1 if the G2 kernel is not available for the current device image.

How much data per launch?
- y_out bytes = `threads × iters × 64`.
- Keep this around 64–128 MB in G1 to avoid PCIe and host SHA3 dominating.

Example configs (RTX 3060, SM 86):
- ~64 MB per launch:
  - `BLOCK_DIM=256`, `THREADS=8192`, `ITERS=128`
- ~128 MB per launch:
  - `BLOCK_DIM=256`, `THREADS=8192`, `ITERS=256`
- Good occupancy with ~117 MB:
  - `BLOCK_DIM=256`, `THREADS=7168` (28 blocks), `ITERS=256`

Service workers:
- Keep `--workers 1` for the GPU engine to avoid competing GPU launches.
- The engine internally orchestrates chunking and cancellation.

---

## Env knobs – build‑time (crate build script)

- `CUDA_ARCH` (required for device image quality)
  - One of: `sm_86` (Ampere 3060), `sm_89` (Ada 4090), `sm_120` (CC 12.0 / 5090), etc.
  - Normalized to `(compute_XX, sm_XX)` internally.
- `NVCC` or `CUDA_HOME` or `CUDA_PATH`
  - Path to `nvcc` or CUDA toolkit root.
- `MINER_CUDA_ALLOW_UNSUPPORTED_COMPILER` = `1` (optional)
  - Adds `-allow-unsupported-compiler` to `nvcc` if host GCC is newer than the toolkit supports.
- `MINER_NVCC_CCBIN` = `/path/to/g++-14` (optional)
  - Forces `nvcc -ccbin` to a specific (supported) host C++ compiler.
- (Container builds) The script attempts `/usr/local/cuda/targets/x86_64-linux/include` and `/usr/local/cuda/include` if `CUDA_HOME` is unset.

The build fails with clear messages if:
- `nvcc` cannot be found,
- CUDA headers cannot be found,
- neither PTX nor CUBIN artifacts were embedded.

---

## Tuning guide (G1)

Goal in G1: balance kernel time (GPU) against copy-back time (PCIe) and host SHA3 time (CPU) to avoid starving the GPU or overwhelming the host. Practical steps:

1) Size the output buffer:
   - Start with 64–128 MB per launch: `bytes ≈ threads × iters × 64`.
   - Increase `threads` to raise occupancy (more blocks). Start with `block_dim=256`.
   - Increase `iters` only while host SHA3 still keeps up.

2) Watch timings:
   - The engine logs `kernel_ms` and `copy_ms`.
   - If `copy_ms > kernel_ms`, try lowering `iters` or increasing `threads` to make the kernel heavier relative to copy.
   - If CPU is pegged (hashing), lower `iters`.

3) SM occupancy:
   - Ensure `grid_dim` (blocks) is at least the number of SMs on the device (e.g., 28+ on RTX 3060).
   - Use `threads ≈ block_dim × blocks` with `block_dim=256`.

4) Service settings:
   - Keep `--workers 1` when testing the GPU engine to avoid contention.

---

## Troubleshooting

- “missing CUDA headers (cuda_runtime.h)” at build time:
  - Install the matching CUDA toolkit or build inside an NVIDIA CUDA devel image (container).
  - Ensure `CUDA_HOME=/usr/local/cuda` (container) or set `NVCC` directly.

- “CUDA init failed” at runtime:
  - Check driver is installed, `nvidia-smi` works under the service user.
  - Ensure the unit has access to `/dev/nvidia*` (no device sandboxing).

- “no embedded PTX/CUBIN” at runtime:
  - The binary you deployed might be corrupted (download/transfer). Verify by:
    - `strings -a /path/to/binary | grep -m1 QPOW_KERNEL_CUBIN`
    - `strings -a /path/to/binary | grep -m1 qpow_montgomery_g1_kernel`
  - Rebuild and redeploy; avoid text-mode transfers; verify checksums.

- Driver/toolkit mismatch:
  - A CUBIN built by a newer toolkit may fail to load on older drivers. Use a toolkit matching your driver to produce the device images (e.g., CUDA 12.9 for driver 12.9).

---

## Roadmap to G2

- Device SHA3‑512 (Keccak‑f[1600], 24 rounds) tuned for 64B input.
- On‑device threshold compare and early‑exit flag (atomic).
- Host polling and tiny candidate copy‑back.
- Constants in `__constant__` memory.
- With G2, copy‑backs and host hashing disappear from the steady‑state path, enabling real GPU‑bound throughput.

---

## Appendix: build in containers (summary)

- Use the repo `Containerfile` and pass:
  - `--build-arg CUDA_TAG=12.9.0|13.0.0`
  - `--build-arg SM=86|89|120`
- The builder stage compiles and embeds PTX/CUBIN; the dist stage contains `/usr/local/bin/quantus-miner`.
- Export the dist stage filesystem (`buildx --target dist --output type=local,dest=./out`) and pick up `./out/usr/local/bin/quantus-miner`.
- Artifacts can be uploaded directly from CI or rsynced to your distribution host.

---

## Quick reference (env knobs)

Runtime:
- `MINER_CUDA_BLOCK_DIM` (default `256`) — threads per block.
- `MINER_CUDA_THREADS` — total threads (increase for more blocks).
- `MINER_CUDA_ITERS` — iterations per thread (controls y_out size).
- `MINER_CUDA_IMAGE` = `cubin|ptx` — force embedded image selection (optional).
- `MINER_CUDA_HASH_THREADS` — parallel host SHA3 workers (optional).
- `MINER_CUDA_PINNED` = `1|true` — use pinned host buffers + async D2H copy (G1 optimization).
- `MINER_CUDA_MODE` = `g2` — try device SHA3 + early-exit; falls back to G1 if G2 kernel isn’t available.

Build-time:
- `CUDA_ARCH` = `sm_86|sm_89|sm_120|…` — SM target for device images (normalized internally).
- `NVCC` or `CUDA_HOME` or `CUDA_PATH` — where to find the toolkit.
- `MINER_CUDA_ALLOW_UNSUPPORTED_COMPILER` = `1` — add `-allow-unsupported-compiler`.
- `MINER_NVCC_CCBIN` = `/path/to/g++-14` — force a specific host compiler.
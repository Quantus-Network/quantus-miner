# CUDA miner environment presets

This directory contains example environment files for commonly deployed NVIDIA GPUs. Each file sets reasonable defaults for the Quantus miner GPU engine and can be used as-is or as a starting point for further tuning.

Environment variables are read by the service via the systemd `EnvironmentFile` directive. You can point the service at any of these example files or copy one to your node’s standard location (for example, `/etc/resonance-miner.env`).

Key variables
- MINER_CUDA_BLOCK_DIM
  - Threads per block. Keep this a multiple of 32 (warp size). 256 is a solid default.
- MINER_CUDA_THREADS
  - Total threads launched. This sets the grid size as blocks = ceil(threads / block_dim).
  - Calculate threads = desired_blocks × block_dim.
- MINER_CUDA_ITERS
  - Iterations per thread.
  - G2 (device SHA3 + early-exit): increase for longer kernel dwell time while keeping early-exit responsiveness acceptable.
  - G1 (host SHA3): sets copy size and host hashing load. Try to keep threads × iters × 64 bytes ≈ 64–128 MB per launch.
- MINER_CUDA_MODE
  - Set to g2 to use the device SHA3 + early-exit kernel when available. If the G2 kernel is not embedded, the engine falls back to G1 automatically.
- MINER_CUDA_PINNED
  - Use pinned host buffers. Helpful in G1 to reduce D2H latency. Safe to leave enabled for G2 (doesn’t hurt).
- MINER_CUDA_HASH_THREADS
  - Only used by G1 (host SHA3). Defaults to available CPU parallelism.

General tuning guidance
- Target at least one block per SM for decent occupancy, and often 2× SMs is better for latency hiding.
- Number of blocks = ceil(MINER_CUDA_THREADS / MINER_CUDA_BLOCK_DIM).
- For a given GPU:
  - “Lower” preset ≈ 1× SMs blocks.
  - “Upper” preset ≈ 2× SMs blocks.
- Leave MINER_CUDA_BLOCK_DIM=256 unless you have a strong reason to change it; adjust blocks via MINER_CUDA_THREADS.
- For G2, prefer higher MINER_CUDA_ITERS (e.g., 2048–4096) to increase kernel durations and reduce host round-trip overhead, while still allowing fast early-exit.

Included presets

RTX 3060 (28 SMs)
- cuda-miner-3060-lower.env
  - 28 blocks × 256 threads/block → MINER_CUDA_THREADS=7168
  - “Lower” preset (≈1× SMs blocks)
- cuda-miner-3060-upper.env
  - 56 blocks × 256 threads/block → MINER_CUDA_THREADS=14336
  - “Upper” preset (≈2× SMs blocks)

RTX 3080 (68 SMs)
- cuda-miner-3080-lower.env
  - 68 blocks × 256 threads/block → MINER_CUDA_THREADS=17408
  - “Lower” preset
- cuda-miner-3080-upper.env
  - 136 blocks × 256 threads/block → MINER_CUDA_THREADS=34816
  - “Upper” preset

RTX 3090 (82 SMs)
- cuda-miner-3090-lower.env
  - 82 blocks × 256 threads/block → MINER_CUDA_THREADS=20992
  - “Lower” preset
- cuda-miner-3090-upper.env
  - 164 blocks × 256 threads/block → MINER_CUDA_THREADS=41984
  - “Upper” preset

RTX 4080 (76 SMs)
- cuda-miner-4080-lower.env
  - 76 blocks × 256 threads/block → MINER_CUDA_THREADS=19456
  - “Lower” preset
- cuda-miner-4080-upper.env
  - 152 blocks × 256 threads/block → MINER_CUDA_THREADS=38912
  - “Upper” preset

RTX 4090 (128 SMs)
- cuda-miner-4090-lower.env
  - 128 blocks × 256 threads/block → MINER_CUDA_THREADS=32768
  - “Lower” preset
- cuda-miner-4090-upper.env
  - 256 blocks × 256 threads/block → MINER_CUDA_THREADS=65536
  - “Upper” preset

RTX 5090 (est.)
- cuda-miner-5090-lower.env
  - Conservative starting point: 128 blocks × 256 threads/block → MINER_CUDA_THREADS=32768
  - Adjust threads toward ≈ (#SMs × 256) for “lower”
- cuda-miner-5090-upper.env
  - 256 blocks × 256 threads/block → MINER_CUDA_THREADS=65536
  - Adjust threads toward ≈ (2 × #SMs × 256) for “upper”

RTX A5000 (64 SMs)
- cuda-miner-a5000-lower.env
  - 64 blocks × 256 threads/block → MINER_CUDA_THREADS=16384
  - “Lower” preset
- cuda-miner-a5000-upper.env
  - 128 blocks × 256 threads/block → MINER_CUDA_THREADS=32768
  - “Upper” preset

RTX A6000 (84 SMs)
- cuda-miner-a6000-lower.env
  - 84 blocks × 256 threads/block → MINER_CUDA_THREADS=21504
  - “Lower” preset
- cuda-miner-a6000-upper.env
  - 168 blocks × 256 threads/block → MINER_CUDA_THREADS=43008
  - “Upper” preset

Datacenter GPUs (A100/H100) — guidance
- Build: use a matching CUDA_ARCH (e.g., sm_80 for A100, sm_90 for H100) to embed native CUBINs for your driver.
- Start with:
  - Lower: blocks ≈ #SMs, threads = blocks × 256
  - Upper: blocks ≈ 2 × #SMs, threads = blocks × 256
- Iterations (G2): start at 4096–8192; increase to reduce host round-trips; decrease if you need faster early-exit responsiveness.
- Keep --workers 1 for single-GPU nodes.
- If you want presets added to this directory, use the “lower/upper” naming convention: cuda-miner-a100-lower.env, cuda-miner-a100-upper.env, cuda-miner-h100-lower.env, cuda-miner-h100-upper.env.

Usage
- With systemd
  - Point your service `EnvironmentFile=` at the selected file in this directory, or copy the file to `/etc/resonance-miner.env`.
  - Example:
    - EnvironmentFile=/path/to/quantus-miner/examples/.env/cuda-miner-4090-256.env
- Manual testing
  - Export the variables in your shell before starting `quantus-miner` with `--engine gpu-cuda`.

Notes
- These presets default to G2 mode (MINER_CUDA_MODE=g2) where applicable. If your binary doesn’t include the G2 kernel image for your device, the engine falls back to G1 automatically.
- For G1-only deployments, consider smaller MINER_CUDA_ITERS so threads × iters × 64 stays within 64–128 MB per launch to keep PCIe and host SHA3 balanced.
- Keep `--workers 1` for GPU runs to avoid contention on a single device.
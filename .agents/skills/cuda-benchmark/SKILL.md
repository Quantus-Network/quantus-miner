---
name: cuda-benchmark
description: Optimize the Quantus native CUDA mining kernel using correctness-gated, same-host GPU experiments on Clore.ai or Vast.ai, and document reproducible throughput results.
---

# Quantus CUDA experiments

Optimize `crates/engine-cuda/src/kernels/mining.cu` and its Rust dispatch code for measured throughput. Read the repository rules and `docs/clore-gpu-benches.md` before changing code. Preserve the existing fixed values in `pow-core::NONCE_HASH_KVS`.

## Rental setup

- Use the provider and total budget authorized in the conversation. Include creation fees, runtime, and storage in the estimate; check current prices. This skill does not authorize spending by itself.
- Source `~/play/quantus-network/agent-tmp.env`; put experiment files and logs under `$CODEX_AGENT_TMP`. Keep credentials out of logs, commits, uploads, and this skill.
- Inspect active rentals before creating one. Reuse a suitable task-owned instance; never cancel someone else's rental. Save the new order ID immediately, establish a budget-based cancellation deadline, and cancel promptly after collecting results.
- If available, inspect `~/play/gpu-sheperd/server/src/providers/clore.ts` or `vast.ts` for maintained API details. Its bootstrap/README currently describe the Vulkan miner; their graphics requirements do not apply to `--cuda-gpu`.
- Clore: API base `https://api.clore.ai/v1`, header `auth`, local configuration `~/.clore/config.json` (`api_key`). Respect roughly 1.1 seconds between calls and 5.2 seconds between creates. Check JSON `code`, even on HTTP success. If creation returns only `{code: 0}`, find the new order by server ID and creation time before retrying. Resolve SSH from `pub_cluster` and the `22:` entry in `tcp_ports`.
- `cloreai/jupyter:ubuntu24.04-v2` has provided working SSH in these experiments. Some CUDA devel images did not. Container startup may rewrite SSH credentials: establish a multiplexed connection early, keep it active, and use a short task-local `ControlPath`. Do not assume yesterday's host or port still applies.
- Verify `nvidia-smi` and a working NVRTC compilation/module load. Match NVRTC's generated PTX to the host driver. For example, driver 535.171.04 worked with `nvidia-cuda-nvrtc-cu12==12.2.140`; set `LD_LIBRARY_PATH` to its `nvidia/cuda_nvrtc/lib` directory. CUDA needs `libcuda` and `libnvrtc`, not Vulkan.
- A missing Clore Vulkan mount can sometimes be repaired inside the container: extract NVIDIA's checksum-verified `.run` archive for the **exact host driver version** with `--extract-only`; never run the driver installer. Install the Vulkan/X11/EGL loader dependencies, supply `libGLX_nvidia.so.0` and `libEGL_nvidia.so.0` symlinks to the extracted matching libraries, set `LD_LIBRARY_PATH`, point `VK_ICD_FILENAMES` at the NVIDIA ICD, and set `__EGL_VENDOR_LIBRARY_FILENAMES` to the extracted `10_nvidia.json`. The EGL vendor manifest was essential on server 90954; libraries alone still returned `ERROR_INCOMPATIBLE_DRIVER`. Require `vulkaninfo --summary` to identify the actual NVIDIA device before measuring WGSL.
- For Vast, consult `~/play/gpu-sheperd/docs/vast-ai/SKILL.md` when available. The adapter registers the public SSH key through the instance API. Destroy instances to end compute and storage charges; stopping is insufficient. Use graphics-capable images for the required same-host WGSL baseline when the provider supports them.

## Correctness and measurement

1. Save the exact baseline revision and kernel. Establish both the unchanged WGSL miner (base 1) and existing CUDA implementation (base 2) on every host before comparing candidates. Use the same host, batch size, duration, and power/clock settings. If Vulkan is unavailable, explicitly record the failed WGSL baseline and attempt a compatible userspace setup; never substitute a different machine's rate as a same-host baseline.
2. Gate every candidate on all five fixed nonce/hash vectors, using their stored midstates. Also test mining with targets `hash - 1`, `hash`, and `hash + 1`: only the last may succeed, and the returned nonce and full hash must match. Equality exercises the lower half of the target comparison.
3. For arithmetic changes, exercise modular edge cases and deterministic CPU/GPU parity batches, including nonce carries. The shipped Rust tests must run on the GPU before accepting a candidate. A skipped CUDA test is not a correctness pass.
4. Change one factor at a time. Keep rejected variants outside the source tree. Explore combinations only after individual measurements justify them. Keep a change only after repeated same-host improvement survives correctness checks.
5. Warm up the GPU, record at least three samples, and alternate baseline/candidate order for final confirmation. Record GPU UUID, driver, NVRTC version/options, power limit, clocks, temperature, batch size, duration, and source hashes. Keep host settings identical.
6. Use kernel-only timing for screening, then confirm with the actual release miner:

   ```sh
   quantus-miner benchmark --cuda-gpu --gpu-devices 1 --cpu-workers 0 --duration 15 --gpu-batch-size 4000000
   ```

   Confirm the current CLI spelling with `benchmark --help`. Report kernel-only and end-to-end rates separately. Do not compare differing hosts, miner versions, or batches as a proven speedup.
7. Run the repository's applicable formatting, lint, build, and test checks with stable Rust. For macOS-to-Linux binaries, `cargo +stable zigbuild --target x86_64-unknown-linux-gnu.2.35` requires both the Linux Rust target and Zig on `PATH`. Upload only the binaries/test fixtures needed, without repository credentials.
8. Record kept/rejected experiments, raw samples, validation, rental IDs, and actual or explicitly estimated charges in `docs/clore-gpu-benches.md`. Cancel the task-owned order and verify it is expired/absent from active orders. Follow the conversation's commit/push instructions.

## Reusable comparison script

Copy [scripts/compare.py](scripts/compare.py) to the single-GPU rental along with the unchanged and candidate release miners. Set the working CUDA/Vulkan environment from the rental setup, then run:

```sh
python3 compare.py --baseline ./miner-baseline --candidate ./miner-candidate --output final-comparison
```

The output directory must be new. Defaults: 5-second warmups, three alternating 15-second measurements per engine, and 1M/4M batches. It records raw logs, binary SHA256 values, GPU UUID, clocks, temperature, power, and rates in `comparison.jsonl`. Run the CUDA correctness tests separately before this comparison; this script measures performance and does not replace them.

## Research and interpretation

Use primary sources to select hypotheses: [NVIDIA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/), [inline PTX constraints](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html), and [PTX carry arithmetic](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html). Higher occupancy, more unrolling, and `__restrict__` do not guarantee a gain. Inspect register counts and confirm with measurements. Do not change field constants, round counts, or hash semantics for speed.

For this kernel, reduction uses `2^64 ≡ 2^32 - 1` and `2^96 ≡ -1` modulo the Goldilocks prime. Combining carry and borrow correction is an experimentally useful direction; preserve its signed correction and test 128-bit edge cases against CPU modulo arithmetic. See the recorded experiments before repeating rejected variants.

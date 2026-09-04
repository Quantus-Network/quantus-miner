# Clore.ai GPU benches

Quantus miner v4.0.2. GPU-only (`--cpu-workers 0 --gpu-devices 1`). Default bench is 10 s `quantus-miner benchmark`.

- **CUDA** = `--cuda-gpu` (native NVRTC kernel, no Vulkan).
- **WGSL** = default wgpu engine (Vulkan on Linux).

Prices are Clore on-demand USD-Blockchain per day at order time. Creation fee is $0.10 extra.

## Summary

| Date (UTC) | Engine | GPU | Clore server | Order | Region | Host CUDA | Driver | Image | Result |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-04 | CUDA | NVIDIA GeForce RTX 3060 Ti 8GB | 103764 | 2086777 | US | 12.4 | 550.144.03 | nvidia/cuda:12.4.1-devel-ubuntu22.04 | **68.04 MH/s** |
| 2026-09-04 | CUDA | NVIDIA GeForce RTX 3070 8GB | 103476 | 2086827 | US | 12.4 | 550.144.03 | nvidia/cuda:12.4.1-devel-ubuntu22.04 | **81.49 MH/s** |
| 2026-09-04 | CUDA | NVIDIA GeForce RTX 4060 Ti 8GB | 66738 | 2086826 | RU | 13.0 | 580.126.09 | nvidia/cuda:12.4.1-devel-ubuntu22.04 | **79.13 MH/s** |
| 2026-09-04 | CUDA | NVIDIA GeForce RTX 5060 Ti 16GB | 102905 | 2086828 | CR | 13.3 | 610.43.02 | nvidia/cuda:12.4.1-devel-ubuntu22.04 | **104.50 MH/s** |
| 2026-09-04 | CUDA | NVIDIA GeForce RTX 5070 Ti 16GB | 109004 | 2086829 | NL | 13.2 | — | nvidia/cuda:12.4.1-devel-ubuntu22.04 | SSH never came up (`Connection reset`) |
| 2026-09-04 | WGSL | NVIDIA GeForce RTX 3070 8GB | 103476 | 2086827 | US | 12.4 | 550.144.03 | nvidia/cuda:12.4.1-devel-ubuntu22.04 | fail: no usable wgpu adapter (`libGLX_nvidia.so.0` missing) |
| 2026-09-04 | WGSL | NVIDIA GeForce RTX 4060 Ti 8GB | 66738 | 2086826 | RU | 13.0 | 580.126.09 | nvidia/cuda:12.4.1-devel-ubuntu22.04 | fail: no usable wgpu adapter |
| 2026-09-04 | WGSL | NVIDIA GeForce RTX 5060 Ti 16GB | 102905 | 2086828 | CR | 13.3 | 610.43.02 | nvidia/cuda:12.4.1-devel-ubuntu22.04 | fail: no usable wgpu adapter (`libGLX_nvidia.so.0` missing) |
| 2026-09-04 | WGSL | NVIDIA GeForce RTX 3060 | unknown | unknown | unknown | unknown | unknown | nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04 | fail: `libGLX_nvidia.so.0` missing |
| 2026-09-04 | WGSL | NVIDIA GeForce RTX 3070 | unknown | unknown | unknown | unknown | unknown | nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04 | fail: `libGLX_nvidia.so.0` missing |
| 2026-09-04 | WGSL | NVIDIA GeForce RTX 4060 Ti | unknown | unknown | unknown | unknown | unknown | nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04 | fail: `libGLX_nvidia.so.0` missing |
| 2026-09-04 | WGSL | NVIDIA GeForce RTX 4070 Ti | unknown | unknown | unknown | unknown | unknown | nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04 | fail: `libGLX_nvidia.so.0` missing |
| 2026-09-04 | WGSL | 40-class (two hosts) | unknown | unknown | unknown | unknown | unknown | nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04 | worked; hashrate not recorded |

Rows without server IDs are from the earlier Clore hunt (pre-CUDA). nvidia-smi listed the card; Vulkan ICD was not injected. Hashrate was not written down for the two WGSL hosts that did boot.

## Same-host CUDA vs WGSL

All four CUDA-image boxes that reached SSH: CUDA worked, WGSL did not.

| Clore server | GPU | CUDA | WGSL |
|---|---|---|---|
| 103764 | RTX 3060 Ti | 68.04 MH/s | not run (first CUDA-only rental) |
| 103476 | RTX 3070 | 81.49 MH/s | fail, no NVIDIA Vulkan ICD |
| 66738 | RTX 4060 Ti | 79.13 MH/s | fail, no NVIDIA Vulkan ICD |
| 102905 | RTX 5060 Ti | 104.50 MH/s | fail, no NVIDIA Vulkan ICD |
| 109004 | RTX 5070 Ti | — | SSH never came up |

68 MH/s on a 3060 Ti is in the same band as these other CUDA numbers (3070 81, 4060 Ti 79). Same-host WGSL on the 2026-09-04 CUDA-image boxes failed (no NVIDIA Vulkan ICD).

## CUDA vs earlier WGSL (session history)

This Grok session’s first message is a truncated recap of a v4.0.2 Clore WGSL hunt (`nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04`). I searched that recap, this session’s terminal logs, other Grok sessions, Claude/Codex/Cursor/Kimi transcripts, and Clore `my_orders`. No MH/s was written down for the two WGSL hosts that booted. The 3060 / 3070 / 4060 Ti / 4070 Ti attempts in that hunt failed at `libGLX_nvidia.so.0`.

The only stored Clore WGSL hashrates are older `gpu-bench/results.csv` rows (commit `65e955b`, 2026-08-10, 30 s samples, 1M batch). That is a different miner vintage than v4.0.2, so these are not a same-kernel comparison.

| Date (UTC) | Engine | GPU | Clore server | Batch | Sample | Result |
|---|---|---|---|---|---|---|
| 2026-08-10 | WGSL | RTX 5080 | 95043 | 1_000_000 | 30 s | **21.29 MH/s** (util 92.4%) |
| 2026-08-10 | WGSL | RTX 5080 | 95043 | 4_194_304 | 30 s | 20.69 MH/s |
| 2026-08-10 | WGSL | 2× RTX 3090 | 107400 | 1_000_000 | 30 s | **18.10 MH/s** (util 93.3%, both GPUs) |
| 2026-08-10 | WGSL | 2× RTX 3090 | 107400 | 4_194_304 | 30 s | 16.74 MH/s |
| 2026-08-10 | WGSL | RTX 5060 Ti | 104602 | 1_000_000 | 30 s | **10.34 MH/s** (util 99.9%) |
| 2026-08-10 | WGSL | RTX 5060 Ti | 104602 | 4_194_304 | 30 s | 9.33 MH/s |
| 2026-08-10 | WGSL | RTX 2070 SUPER | 30480 | 1_000_000 | 30 s | **3.37 MH/s** (util 98.9%) |
| 2026-08-10 | WGSL | RTX 2070 SUPER | 30480 | 4_194_304 | 30 s | 2.48 MH/s |

Closest same-name card, still not same kernel or duration:

| GPU | WGSL (2026-08-10 Clore, 30 s) | CUDA (2026-09-04 Clore, 10 s) |
|---|---|---|
| RTX 5060 Ti | 10.34 MH/s (server 104602) | 104.50 MH/s (server 102905) |

That is about 10× on paper. Do not treat it as a kernel speedup. The August WGSL numbers are from the pre-v4.0.2 gpu-bench miner. A fair Clore WGSL vs CUDA number still needs a host that injects `libGLX_nvidia.so.0`.

## CUDA detail

Command:

```
quantus-miner benchmark --cuda-gpu --gpu-devices 1 --cpu-workers 0 --duration 10
```

Image: `nvidia/cuda:12.4.1-devel-ubuntu22.04` with `NVIDIA_DRIVER_CAPABILITIES=all`. Binary built once on the 3070 box (`cargo build -p miner-cli --release`) and copied.

KV tests `cuda_matches_nonce_hash_golden_vectors` and `cuda_search_finds_cpu_verified_solution` passed on 103764 (3060 Ti).

### NVIDIA GeForce RTX 3060 Ti

| Field | Value |
|---|---|
| Date (UTC) | 2026-09-04 |
| Miner | v4.0.2 (local tree with `engine-cuda`) |
| Engine | CUDA (`gpu-cuda`) |
| Clore server ID | 103764 |
| Order ID | 2086777 |
| GPU | 1x NVIDIA GeForce RTX 3060 Ti |
| VRAM | 8 GB |
| UUID | GPU-5a345e96-0ad7-6a20-43de-2bb4886bd74e |
| Region | US (`n1.us.clorecloud.net:1293`) |
| CPU | AMD Ryzen 9 9900X 12-Core |
| RAM | 31.6 GB |
| Host CUDA / driver | 12.4 / 550.144.03 |
| Price | $1.75/day + $0.10 create |
| Reliability | 0.9991 |
| Bench | 10.01 s, 681_000_000 hashes, **68.04 MH/s** |
| KV tests | pass |

### NVIDIA GeForce RTX 3070

| Field | Value |
|---|---|
| Date (UTC) | 2026-09-04 |
| Miner | v4.0.2 (local tree with `engine-cuda`) |
| Engine | CUDA (`gpu-cuda`) |
| Clore server ID | 103476 |
| Order ID | 2086827 |
| GPU | 1x NVIDIA GeForce RTX 3070 |
| VRAM | 8 GB |
| UUID | GPU-0eedb451-7a55-9b15-d20f-a1ee0a790617 |
| Region | US (`n1.us.clorecloud.net:1293`) |
| CPU | AMD Ryzen 9 5900X 12-Core |
| RAM | 32.0 GB |
| Host CUDA / driver | 12.4 / 550.144.03 |
| Price | $1.75/day + $0.10 create |
| Reliability | 0.9985 |
| Bench | 816_000_000 hashes, **81.49 MH/s** |
| WGSL on same host | fail: `libGLX_nvidia.so.0` missing; wgpu enumerated only llvmpipe |

### NVIDIA GeForce RTX 4060 Ti

| Field | Value |
|---|---|
| Date (UTC) | 2026-09-04 |
| Miner | v4.0.2 (local tree with `engine-cuda`) |
| Engine | CUDA (`gpu-cuda`) |
| Clore server ID | 66738 |
| Order ID | 2086826 |
| GPU | 1x NVIDIA GeForce RTX 4060 Ti |
| VRAM | 8 GB |
| UUID | GPU-88467adf-a7d3-c382-79ff-4c7c3f000844 |
| Region | RU (`n1.msk.cloreai.ru:1266`) |
| CPU | Intel Xeon E3-1275 V2 @ 3.50GHz |
| RAM | 32.0 GB |
| Host CUDA / driver | 13.0 / 580.126.09 |
| nvidia-smi power cap | 100 W |
| Price | $1.05/day + $0.10 create |
| Reliability | 0.9988 |
| Bench | 792_000_000 hashes, **79.13 MH/s** |
| WGSL on same host | fail: no usable wgpu adapter |

### NVIDIA GeForce RTX 5060 Ti

| Field | Value |
|---|---|
| Date (UTC) | 2026-09-04 |
| Miner | v4.0.2 (local tree with `engine-cuda`) |
| Engine | CUDA (`gpu-cuda`) |
| Clore server ID | 102905 |
| Order ID | 2086828 |
| GPU | 1x NVIDIA GeForce RTX 5060 Ti |
| VRAM | 16 GB (16311 MiB) |
| UUID | GPU-21492b68-3eba-e246-dcff-5e144a647f2c |
| Region | CR (`n1.us.clorecloud.net:1548`) |
| CPU | AMD Ryzen 5 5500 |
| RAM | 27.1 GB |
| Host CUDA / driver | 13.3 / 610.43.02 |
| Price | $1.70/day + $0.10 create |
| Reliability | 0.9969 |
| Bench | 1_046_000_000 hashes, **104.50 MH/s** |
| WGSL on same host | fail: `libGLX_nvidia.so.0` missing; vulkaninfo had only Mesa ICDs (intel/lvp/radeon/virtio), no NVIDIA ICD |

### NVIDIA GeForce RTX 5070 Ti (no SSH)

| Field | Value |
|---|---|
| Date (UTC) | 2026-09-04 |
| Clore server ID | 109004 |
| Order ID | 2086829 |
| GPU | 1x NVIDIA GeForce RTX 5070 Ti |
| VRAM | 15.92 GB |
| Region | NL (`n1.de.clorecloud.net:1314`) |
| CPU | Intel Xeon E5-2680 v4 @ 2.40GHz |
| RAM | 32.0 GB |
| Host CUDA | 13.2 |
| Price | $2.40/day + $0.10 create |
| Reliability | 0.9983 |
| Result | SSH mapped but `kex_exchange_identification: Connection reset by peer` for ~5 min; cancelled |

## WGSL / wgpu (earlier Clore hunt)

Image was always `nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04` with `NVIDIA_DRIVER_CAPABILITIES=all`. Miner v4.0.2 release, GPU-only. Server IDs were not written down.

| GPU (as remembered) | Clore server ID | Result |
|---|---|---|
| RTX 3060 | not recorded | nvidia-smi ok, Vulkan fail (`libGLX_nvidia.so.0` missing) |
| RTX 3070 | not recorded | same Vulkan fail |
| RTX 4060 Ti | not recorded | same Vulkan fail |
| RTX 4070 Ti | not recorded | same Vulkan fail |
| 40-class (2 hosts) | not recorded | WGSL miner started; hashrate not recorded |
| RTX 5060 Ti (NL owner, lead) | not recorded | graphics stack was injected on that owner’s 5060 Ti; not re-tested here |

Other attempts never published SSH.

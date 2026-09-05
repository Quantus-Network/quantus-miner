# Vast.ai same-host WGSL vs CUDA

Fair comparison on one NVIDIA box where **both** engines can run: wgpu/Vulkan
(official Linux release) and native CUDA (`--cuda-gpu` from
`quantus-miner-private`). GPU-only, `--cpu-workers 0 --gpu-devices 1`.

CUDA flag (serve and benchmark):

```
quantus-miner benchmark --cuda-gpu --gpu-devices 1 --cpu-workers 0 --duration 10
```

WGSL is the default (no `--cuda-gpu`). CUDA needs `libcuda` (NVIDIA driver) and
`libnvrtc` (CUDA toolkit). The OpenGL image used for wgpu does not ship NVRTC;
install `cuda-nvrtc-12-4` and set `LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64`.

## Same host (2026-09-04)

Vast instance `49862636`, offer ~$0.108/h, Quebec CA. Image
`nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04` with
`NVIDIA_DRIVER_CAPABILITIES=all`. Driver 560.35.03, 300 W cap, 12 GB.

| Engine | Binary | Duration | Hashes | Rate |
|---|---|---|---|---|
| WGSL native-u64 | official **v4.0.2** `quantus-miner-linux-x86_64` | 10 s | 1.062e9 | **106.03 MH/s** |
| WGSL native-u64 | private tree (same crate as CUDA, no `--cuda-gpu`) | 10 s | 1.065e9 | **106.35 MH/s** |
| CUDA `--cuda-gpu` | private `engine-cuda` v8 kernel, built on box | 10 s | 2.743e9 | **273.83 MH/s** |
| CUDA `--cuda-gpu` | same binary | 15 s | 4.112e9 | **272.86 MH/s** |

CUDA / official WGSL ≈ **2.58×** on this 3080 Ti.

Private WGSL matching official WGSL means the CUDA gap is the kernel, not a
different Poseidon2 implementation in the private tree.

## Notes

- Official WGSL log: `DiscreteGpu, Vulkan`, `using native-u64`.
- CUDA log: `CUDA device 0: NVIDIA GeForce RTX 3080 Ti`.
- Without `libnvrtc`, the CUDA binary `abort`s (`panic = "abort"`) before
  logging, because NVRTC load is wrapped in `catch_unwind`.
- Instance destroyed after the benches.

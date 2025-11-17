# External Miner Service for Quantus Network

Note: This repository is now a Cargo workspace. Build and run the CLI with:
- cargo build -p miner-cli --release
- cargo run -p miner-cli -- --port 9833 [--metrics-port 9900] [--workers N]

This crate provides an external mining service that can be used with a Quantus Network node. It exposes an HTTP API for
managing mining jobs.

## Building

To build the external miner service, navigate to the `miner` directory within the repository and use Cargo:

```bash
cd quantus-miner
cargo build --release
```

This will compile the binary and place it in the `target/release/` directory.

## CUDA Build (optional, Linux only)

The CUDA backend is feature-gated and currently supported on Linux with NVIDIA GPUs. macOS is not supported for CUDA.

Build with CUDA enabled:
```bash
# Build the CLI with CUDA feature (compiles .cu kernels to PTX via nvcc)
cargo build -p miner-cli --features cuda --release
```

At runtime, select the engine with:
```bash
# Will error if CUDA runtime/driver is unavailable
./target/release/quantus-miner --engine gpu-cuda --metrics-port 9919
```

Environment knobs used by the CUDA build:
- NVCC: Path to the nvcc binary (optional if in PATH)
- CUDA_HOME or CUDA_PATH: Used to locate nvcc at $CUDA_HOME/bin/nvcc when NVCC is unset
- CUDA_ARCH: Compute capability target for PTX (default: sm_70)

If nvcc isn’t found, the build will continue but skip compiling kernels; the GPU engine will then fall back to CPU at runtime.

### Ubuntu (22.04/24.04) setup

Install NVIDIA driver and CUDA toolkit:
```bash
# Install proprietary driver (pick latest recommended)
sudo ubuntu-drivers autoinstall
sudo reboot

# CUDA toolkit (provides nvcc). Option A: Ubuntu package (may be older):
sudo apt update
sudo apt install -y nvidia-cuda-toolkit

# Verify nvcc
nvcc --version
```
For newer toolkits, consider NVIDIA’s official repo: https://developer.nvidia.com/cuda-downloads

### Fedora 42 (dnf5) setup

Install NVIDIA driver and CUDA toolkit:
```bash
# Enable RPM Fusion for proprietary NVIDIA drivers (driver comes from here)
sudo dnf5 install -y https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm

# Install driver (akmod builds the module for your current kernel)
sudo dnf5 install -y akmod-nvidia
sudo reboot

# Add NVIDIA CUDA repo (for toolkit only) by creating a pinned repo file
sudo tee /etc/yum.repos.d/cuda-fedora$(rpm -E %fedora).repo >/dev/null <<'EOF'
[cuda-fedora$releasever-x86_64]
name=NVIDIA CUDA Fedora $releasever - x86_64
baseurl=https://developer.download.nvidia.com/compute/cuda/repos/fedora$releasever/x86_64/
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://developer.download.nvidia.com/compute/cuda/repos/fedora$releasever/x86_64/7fa2af80.pub
# Keep NVIDIA drivers from RPM Fusion; do not install any drivers from this repo
excludepkgs=cuda-drivers*,nvidia-driver*,xorg-x11-drv-nvidia*,kernel*
EOF

# Install CUDA toolkit and versioned nvcc; drivers remain from RPM Fusion due to excludepkgs above
# Discover nvcc package version and install alongside toolkit (example uses 13-0):
sudo dnf5 search cuda-nvcc
sudo dnf5 install -y cuda-toolkit cuda-nvcc-13-0 gcc14

# Verify nvcc
nvcc --version

# If nvcc is not on PATH, add it or set CUDA_HOME/NVCC (example for version 13.0):
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
# Or point the build directly at nvcc:
export NVCC="$CUDA_HOME/bin/nvcc"
```

#### Fedora 42 (dnf5): match CUDA toolkit to the installed driver (RPM Fusion)

If your NVIDIA driver comes from RPM Fusion (recommended) and reports a CUDA Version (via nvidia-smi) that doesn’t match the CUDA toolkit available in the NVIDIA Fedora 42 repo, install the matching toolkit from the NVIDIA archive. This avoids PTX/CUBIN incompatibilities between toolkit and driver.

- Check your driver’s CUDA Version:
```bash
nvidia-smi | grep "CUDA Version"
# Example: CUDA Version: 12.9
```

- Download the matching “local installer” repo RPM from the NVIDIA archive with a resilient downloader (NVIDIA servers can be flaky; use resume/retry):
```bash
# Example for CUDA 12.9 on Fedora 41 (works on Fedora 42 too):
curl --fail --location --retry 9999 --retry-delay 3 --retry-max-time 0 \
  --continue-at - \
  --output ~/Downloads/cuda-repo-fedora41-12-9-local-12.9.0_575.51.03-1.x86_64.rpm \
  --url https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-fedora41-12-9-local-12.9.0_575.51.03-1.x86_64.rpm
```

- Install the local repo RPM:
```bash
sudo dnf5 install -y ~/Downloads/cuda-repo-fedora41-12-9-local-12.9.0_575.51.03-1.x86_64.rpm
```

- Prevent driver packages from the NVIDIA repo (keep drivers from RPM Fusion):
```bash
# Add excludes to the generated repo file (name may vary slightly)
sudo sed -i '/^\[cuda-/,/^$/ {
  /^\s*excludepkgs=/d
}' /etc/yum.repos.d/cuda-fedora41-12-9-local.repo

echo "excludepkgs=cuda-drivers*,nvidia-driver*,xorg-x11-drv-nvidia*,kernel*" | \
  sudo tee -a /etc/yum.repos.d/cuda-fedora41-12-9-local.repo
```

- Install the matching toolkit and nvcc from the local NVIDIA repo (exclude drivers):
```bash
sudo dnf5 install -y cuda-toolkit cuda-nvcc-12-9 --exclude='cuda-drivers*'
```

- Make nvcc available to the build (12.9 example):
```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export NVCC="$CUDA_HOME/bin/nvcc"

# Verify
nvcc --version
```

- Build with the matching toolkit (example, RTX 3060/Ampere):
```bash
CUDA_ARCH=sm_86 cargo build -p miner-cli --features cuda --release
```

Notes:
- Using the Fedora 41 local repo RPM on Fedora 42 is acceptable for the CUDA user-space toolkit; we only need nvcc and toolchain, not the driver.
- Always keep the NVIDIA driver from RPM Fusion. The excludepkgs line ensures the toolkit install will not replace your driver.
- If nvcc is still not on PATH after install, set NVCC explicitly as above.

### Notes

- Ensure your user can access the GPU device nodes (e.g., part of the video group when required).
- The build script compiles any .cu under crates/engine-gpu-cuda/src/kernels into PTX and sets ENGINE_GPU_CUDA_PTX_DIR for the crate to load at runtime.
- Architecture targeting (optional, per-GPU tuning):
  - Ampere (RTX 3060): `CUDA_ARCH=sm_86 cargo build -p miner-cli --features cuda --release`
  - Ada (RTX 4090): `CUDA_ARCH=sm_89 cargo build -p miner-cli --features cuda --release`
  - CC 12.0 (e.g., RTX 5090):
    - Forward-compatible PTX: `CUDA_ARCH=compute_120 cargo build -p miner-cli --features cuda --release`
    - Tuned for SM: `CUDA_ARCH=sm_120 cargo build -p miner-cli --features cuda --release`
  - If unset, the default `sm_70` PTX will still JIT on newer GPUs (just with less arch-specific tuning).

## Configuration

The service can be configured using command-line arguments or environment variables.

| Argument          | Environment Variable | Description                                | Default       |
|-------------------|----------------------|--------------------------------------------|---------------|
| `--port <PORT>`   | `MINER_PORT`         | The port for the HTTP server to listen on. | `9833`        |
| `--workers <N>` | `MINER_WORKERS` | The number of worker threads (logical CPUs) to use for mining. | Auto-detected (leaves ~half available) |

Example:

```bash
# Run on the default port 9833 using about half of total cpu resources
../target/release/quantus-miner

# Run on a custom port with 4 workers (logical CPUs)
../target/release/quantus-miner --port 8000 --workers 4

# Equivalent using environment variables
export MINER_PORT=8000
export MINER_WORKERS=4
../target/release/quantus-miner
```

## Running

After building the service, you can run it directly from the command line:

```bash
# Run with default settings
RUST_LOG=info ../target/release/quantus-miner

# Run with a specific port and 2 workers
RUST_LOG=info ../target/release/quantus-miner --port 12345 --workers 2

# Run in debug mode
RUST_LOG=info,miner=debug ../target/release/quantus-miner --workers 4

```

The service will start and log messages to the console, indicating the port it's listening on and the number of worker threads in use.

Example output:

```
INFO  external_miner > Starting external miner service...
INFO  external_miner > Using auto-detected workers (leaving headroom): 4
INFO  external_miner > Server starting on 0.0.0.0:9833
```

## API Specification

The detailed API specification is defined using OpenAPI 3.0 and can be found in the `api/openapi.yaml` file.

This specification details all endpoints, request/response formats, and expected status codes.
You can use tools like [Swagger Editor](https://editor.swagger.io/)
or [Swagger UI](https://swagger.io/tools/swagger-ui/) to view and interact with the API definition.

## A note on workers

The miner previously used a flag named `--num-cores`. To better reflect intent, this has been replaced by `--workers`, which specifies the number of worker threads (logical CPUs). When not provided, the miner auto-detects an effective CPU set (honoring cgroup cpusets when present) and defaults to a value that leaves roughly half of the system resources available to other processes.

## API Endpoints (Summary)

* `POST /mine`: Submits a new mining job.
* `GET /result/{job_id}`: Retrieves the status and result of a specific mining job.
* `POST /cancel/{job_id}`: Cancels an ongoing mining job.

## Docker

### Quick Start

```bash
# Pull the latest image
docker pull ghcr.io/quantus-network/quantus-miner:latest

# Run with 3 workers
docker run -d \
  --name quantus-miner \
  -p 9833:9833 \
  -p 9900:9900 \
  ghcr.io/quantus-network/quantus-miner:latest \
  --engine cpu-fast --workers 3 --metrics-port 9900

# Check logs
docker logs -f quantus-miner
```

### Build from Source

```bash
docker build -t quantus-miner .
docker run -d -p 9833:9833 -p 9900:9900 quantus-miner --workers 3
```

## Implementation and PR review docs

These documents provide reviewers with the authoritative context for changes. Commits and pull requests should link to the relevant prompt/response entry.

- Authoring/process guide: agents.md

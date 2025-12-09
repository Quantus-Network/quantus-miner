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

## GPU Mining (cross-platform)

The GPU engine uses WGPU for cross-platform GPU acceleration and supports:
- **macOS**: Apple Metal (Apple Silicon & Intel Macs)  
- **Linux**: Vulkan, OpenGL
- **Windows**: DirectX 12, Vulkan

Build with GPU support:
```bash
# Build the CLI with GPU feature enabled
cargo build -p miner-cli --features gpu --release
```

At runtime, select the GPU engine:
```bash
# GPU engine will auto-detect the best available GPU backend
./target/release/quantus-miner --engine gpu --metrics-port 9919
```

The GPU engine automatically selects the optimal backend and configuration for your hardware.

### Platform-specific GPU Setup

**macOS (Apple Silicon/Intel):**
- GPU support works out-of-the-box with Metal backend
- No additional setup required

**Linux:**
- Install graphics drivers for your GPU:
```bash
# For NVIDIA (Ubuntu/Debian):
sudo ubuntu-drivers autoinstall

# For AMD (Ubuntu/Debian):  
sudo apt install mesa-vulkan-drivers

# For Intel integrated graphics:
sudo apt install intel-media-va-driver
```

**Windows:**
- Ensure you have recent graphics drivers installed
- DirectX 12 or Vulkan support recommended

### GPU Performance Tips

- GPU mining works best with large workloads - let it run on high-difficulty targets
- Monitor GPU utilization with platform tools:
  - **macOS**: `sudo powermetrics --samplers gpu_power -i 1000`  
  - **Linux**: `nvidia-smi` (NVIDIA) or `radeontop` (AMD)
  - **Windows**: Task Manager GPU tab or GPU-Z
- The GPU engine automatically optimizes for your hardware architecture

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
  --engine cpu --workers 3 --metrics-port 9900

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

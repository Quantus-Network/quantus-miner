# External Miner Service for Quantus Network

High-performance external mining service for Quantus Network with support for CPU, GPU, and hybrid CPU+GPU mining.

## Building

```bash
# CPU-only build (default)
cargo build -p miner-cli --release

# With GPU support (recommended)
cargo build -p miner-cli --release
```

The binary will be available at `target/release/quantus-miner`.

## Running

```bash
# CPU-only mining (default: auto-detected CPU cores)
./target/release/quantus-miner serve --cpu-workers 4

# GPU-only mining 
./target/release/quantus-miner serve --gpu-devices 1

# Hybrid CPU+GPU mining
./target/release/quantus-miner serve --cpu-workers 4 --gpu-devices 1

# Custom port and metrics
./target/release/quantus-miner serve --cpu-workers 2 --port 8000 --metrics-port 9900
```

## Configuration

| Argument | Environment Variable | Description | Default |
|----------|---------------------|-------------|---------|
| `--cpu-workers <N>` | `MINER_CPU_WORKERS` | Number of CPU worker threads | Auto-detect |
| `--gpu-devices <N>` | `MINER_GPU_DEVICES` | Number of GPU devices | 0 |
| `--port <PORT>` | `MINER_PORT` | HTTP API port | 9833 |
| `--metrics-port <PORT>` | `MINER_METRICS_PORT` | Prometheus metrics port | Disabled |

## GPU Mining

GPU support uses WGPU for cross-platform acceleration:

- **macOS**: Metal backend (Apple Silicon & Intel)
- **Linux**: Vulkan/OpenGL backends
- **Windows**: DirectX 12/Vulkan backends

### Setup

**Build with GPU support:**
```bash
cargo build -p miner-cli --release
```

**Platform requirements:**
- **macOS**: Works out-of-the-box
- **Linux**: Install GPU drivers (`nvidia-driver`, `mesa-vulkan-drivers`)
- **Windows**: Ensure recent graphics drivers are installed

### Performance Monitoring

- **macOS**: `sudo powermetrics --samplers gpu_power -i 1000`
- **Linux**: `nvidia-smi` (NVIDIA) or `radeontop` (AMD)  
- **Windows**: Task Manager GPU tab

## Examples

```bash
# CPU mining with 8 workers
./target/release/quantus-miner serve --cpu-workers 8

# Pure GPU mining
./target/release/quantus-miner serve --gpu-devices 1

# Hybrid mining: 4 CPU + 1 GPU workers
./target/release/quantus-miner serve --cpu-workers 4 --gpu-devices 1

# With verbose logging
RUST_LOG=debug ./target/release/quantus-miner serve --cpu-workers 2 --gpu-devices 1

# Production setup with metrics
./target/release/quantus-miner serve \
  --cpu-workers 6 \
  --gpu-devices 1 \
  --port 9833 \
  --metrics-port 9900
```

## API Endpoints

- `POST /mine`: Submit mining job
- `GET /result/{job_id}`: Get job status/result
- `POST /cancel/{job_id}`: Cancel job

Full API specification: `api/openapi.yaml`

## Docker

```bash
# Quick start
docker pull ghcr.io/quantus-network/quantus-miner:latest
docker run -d -p 9833:9833 -p 9900:9900 \
  ghcr.io/quantus-network/quantus-miner:latest \
  --cpu-workers 4 --metrics-port 9900

# Build from source
docker build -t quantus-miner .
docker run -d -p 9833:9833 quantus-miner serve --cpu-workers 4
```

## Benchmarking

```bash
# Benchmark CPU performance
./target/release/quantus-miner benchmark --cpu-workers 8 --duration 30

# Benchmark GPU performance  
./target/release/quantus-miner benchmark --gpu-devices 1 --duration 30

# Benchmark hybrid performance
./target/release/quantus-miner benchmark --cpu-workers 4 --gpu-devices 1 --duration 30
```

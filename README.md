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

The node requires a shared auth token and TLS cert pin. Both live under the
node's chain config dir (`<base-path>/chains/<chain>/`):

- `miner-auth-token` — shared secret (**not** logged by the node; read this file)
- `miner-tls-cert-sha256` — SHA-256 of the miner TLS cert (also printed in node logs)

```bash
# Preferred: mount/read the node's chain config files
./target/release/quantus-miner serve \
  --node-addr 127.0.0.1:9833 \
  --auth-token-file /path/to/miner-auth-token \
  --tls-cert-sha256-file /path/to/miner-tls-cert-sha256 \
  --cpu-workers 4

# Or pass values directly (token from miner-auth-token; fingerprint also in node logs)
./target/release/quantus-miner serve \
  --node-addr 127.0.0.1:9833 \
  --auth-token <TOKEN> \
  --tls-cert-sha256 <FINGERPRINT> \
  --gpu-devices 1

# Hybrid CPU+GPU mining
./target/release/quantus-miner serve \
  --node-addr 127.0.0.1:9833 \
  --auth-token-file /path/to/miner-auth-token \
  --tls-cert-sha256-file /path/to/miner-tls-cert-sha256 \
  --cpu-workers 4 \
  --gpu-devices 1
```

## Configuration

| Argument | Environment Variable | Description | Default |
|----------|---------------------|-------------|---------|
| `--node-addr <ADDR>` | `MINER_NODE_ADDR` | Node address to connect to | `127.0.0.1:9833` |
| `--auth-token <TOKEN>` | `MINER_AUTH_TOKEN` | Shared secret from the node's `miner-auth-token` file (not logged) | required |
| `--auth-token-file <PATH>` | `MINER_AUTH_TOKEN_FILE` | Read the shared secret from a file (preferred) | — |
| `--tls-cert-sha256 <HEX>` | `MINER_TLS_CERT_SHA256` | SHA-256 of the node's miner TLS cert (`miner-tls-cert-sha256` / node logs) | required |
| `--tls-cert-sha256-file <PATH>` | `MINER_TLS_CERT_SHA256_FILE` | Read the TLS cert fingerprint from a file | — |
| `--cpu-workers <N>` | `MINER_CPU_WORKERS` | Number of CPU worker threads | Auto-detect |
| `--gpu-devices <N>` | `MINER_GPU_DEVICES` | Number of GPU devices | Auto-detect |
| `--gpu-batch-size <N>` | `MINER_GPU_BATCH_SIZE` | GPU batch size in nonces | 1000000 |
| `--cpu-batch-size <N>` | `MINER_CPU_BATCH_SIZE` | CPU batch size in hashes | 10000 |
| `--gpu-throttle-ms <MS>` | `MINER_GPU_THROTTLE_MS` | Sleep duration (ms) between GPU batches | 0 |
| `--metrics-port <PORT>` | `MINER_METRICS_PORT` | Prometheus metrics port | 9900 |

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

All `serve` examples need the auth token and TLS pin (files or inline values).

```bash
# CPU mining with 8 workers
./target/release/quantus-miner serve \
  --auth-token-file /path/to/miner-auth-token \
  --tls-cert-sha256-file /path/to/miner-tls-cert-sha256 \
  --cpu-workers 8

# Pure GPU mining
./target/release/quantus-miner serve \
  --auth-token-file /path/to/miner-auth-token \
  --tls-cert-sha256-file /path/to/miner-tls-cert-sha256 \
  --gpu-devices 1

# GPU mining with throttle (reduce GPU utilization)
./target/release/quantus-miner serve \
  --auth-token-file /path/to/miner-auth-token \
  --tls-cert-sha256-file /path/to/miner-tls-cert-sha256 \
  --gpu-devices 1 --gpu-throttle-ms 50

# Hybrid mining: 4 CPU + 1 GPU workers
./target/release/quantus-miner serve \
  --auth-token-file /path/to/miner-auth-token \
  --tls-cert-sha256-file /path/to/miner-tls-cert-sha256 \
  --cpu-workers 4 --gpu-devices 1

# With verbose logging
RUST_LOG=debug ./target/release/quantus-miner serve \
  --auth-token-file /path/to/miner-auth-token \
  --tls-cert-sha256-file /path/to/miner-tls-cert-sha256 \
  --cpu-workers 2 --gpu-devices 1

# Production setup with metrics
./target/release/quantus-miner serve \
  --node-addr 127.0.0.1:9833 \
  --auth-token-file /path/to/miner-auth-token \
  --tls-cert-sha256-file /path/to/miner-tls-cert-sha256 \
  --cpu-workers 6 \
  --gpu-devices 1 \
  --metrics-port 9900
```

## Protocol

The miner uses a QUIC-based protocol for communication with the node:

- **Transport**: QUIC with TLS 1.3 (self-signed certificate, pinned by SHA-256)
- **Auth**: `Ready { token }` must match the node's `miner-auth-token`
- **ALPN**: `quantus-miner/2`
- **Port**: 9833 (default)
- **Messages**: `Ready` (miner→node), `NewJob` (node→miner), `JobResult` (miner→node)

For full protocol specification, see the node's `MINING.md`.

## Benchmarking

```bash
# Benchmark CPU performance
./target/release/quantus-miner benchmark --cpu-workers 8 --duration 30

# Benchmark GPU performance  
./target/release/quantus-miner benchmark --gpu-devices 1 --duration 30

# Benchmark hybrid performance
./target/release/quantus-miner benchmark --cpu-workers 4 --gpu-devices 1 --duration 30
```

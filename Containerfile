# syntax=docker/dockerfile:1.7

# Containerized CUDA build for Quantus Miner.
# Supports matrix builds via:
#   - CUDA_TAG: CUDA toolkit tag (e.g., 12.9.0, 13.0.0)
#   - SM:       GPU SM architecture (e.g., 86 for RTX 3060, 89 for RTX 4090, 120 for CC 12.0)
#
# Examples (podman or docker):
#   podman build -t quantus-miner:cuda12.9-sm86 \
#     --build-arg CUDA_TAG=12.9.0 --build-arg SM=86 -f Containerfile .
#
#   podman build -t quantus-miner:cuda13.0-sm89 \
#     --build-arg CUDA_TAG=13.0.0 --build-arg SM=89 -f Containerfile .
#
# The resulting image contains /usr/local/bin/quantus-miner.
# You can extract the binary for distribution:
#   cid=$(podman create quantus-miner:cuda12.9-sm86)
#   podman cp "$cid":/usr/local/bin/quantus-miner ./quantus-miner-cuda-12.9-sm-86
#   podman rm "$cid"

################################################################################
# Stage 1: Build with NVIDIA CUDA devel + Rust
################################################################################
ARG CUDA_TAG=12.9.0
FROM nvcr.io/nvidia/cuda:${CUDA_TAG}-devel-ubuntu22.04 AS builder

# SM arch for the device code (86=Ampere 3060, 89=Ada 4090, 120=CC 12.0/5090).
ARG SM=86

# System deps for Rust toolchain and linking
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl ca-certificates pkg-config build-essential \
      libssl-dev git clang cmake && \
    rm -rf /var/lib/apt/lists/*

# Install Rust (stable)
ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/cargo/bin:/usr/local/cuda/bin:${PATH}
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable && \
    rustc --version && cargo --version

WORKDIR /src

# (Optional) set CUDA arch for build.rs (normalized internally to compute_XX/sm_XX)
ENV CUDA_ARCH=sm_${SM}

# (Optional) permit newer host compilers inside future images if needed
# ENV MINER_CUDA_ALLOW_UNSUPPORTED_COMPILER=1
# (Optional) select specific host C++ compiler for nvcc (only if needed)
# ENV MINER_NVCC_CCBIN=/usr/bin/g++-14

# Copy manifests first to leverage build cache
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

# Build the miner with CUDA feature (kernels compiled & embedded by build.rs)
# Cache Cargo registry and target dirs for faster matrix builds.
RUN --mount=type=cache,target=/opt/cargo/registry \
    --mount=type=cache,target=/src/target \
    set -eux; \
    echo "CUDA version:" && nvcc --version; \
    echo "Building for SM=${SM}, CUDA_TAG=${CUDA_TAG} (CUDA_ARCH=${CUDA_ARCH})"; \
    cargo build -p miner-cli --features cuda --release; \
    strip /src/target/release/quantus-miner || true

################################################################################
# Stage 2: Minimal runtime image with the miner binary
################################################################################
FROM ubuntu:22.04 AS dist

# Ensure certificates exist (for any outbound metrics/HTTP if used)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /src/target/release/quantus-miner /usr/local/bin/quantus-miner

# Default entrypoint; override with args as needed
ENTRYPOINT ["/usr/local/bin/quantus-miner"]

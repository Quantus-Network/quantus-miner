# syntax=docker/dockerfile:1

############################
# Builder stage
############################
FROM rust:1.85-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy workspace files
COPY Cargo.toml Cargo.lock rust-toolchain taplo.toml ./
COPY crates ./crates
COPY tests ./tests

# Build the miner-cli in release mode
RUN cargo build --release -p miner-cli --locked

# Strip debug symbols to reduce binary size
RUN strip target/release/quantus-miner || true

############################
# Runtime-only stage
############################
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy binary from builder stage
COPY --from=builder /build/target/release/quantus-miner /usr/local/bin/quantus-miner

# Expose miner API port and metrics port
EXPOSE 9833 9900

# Run as unprivileged user
RUN useradd --system --uid 10001 quantus
USER 10001:10001

# Default working directory
WORKDIR /data

# Start the miner
ENTRYPOINT ["quantus-miner"]


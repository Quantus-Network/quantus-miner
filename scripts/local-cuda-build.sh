#!/usr/bin/env bash
set -euo pipefail

declare -A ARCHES=([86]="Ampere (RTX 3060)" [89]="Ada (RTX 4090)" [120]="CC 12.0 (RTX 5090)")
CUDA_VERSIONS=(13.0)  # build only the version matching your driver

build_one() {
  local cuda_ver="$1" sm="$2"
  export MINER_NVCC_CCBIN=/usr/bin/g++-14
  export CUDA_HOME="/usr/local/cuda-${cuda_ver}"
  export NVCC="${CUDA_HOME}/bin/nvcc"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export CARGO_TARGET_DIR="target/cuda-${cuda_ver}"

  if [[ ! -x "${NVCC}" ]]; then
    echo "NVCC not found for CUDA ${cuda_ver}: ${NVCC}" >&2; return 1
  fi
  echo "==== Building CUDA ${cuda_ver}, sm_${sm} (${ARCHES[$sm]}) ===="
  "${NVCC}" --version || true

  # Force fresh engine-gpu-cuda OUT_DIR for this (version,arch)
  cargo clean -p engine-gpu-cuda

  CUDA_ARCH="sm_${sm}" cargo build -p miner-cli --features cuda --release

  local outdir
  outdir="$(ls -d "${CARGO_TARGET_DIR}"/release/build/engine-gpu-cuda-*/out | head -n1 || true)"
  [[ -n "${outdir}" ]] || { echo "ERROR: OUT_DIR not found"; return 1; }
  [[ -f "${outdir}/qpow_kernel.cubin" ]] || { echo "ERROR: missing CUBIN in ${outdir}"; return 1; }
  [[ -f "${outdir}/qpow_kernel.ptx" ]] || { echo "WARN: PTX missing (CUBIN present, OK)"; }

  local src="${CARGO_TARGET_DIR}/release/quantus-miner"
  local dst="target/release/quantus-miner-cuda-${cuda_ver}-sm-${sm}"
  cp -f "${src}" "${dst}"
  ls -l "${dst}"
  sha256sum "${dst}"
}

# Stop services if running
#for unit in resonance-{node,miner}.service; do
#  systemctl is-active --quiet "${unit}" && sudo systemctl stop "${unit}"
#done

# Build only the driver-matching toolkit and the 3060 arch
build_one "13.0" "120"

# Install the built binary
#sudo install -o root -g root -m 0755 target/release/quantus-miner-cuda-12.9-sm-86 /usr/local/bin/
#echo "Installed /usr/local/bin/quantus-miner-cuda-12.9-sm-86"

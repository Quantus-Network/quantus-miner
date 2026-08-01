#!/usr/bin/env bash
# Run on a GPU host (e.g. RunPod): fetch release binaries, start --dev node +
# GPU miner, sample Prometheus into results.csv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-/workspace/quantus-gpu-bench}"
BIN_DIR="${WORK_DIR}/bin"
RUN_DIR="${WORK_DIR}/.run"
RESULTS_CSV="${WORK_DIR}/results.csv"

MINER_VERSION="${MINER_VERSION:-v3.3.1}"
MINER_URL="${MINER_URL:-https://github.com/Quantus-Network/quantus-miner/releases/download/${MINER_VERSION}/quantus-miner-linux-x86_64}"
NODE_URL="${NODE_URL:-}" # optional override; default = latest chain release

PROVIDER="${PROVIDER:-runpod}"
COST_PER_HOUR="${COST_PER_HOUR:-}"
DURATION="${DURATION:-60}"
GPU_DEVICES="${GPU_DEVICES:-1}"
METRICS_PORT="${METRICS_PORT:-9900}"
MINER_LISTEN_PORT="${MINER_LISTEN_PORT:-9833}"
WARMUP_SECONDS="${WARMUP_SECONDS:-30}"
NOTES="${NOTES:-}"

usage() {
  cat <<'EOF'
Usage: ./remote-run.sh [options]

  --provider NAME         cloud_provider column (default: runpod)
  --cost-per-hour USD     required for efficiency column
  --duration SECONDS      Prometheus sample window (default: 60)
  --gpu-devices N         default 1
  --warmup SECONDS        wait after miner start before sampling (default: 30)
  --notes TEXT
  --miner-url URL         override miner binary URL
  --help

Downloads linux x86_64 release binaries, runs quantus-node --dev with
--miner-listen-port, runs quantus-miner against 127.0.0.1, then samples
:9900/metrics into results.csv (via record.sh --live).
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --provider) PROVIDER="${2:-}"; shift 2 ;;
    --cost-per-hour) COST_PER_HOUR="${2:-}"; shift 2 ;;
    --duration) DURATION="${2:-}"; shift 2 ;;
    --gpu-devices) GPU_DEVICES="${2:-}"; shift 2 ;;
    --warmup) WARMUP_SECONDS="${2:-}"; shift 2 ;;
    --notes) NOTES="${2:-}"; shift 2 ;;
    --miner-url) MINER_URL="${2:-}"; shift 2 ;;
    --node-url) NODE_URL="${2:-}"; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${COST_PER_HOUR}" ]]; then
  echo "error: --cost-per-hour is required" >&2
  exit 1
fi

require_cmd curl
require_cmd tar
require_cmd nvidia-smi
if ! nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi failed" >&2
  exit 1
fi

mkdir -p "${BIN_DIR}" "${RUN_DIR}"
cd "${WORK_DIR}"

# Copy record.sh next to us if we're invoked from gpu-bench/
if [[ -f "${SCRIPT_DIR}/record.sh" && ! -f "${WORK_DIR}/record.sh" ]]; then
  cp "${SCRIPT_DIR}/record.sh" "${WORK_DIR}/record.sh"
  chmod +x "${WORK_DIR}/record.sh"
fi
if [[ ! -x "${WORK_DIR}/record.sh" ]]; then
  echo "error: record.sh not found next to remote-run.sh" >&2
  exit 1
fi

# Exit code for sweep: host has CUDA but no usable NVIDIA Vulkan (retry new pod).
EXIT_COMPUTE_ONLY=42

find_libglx_nvidia() {
  local candidate
  for candidate in \
    /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 \
    /usr/lib64/libGLX_nvidia.so.0 \
    /usr/lib/libGLX_nvidia.so.0 \
    /usr/local/nvidia/lib64/libGLX_nvidia.so.0 \
    "${WORK_DIR}/nvidia-gl/libGLX_nvidia.so.0"; do
    if [[ -e "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  ldconfig -p 2>/dev/null | awk '/libGLX_nvidia\.so\.0/ { print $NF; exit }' || true
}

# Many Community hosts only mount compute libs. Install matching graphics
# userspace via apt, then (if needed) extract from the NVIDIA .run installer.
install_nvidia_gl_userspace() {
  local ver major dest
  ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
    | head -n 1 | tr -d '[:space:]')"
  if [[ -z "${ver}" ]]; then
    return 1
  fi
  major="${ver%%.*}"
  dest="${WORK_DIR}/nvidia-gl"
  mkdir -p "${dest}"

  if command -v apt-get >/dev/null 2>&1; then
    echo "Trying apt libnvidia-gl-${major} (host driver ${ver}) ..." >&2
    apt-get install -y -qq "libnvidia-gl-${major}" >/dev/null 2>&1 || true
    if [[ -n "$(find_libglx_nvidia)" ]]; then
      return 0
    fi
  fi

  if [[ -e "${dest}/libGLX_nvidia.so.0" ]]; then
    export LD_LIBRARY_PATH="${dest}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    return 0
  fi

  echo "Extracting NVIDIA ${ver} userspace libs (compute-only host) ..." >&2
  local tmp runfile url
  tmp="$(mktemp -d)"
  runfile="${tmp}/NVIDIA.run"
  # GeForce/pro first; Tesla URL as fallback for datacenter hosts.
  for url in \
    "https://download.nvidia.com/XFree86/Linux-x86_64/${ver}/NVIDIA-Linux-x86_64-${ver}.run" \
    "https://us.download.nvidia.com/XFree86/Linux-x86_64/${ver}/NVIDIA-Linux-x86_64-${ver}.run" \
    "https://us.download.nvidia.com/tesla/${ver}/NVIDIA-Linux-x86_64-${ver}.run"; do
    if curl -fL --connect-timeout 20 --max-time 600 "${url}" -o "${runfile}"; then
      break
    fi
    rm -f "${runfile}"
  done
  if [[ ! -f "${runfile}" ]]; then
    echo "error: could not download NVIDIA ${ver} .run installer" >&2
    rm -rf "${tmp}"
    return 1
  fi

  chmod +x "${runfile}"
  # Extract only — do not install kernel modules into the container.
  if ! sh "${runfile}" --extract-only --target "${tmp}/extract" >/tmp/nvidia-extract.log 2>&1; then
    echo "error: NVIDIA .run extract failed; log:" >&2
    tail -n 40 /tmp/nvidia-extract.log >&2 || true
    rm -rf "${tmp}"
    return 1
  fi

  local f
  for f in \
    libGLX_nvidia.so.* \
    libEGL_nvidia.so.* \
    libnvidia-glcore.so.* \
    libnvidia-glsi.so.* \
    libnvidia-tls.so.* \
    libnvidia-glvkspirv.so.* \
    libnvidia-gpucomp.so.* \
    libnvidia-rtcore.so.* \
    libnvoptix.so.*; do
    # shellcheck disable=SC2086
    cp -n ${tmp}/extract/${f} "${dest}/" 2>/dev/null || true
  done
  # Stable SONAME links expected by the ICD.
  local so
  for so in libGLX_nvidia.so libEGL_nvidia.so; do
    if [[ ! -e "${dest}/${so}.0" ]]; then
      local real
      real="$(ls -1 "${dest}/${so}".* 2>/dev/null | grep -v '\.so$' | sort -V | tail -n 1 || true)"
      if [[ -n "${real}" ]]; then
        ln -sfn "$(basename "${real}")" "${dest}/${so}.0"
      fi
    fi
  done

  if [[ -f "${tmp}/extract/nvidia_icd.json" ]]; then
    mkdir -p /usr/share/vulkan/icd.d
    # Rewrite library_path to our extracted lib if needed later.
    cp -f "${tmp}/extract/nvidia_icd.json" /usr/share/vulkan/icd.d/nvidia_icd.json || true
  fi

  echo "${dest}" >"${dest}/.ld_path"
  export LD_LIBRARY_PATH="${dest}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "${dest}" >>/etc/ld.so.conf.d/nvidia-gl-bench.conf 2>/dev/null || true
  ldconfig 2>/dev/null || true

  rm -rf "${tmp}"
  if [[ -e "${dest}/libGLX_nvidia.so.0" ]] || [[ -n "$(find_libglx_nvidia)" ]]; then
    echo "NVIDIA GL userspace ready in ${dest}" >&2
    return 0
  fi
  return 1
}

# WGPU uses Vulkan. RunPod/CUDA images often only expose compute; Mesa's
# llvmpipe then becomes the only adapter and the miner refuses it.
ensure_nvidia_vulkan() {
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq >/dev/null 2>&1 || true
    # libegl1/libxext6 are required for libGLX_nvidia to export vk_icd* in
    # headless containers (without them: ERROR_INCOMPATIBLE_DRIVER).
    # Do NOT install mesa-vulkan-drivers (llvmpipe distracts / can win).
    apt-get install -y -qq \
      curl ca-certificates \
      libvulkan1 vulkan-tools \
      libegl1 libxext6 libgl1 \
      >/dev/null 2>&1 || \
      apt-get install -y -qq curl ca-certificates libvulkan1 libegl1 libxext6 >/dev/null 2>&1 || true
  fi
  mkdir -p /tmp/runtime-root
  chmod 700 /tmp/runtime-root 2>/dev/null || true
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-root}"

  local lib
  lib="$(find_libglx_nvidia)"
  if [[ -z "${lib}" || ! -e "${lib}" ]]; then
    echo "libGLX_nvidia.so.0 not mounted (compute-only host); installing userspace ..." >&2
    if ! install_nvidia_gl_userspace; then
      echo "error: NVIDIA Vulkan/GL library missing (libGLX_nvidia.so.0)." >&2
      echo "exit ${EXIT_COMPUTE_ONLY}: sweep should retry on another host" >&2
      exit "${EXIT_COMPUTE_ONLY}"
    fi
    lib="$(find_libglx_nvidia)"
  fi

  if [[ -z "${lib}" || ! -e "${lib}" ]]; then
    echo "error: still no libGLX_nvidia.so.0 after userspace install" >&2
    exit "${EXIT_COMPUTE_ONLY}"
  fi

  # Prefer extracted dir on LD_LIBRARY_PATH when we installed locally.
  if [[ -d "${WORK_DIR}/nvidia-gl" ]]; then
    export LD_LIBRARY_PATH="${WORK_DIR}/nvidia-gl${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi

  local icd_dir="/usr/share/vulkan/icd.d"
  local icd="${icd_dir}/nvidia_icd.json"
  mkdir -p "${icd_dir}" /etc/vulkan/icd.d
  cat >"${icd}" <<EOF
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "${lib}",
        "api_version": "1.3"
    }
}
EOF
  cp -f "${icd}" /etc/vulkan/icd.d/nvidia_icd.json 2>/dev/null || true
  echo "NVIDIA Vulkan ICD: ${icd} -> ${lib}" >&2

  # Prefer NVIDIA ICD only (ignore any Mesa CPU ICDs that may already exist).
  export VK_ICD_FILENAMES="${icd}"
  export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
  export __GLX_VENDOR_LIBRARY_NAME=nvidia

  if command -v vulkaninfo >/dev/null 2>&1; then
    if ! vulkaninfo --summary 2>/dev/null | grep -qiE 'NVIDIA|GeForce|Tesla|Quadro|RTX|A100|L4|L40'; then
      echo "error: vulkaninfo sees no NVIDIA GPU (only software/CPU adapters?)." >&2
      vulkaninfo --summary 2>&1 | tail -n 40 >&2 || true
      echo "exit ${EXIT_COMPUTE_ONLY}: bad Vulkan host — sweep should retry" >&2
      exit "${EXIT_COMPUTE_ONLY}"
    fi
    echo "Vulkan NVIDIA adapter OK" >&2
  else
    echo "warning: vulkan-tools not installed; skipping vulkaninfo check" >&2
  fi
}

download_miner() {
  local dest="${BIN_DIR}/quantus-miner"
  if [[ -x "${dest}" ]]; then
    echo "Using existing ${dest}" >&2
    echo "${dest}"
    return
  fi
  echo "Downloading miner: ${MINER_URL}" >&2
  curl -fL "${MINER_URL}" -o "${dest}"
  chmod +x "${dest}"
  echo "${dest}"
}

download_node() {
  local dest="${BIN_DIR}/quantus-node"
  if [[ -x "${dest}" ]]; then
    echo "Using existing ${dest}" >&2
    echo "${dest}"
    return
  fi

  local url="${NODE_URL}"
  if [[ -z "${url}" ]]; then
    echo "Resolving latest quantus-node release ..." >&2
    local release_json tag
    release_json="$(curl -fsSL https://api.github.com/repos/Quantus-Network/chain/releases/latest)"
    tag="$(echo "${release_json}" | grep -o '"tag_name": "[^"]*"' | head -n 1 | cut -d'"' -f4)"
    url="https://github.com/Quantus-Network/chain/releases/download/${tag}/quantus-node-${tag}-x86_64-unknown-linux-gnu.tar.gz"
  fi

  echo "Downloading node: ${url}" >&2
  local tmp
  tmp="$(mktemp -d)"
  curl -fL "${url}" -o "${tmp}/node.tar.gz"
  tar -xzf "${tmp}/node.tar.gz" -C "${tmp}"
  if [[ ! -f "${tmp}/quantus-node" ]]; then
    echo "error: archive missing quantus-node" >&2
    exit 1
  fi
  mv "${tmp}/quantus-node" "${dest}"
  chmod +x "${dest}"
  rm -rf "${tmp}"
  echo "${dest}"
}

stop_pidfile() {
  local pidfile="$1"
  if [[ ! -f "${pidfile}" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "${pidfile}")"
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    sleep 1
    kill -9 "${pid}" 2>/dev/null || true
  fi
  rm -f "${pidfile}"
}

cleanup() {
  stop_pidfile "${RUN_DIR}/miner.pid"
  stop_pidfile "${RUN_DIR}/node.pid"
}
trap cleanup EXIT

ensure_nvidia_vulkan
NODE_BIN="$(download_node)"
MINER_BIN="$(download_miner)"

# Fail fast with a clear hint if the host glibc is too old for the release binary.
if ! "${NODE_BIN}" --version >/tmp/qnode-ver.txt 2>/tmp/qnode-ver.err; then
  if grep -q 'GLIBC_' /tmp/qnode-ver.err 2>/dev/null; then
    echo "error: quantus-node needs a newer glibc than this image provides:" >&2
    cat /tmp/qnode-ver.err >&2
    echo "Use an Ubuntu 24.04+ image, e.g. IMAGE_NAME=runpod/base:1.1.0-cuda1281-ubuntu2404" >&2
    exit 1
  fi
fi

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || nvidia-smi || true

mkdir -p "${RUN_DIR}/node-data"
echo "Starting quantus-node --dev ..."
nohup "${NODE_BIN}" \
  --dev \
  --base-path "${RUN_DIR}/node-data" \
  --rpc-port 9944 \
  --prometheus-port 9615 \
  --prometheus-external \
  --miner-listen-port "${MINER_LISTEN_PORT}" \
  --rpc-cors all \
  >"${RUN_DIR}/node.log" 2>&1 &
echo $! >"${RUN_DIR}/node.pid"
sleep 3

if ! kill -0 "$(cat "${RUN_DIR}/node.pid")" 2>/dev/null; then
  echo "error: node failed to start; log:" >&2
  tail -n 80 "${RUN_DIR}/node.log" >&2 || true
  if grep -q 'GLIBC_' "${RUN_DIR}/node.log" 2>/dev/null; then
    echo "hint: IMAGE_NAME=runpod/base:1.1.0-cuda1281-ubuntu2404 (GLIBC 2.38+)" >&2
  fi
  exit 1
fi

echo "Starting quantus-miner (gpu-devices=${GPU_DEVICES}) ..."
# Keep VK_ICD_FILENAMES / NVIDIA caps in the miner environment.
nohup env \
  VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-}" \
  NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}" \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-root}" \
  "${MINER_BIN}" serve \
  --node-addr "127.0.0.1:${MINER_LISTEN_PORT}" \
  --gpu-devices "${GPU_DEVICES}" \
  --cpu-workers 0 \
  --metrics-port "${METRICS_PORT}" \
  >"${RUN_DIR}/miner.log" 2>&1 &
echo $! >"${RUN_DIR}/miner.pid"
sleep 3

if ! kill -0 "$(cat "${RUN_DIR}/miner.pid")" 2>/dev/null; then
  echo "error: miner failed to start; log:" >&2
  tail -n 80 "${RUN_DIR}/miner.log" >&2 || true
  if grep -qiE 'No usable GPU|llvmpipe|Vulkan' "${RUN_DIR}/miner.log" 2>/dev/null; then
    echo "hint: need NVIDIA Vulkan ICD + NVIDIA_DRIVER_CAPABILITIES=all (not Mesa llvmpipe)" >&2
  fi
  exit 1
fi

# Wait for metrics + non-zero hashrate (jobs from --dev)
echo "Warming up ${WARMUP_SECONDS}s / waiting for hashrate ..."
ready=0
for ((i = 0; i < WARMUP_SECONDS + 60; i++)); do
  if ! kill -0 "$(cat "${RUN_DIR}/miner.pid")" 2>/dev/null; then
    echo "error: miner exited during warmup; log:" >&2
    tail -n 80 "${RUN_DIR}/miner.log" >&2 || true
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${METRICS_PORT}/metrics" \
    | awk '/^miner_gpu_hash_rate[[:space:]]/ { if ($2+0 > 0) found=1 } END { exit !found }'; then
    ready=1
    break
  fi
  sleep 1
done

if [[ "${ready}" -ne 1 ]]; then
  echo "warning: no positive miner_gpu_hash_rate yet; sampling anyway" >&2
  echo "--- miner log ---" >&2
  tail -n 40 "${RUN_DIR}/miner.log" >&2 || true
  echo "--- node log ---" >&2
  tail -n 40 "${RUN_DIR}/node.log" >&2 || true
fi

# Point record.sh at our run dir / metrics
export METRICS_PORT
mkdir -p "${SCRIPT_DIR}/.run" 2>/dev/null || true
# record.sh reads .run/metrics.port relative to its own dir — use WORK_DIR copy
echo "${METRICS_PORT}" >"${WORK_DIR}/.run/metrics.port"

NOTE_ARGS=()
if [[ -n "${NOTES}" ]]; then
  NOTE_ARGS=(--notes "${NOTES}")
fi

cd "${WORK_DIR}"
./record.sh --live \
  --provider "${PROVIDER}" \
  --cost-per-hour "${COST_PER_HOUR}" \
  --duration "${DURATION}" \
  "${NOTE_ARGS[@]}"

echo "Done. Results: ${RESULTS_CSV}"
cat "${RESULTS_CSV}"
# Leave processes up until EXIT trap on script end — stop after record so CSV is final

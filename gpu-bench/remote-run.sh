#!/usr/bin/env bash
# Run on a GPU host (e.g. RunPod): build miner from git, ensure NVIDIA Vulkan,
# run quantus-miner benchmark across batch sizes → results.csv (no node).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-/workspace/quantus-gpu-bench}"
BIN_DIR="${WORK_DIR}/bin"
RUN_DIR="${WORK_DIR}/.run"
RESULTS_CSV="${WORK_DIR}/results.csv"

# Miner: default builds from git (iterate without re-releasing).
# MINER_SOURCE=release still downloads a GitHub release binary.
MINER_SOURCE="${MINER_SOURCE:-git}"
MINER_REPO="${MINER_REPO:-https://github.com/Quantus-Network/quantus-miner.git}"
MINER_BRANCH="${MINER_BRANCH:-illuzen/gpu-bench}"
MINER_VERSION="${MINER_VERSION:-v3.3.1}"
MINER_URL="${MINER_URL:-https://github.com/Quantus-Network/quantus-miner/releases/download/${MINER_VERSION}/quantus-miner-linux-x86_64}"
FORCE_MINER_BUILD="${FORCE_MINER_BUILD:-0}"

PROVIDER="${PROVIDER:-runpod}"
COST_PER_HOUR="${COST_PER_HOUR:-}"
DURATION="${DURATION:-30}"
GPU_DEVICES="${GPU_DEVICES:-1}"
# Downward + 1M + one larger step. Override with --batch-sizes or BATCH_SIZES.
BATCH_SIZES="${BATCH_SIZES:-262144 524288 1000000 4194304}"
# Simulate NewJob churn (seconds). 0 = sustained peak H/s (no job switches).
JOB_INTERVAL="${JOB_INTERVAL:-2}"
# ±fraction of JOB_INTERVAL (uniform). 0 = metronomic.
JOB_JITTER="${JOB_JITTER:-0.2}"
# Optional decimal difficulty for job sim (miner default: max / cancel-only).
DIFFICULTY="${DIFFICULTY:-}"
NOTES="${NOTES:-}"

usage() {
  cat <<'EOF'
Usage: ./remote-run.sh [options]

  --provider NAME         cloud_provider column (default: runpod)
  --cost-per-hour USD     required for hash_per_dollar column
  --duration SECONDS      per-batch-size benchmark window (default: 30)
  --gpu-devices N         default 1
  --batch-sizes "N N N"   GPU batch sizes (default: 256K 512K 1M 4M)
  --job-interval SECONDS  simulated NewJob period (default: 2; 0 = sustained)
  --job-jitter FRAC       ±fraction of job-interval (default 0.2; 0 = metronomic)
  --difficulty DEC|max    PoW difficulty for job sim (miner default: max)
  --notes TEXT
  --miner-branch REF      git branch/tag/commit to build (default: illuzen/gpu-bench)
  --miner-repo URL        git remote (default: Quantus-Network/quantus-miner)
  --miner-source git|release
                          git = clone+cargo build (default); release = binary URL
  --miner-url URL         release binary URL (with --miner-source release)
  --force-miner-build     rebuild even if binary exists
  --help

Builds quantus-miner from git (or downloads a release), sets up NVIDIA Vulkan,
runs `quantus-miner benchmark` for each batch size (no node), appends rows to
results.csv (notes include batch=N; job_interval=… when set).

Env: MINER_SOURCE, MINER_REPO, MINER_BRANCH, FORCE_MINER_BUILD, MINER_URL,
     BATCH_SIZES, JOB_INTERVAL, JOB_JITTER, DIFFICULTY
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
    --batch-sizes) BATCH_SIZES="${2:-}"; shift 2 ;;
    --job-interval) JOB_INTERVAL="${2:-}"; shift 2 ;;
    --job-jitter) JOB_JITTER="${2:-}"; shift 2 ;;
    --difficulty) DIFFICULTY="${2:-}"; shift 2 ;;
    --notes) NOTES="${2:-}"; shift 2 ;;
    --miner-branch) MINER_BRANCH="${2:-}"; shift 2 ;;
    --miner-repo) MINER_REPO="${2:-}"; shift 2 ;;
    --miner-source) MINER_SOURCE="${2:-}"; shift 2 ;;
    --miner-url) MINER_URL="${2:-}"; MINER_SOURCE="release"; shift 2 ;;
    --force-miner-build) FORCE_MINER_BUILD=1; shift ;;
    # Deprecated no-ops (node path removed)
    --warmup | --node-url) shift 2 ;;
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
  # Prefer exact .run extract over apt (apt major-version packages often
  # fail vkCreateInstance against the host kernel module).
  local candidate
  for candidate in \
    "${WORK_DIR}/nvidia-gl/libGLX_nvidia.so.0" \
    /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 \
    /usr/lib64/libGLX_nvidia.so.0 \
    /usr/lib/libGLX_nvidia.so.0 \
    /usr/local/nvidia/lib64/libGLX_nvidia.so.0; do
    if [[ -e "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  ldconfig -p 2>/dev/null | awk '/libGLX_nvidia\.so\.0/ { print $NF; exit }' || true
}

host_driver_version() {
  nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
    | head -n 1 | tr -d '[:space:]'
}

vulkan_has_nvidia() {
  command -v vulkaninfo >/dev/null 2>&1 || return 1
  vulkaninfo --summary 2>/dev/null | grep -qiE 'NVIDIA|GeForce|Tesla|Quadro|RTX|A100|L4|L40'
}

write_nvidia_icd() {
  local lib="$1"
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
  export VK_ICD_FILENAMES="${icd}"
  export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
  local libdir
  libdir="$(dirname "${lib}")"
  export LD_LIBRARY_PATH="${libdir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "NVIDIA Vulkan ICD: ${icd} -> ${lib}" >&2
}

# Download + extract the exact host driver userspace (matches nvidia-smi version).
extract_nvidia_run_userspace() {
  local ver="$1"
  local dest="${WORK_DIR}/nvidia-gl"
  mkdir -p "${dest}"

  if [[ -e "${dest}/libGLX_nvidia.so.0" && -f "${dest}/.driver_version" ]] \
    && [[ "$(cat "${dest}/.driver_version")" == "${ver}" ]]; then
    echo "Using cached NVIDIA ${ver} userspace in ${dest}" >&2
    export LD_LIBRARY_PATH="${dest}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    return 0
  fi

  echo "Extracting NVIDIA ${ver} userspace libs (exact match for host driver) ..." >&2
  local tmp runfile url
  tmp="$(mktemp -d)"
  runfile="${tmp}/NVIDIA.run"
  for url in \
    "https://download.nvidia.com/XFree86/Linux-x86_64/${ver}/NVIDIA-Linux-x86_64-${ver}.run" \
    "https://us.download.nvidia.com/XFree86/Linux-x86_64/${ver}/NVIDIA-Linux-x86_64-${ver}.run" \
    "https://us.download.nvidia.com/tesla/${ver}/NVIDIA-Linux-x86_64-${ver}.run"; do
    echo "  trying ${url}" >&2
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
  if ! sh "${runfile}" --extract-only --target "${tmp}/extract" >/tmp/nvidia-extract.log 2>&1; then
    echo "error: NVIDIA .run extract failed; log:" >&2
    tail -n 40 /tmp/nvidia-extract.log >&2 || true
    rm -rf "${tmp}"
    return 1
  fi

  rm -rf "${dest}"
  mkdir -p "${dest}"
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
    cp -f ${tmp}/extract/${f} "${dest}/" 2>/dev/null || true
  done
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

  echo "${ver}" >"${dest}/.driver_version"
  export LD_LIBRARY_PATH="${dest}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "${dest}" >>/etc/ld.so.conf.d/nvidia-gl-bench.conf 2>/dev/null || true
  ldconfig 2>/dev/null || true
  rm -rf "${tmp}"

  if [[ -e "${dest}/libGLX_nvidia.so.0" ]]; then
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

  local ver major lib
  ver="$(host_driver_version)"
  major="${ver%%.*}"

  # 1) Host-mounted lib (best case).
  lib="$(find_libglx_nvidia)"
  if [[ -n "${lib}" && -e "${lib}" ]]; then
    write_nvidia_icd "${lib}"
    if vulkan_has_nvidia; then
      echo "Vulkan NVIDIA adapter OK (host mount)" >&2
      return 0
    fi
    echo "Host/apt libGLX present but vulkaninfo failed; trying exact .run extract ..." >&2
    vulkaninfo --summary 2>&1 | tail -n 20 >&2 || true
  else
    echo "libGLX_nvidia.so.0 not mounted (compute-only host)" >&2
  fi

  # 2) apt major package — often wrong patch level vs host (e.g. 580.95.05).
  if [[ -n "${major}" ]] && command -v apt-get >/dev/null 2>&1; then
    echo "Trying apt libnvidia-gl-${major} (host driver ${ver}) ..." >&2
    apt-get install -y -qq "libnvidia-gl-${major}" >/dev/null 2>&1 || true
    lib="$(find_libglx_nvidia)"
    if [[ -n "${lib}" && -e "${lib}" ]]; then
      write_nvidia_icd "${lib}"
      if vulkan_has_nvidia; then
        echo "Vulkan NVIDIA adapter OK (apt)" >&2
        return 0
      fi
      echo "apt libnvidia-gl-${major} installed but vulkaninfo still fails (version skew?)" >&2
    fi
  fi

  # 3) Exact driver .run extract — required on many Community compute-only hosts.
  if [[ -z "${ver}" ]]; then
    echo "error: cannot determine NVIDIA driver version" >&2
    exit "${EXIT_COMPUTE_ONLY}"
  fi
  if ! extract_nvidia_run_userspace "${ver}"; then
    echo "error: failed to install matching NVIDIA GL userspace for ${ver}" >&2
    exit "${EXIT_COMPUTE_ONLY}"
  fi
  lib="${WORK_DIR}/nvidia-gl/libGLX_nvidia.so.0"
  write_nvidia_icd "${lib}"
  if vulkan_has_nvidia; then
    echo "Vulkan NVIDIA adapter OK (.run extract ${ver})" >&2
    return 0
  fi

  echo "error: vulkaninfo still sees no NVIDIA GPU after .run extract." >&2
  vulkaninfo --summary 2>&1 | tail -n 40 >&2 || true
  echo "exit ${EXIT_COMPUTE_ONLY}: bad Vulkan host — sweep should retry" >&2
  exit "${EXIT_COMPUTE_ONLY}"
}

ensure_rust() {
  if command -v cargo >/dev/null 2>&1; then
    return 0
  fi
  echo "Installing Rust toolchain (rustup) ..." >&2
  # rustup writes the welcome banner to stdout — must not leak into $(download_miner).
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable >&2
  # shellcheck disable=SC1091
  source "${HOME}/.cargo/env"
  if ! command -v cargo >/dev/null 2>&1; then
    echo "error: cargo not available after rustup install" >&2
    exit 1
  fi
}

ensure_build_deps() {
  if ! command -v apt-get >/dev/null 2>&1; then
    return 0
  fi
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y -qq \
    git curl ca-certificates build-essential pkg-config libssl-dev \
    >/dev/null 2>&1 || \
    apt-get install -y -qq git curl build-essential pkg-config libssl-dev || true
}

download_miner_release() {
  local dest="${BIN_DIR}/quantus-miner"
  if [[ -x "${dest}" && "${FORCE_MINER_BUILD}" != "1" ]]; then
    echo "Using existing ${dest}" >&2
    printf '%s\n' "${dest}"
    return
  fi
  echo "Downloading miner release: ${MINER_URL}" >&2
  curl -fL "${MINER_URL}" -o "${dest}"
  chmod +x "${dest}"
  printf '%s\n' "${dest}"
}

# Clone MINER_BRANCH from MINER_REPO and cargo build -p miner-cli --release.
build_miner_from_git() {
  local dest="${BIN_DIR}/quantus-miner"
  local src="${WORK_DIR}/quantus-miner-src"
  local rev_file="${BIN_DIR}/quantus-miner.rev"
  local built_rev=""

  ensure_build_deps
  require_cmd git
  ensure_rust
  # shellcheck disable=SC1091
  [[ -f "${HOME}/.cargo/env" ]] && source "${HOME}/.cargo/env"

  echo "Miner source: ${MINER_REPO} @ ${MINER_BRANCH}" >&2
  # Explicit checks: errexit is cleared inside the $(download_miner) command
  # substitution, so unchecked failures here would silently benchmark stale code.
  if [[ -d "${src}/.git" ]]; then
    if ! git -C "${src}" remote set-url origin "${MINER_REPO}" ||
      ! git -C "${src}" fetch --depth 1 origin "${MINER_BRANCH}" >&2 ||
      ! git -C "${src}" checkout -f FETCH_HEAD >&2; then
      echo "error: git fetch/checkout failed for ${MINER_REPO} @ ${MINER_BRANCH}" >&2
      exit 1
    fi
  else
    rm -rf "${src}"
    if ! git clone --depth 1 --branch "${MINER_BRANCH}" "${MINER_REPO}" "${src}" >&2; then
      echo "error: git clone failed for ${MINER_REPO} branch ${MINER_BRANCH}" >&2
      echo "Push the branch to origin, or set MINER_BRANCH / MINER_REPO." >&2
      exit 1
    fi
  fi

  built_rev="$(git -C "${src}" rev-parse HEAD)" || exit 1
  if [[ -x "${dest}" && "${FORCE_MINER_BUILD}" != "1" && -f "${rev_file}" ]]; then
    if [[ "$(cat "${rev_file}")" == "${built_rev}" ]]; then
      echo "Using existing ${dest} (rev ${built_rev})" >&2
      printf '%s\n' "${dest}"
      return
    fi
  fi

  echo "Building quantus-miner (release, rev ${built_rev}) ..." >&2
  if ! (
    cd "${src}"
    cargo build -p miner-cli --release >&2
  ); then
    echo "error: cargo build failed (rev ${built_rev})" >&2
    exit 1
  fi
  if [[ ! -x "${src}/target/release/quantus-miner" ]]; then
    echo "error: cargo build did not produce target/release/quantus-miner" >&2
    exit 1
  fi
  mkdir -p "${BIN_DIR}"
  cp -f "${src}/target/release/quantus-miner" "${dest}"
  chmod +x "${dest}"
  echo "${built_rev}" >"${rev_file}"
  echo "Built ${dest}" >&2
  # Sole stdout line — consumed by MINER_BIN="$(download_miner)"
  printf '%s\n' "${dest}"
}

download_miner() {
  case "${MINER_SOURCE}" in
    git | build | source)
      build_miner_from_git
      ;;
    release | binary)
      download_miner_release
      ;;
    *)
      echo "error: unknown MINER_SOURCE=${MINER_SOURCE} (use git or release)" >&2
      exit 1
      ;;
  esac
}

ensure_nvidia_vulkan
MINER_BIN="$(download_miner | tail -n 1)"
if [[ ! -x "${MINER_BIN}" ]]; then
  echo "error: miner binary missing or not executable: '${MINER_BIN}'" >&2
  exit 1
fi

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || nvidia-smi || true

# Smoke-test GPU path before spending minutes on a batch sweep.
echo "Smoke-testing benchmark (5s) ..."
set +e
smoke_log="$(mktemp)"
env \
  VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-}" \
  NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}" \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-root}" \
  "${MINER_BIN}" benchmark \
  --cpu-workers 0 \
  --gpu-devices "${GPU_DEVICES}" \
  --gpu-batch-size 1000000 \
  --duration 5 \
  >"${smoke_log}" 2>&1
smoke_rc=$?
set -e
if [[ "${smoke_rc}" -ne 0 ]]; then
  echo "error: benchmark smoke test failed; log:" >&2
  cat "${smoke_log}" >&2
  if grep -qiE 'No usable GPU|llvmpipe|Vulkan|Failed to initialize GPU' "${smoke_log}" 2>/dev/null; then
    echo "hint: need NVIDIA Vulkan ICD + NVIDIA_DRIVER_CAPABILITIES=all (not Mesa llvmpipe)" >&2
    rm -f "${smoke_log}"
    exit "${EXIT_COMPUTE_ONLY}"
  fi
  rm -f "${smoke_log}"
  exit 1
fi
rm -f "${smoke_log}"

NOTE_ARGS=()
if [[ -n "${NOTES}" ]]; then
  NOTE_ARGS=(--notes "${NOTES}")
fi

cd "${WORK_DIR}"
echo "Batch-size sweep: ${BATCH_SIZES} (${DURATION}s each, job_interval=${JOB_INTERVAL})"
EXTRA_ARGS=()
if awk -v j="${JOB_INTERVAL}" 'BEGIN { exit !(j+0 > 0) }'; then
  EXTRA_ARGS+=(--job-interval "${JOB_INTERVAL}" --job-jitter "${JOB_JITTER}")
fi
if [[ -n "${DIFFICULTY}" ]]; then
  EXTRA_ARGS+=(--difficulty "${DIFFICULTY}")
fi
export MINER_BIN
env \
  VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-}" \
  NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}" \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-root}" \
  MINER_BIN="${MINER_BIN}" \
  ./record.sh --benchmark \
  --provider "${PROVIDER}" \
  --cost-per-hour "${COST_PER_HOUR}" \
  --duration "${DURATION}" \
  --gpu-devices "${GPU_DEVICES}" \
  --batch-sizes "${BATCH_SIZES}" \
  "${EXTRA_ARGS[@]}" \
  "${NOTE_ARGS[@]}"

echo "Done. Results: ${RESULTS_CSV}"
tail -n 8 "${RESULTS_CSV}" || true

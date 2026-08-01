#!/usr/bin/env bash
# Start (or stop) a native Quantus node + GPU miner on this host.
# Same-machine layout for cloud GPU rentals (RunPod, Vast, etc.).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="${SCRIPT_DIR}/.run"
ENV_FILE="${SCRIPT_DIR}/.env"
BIN_DIR="${RUN_DIR}/bin"

DEFAULT_MINER_DIR="${REPO_ROOT}"
DEFAULT_CHAIN_DIR="$(cd "${REPO_ROOT}/.." && pwd)/chain"
NODE_CONTAINER_NAME="${NODE_CONTAINER_NAME:-quantus-gpu-bench-node}"

NODE_MODE="native" # native | docker

usage() {
  cat <<'EOF'
Usage: ./setup.sh [start|stop|status|fetch-node|wormhole] [--docker]

  start       Download/build binaries, start native node + GPU miner (default)
  stop        Stop miner and node
  status      Show run state
  fetch-node  Download quantus-node into .run/bin/ (no start)
  wormhole    Print a new wormhole address + inner_hash (for REWARDS_INNER_HASH)
  --docker    Use Docker for the node instead of a native binary
              (not recommended on RunPod / nested-Docker hosts)

Environment (see .env.example):
  REWARDS_INNER_HASH   Required for start
  QUANTUS_NODE_BIN     Prebuilt quantus-node (skips download)
  QUANTUS_MINER_DIR    Path to quantus-miner checkout (default: parent of gpu-bench)
  MINER_BIN            Prebuilt quantus-miner (skips cargo build)
  GPU_DEVICES          Number of GPUs (default: 1)
EOF
}

load_env() {
  if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    set -a
    source "${ENV_FILE}"
    set +a
  fi
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    exit 1
  fi
}

check_prereqs() {
  require_cmd nvidia-smi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "error: nvidia-smi failed — NVIDIA driver required for this toolkit" >&2
    exit 1
  fi
  if [[ "${NODE_MODE}" == "docker" ]]; then
    require_cmd docker
  else
    require_cmd curl
    require_cmd tar
  fi
}

stop_pidfile() {
  local pidfile="$1"
  local label="$2"
  if [[ ! -f "${pidfile}" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "${pidfile}")"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping ${label} (pid ${pid}) ..."
    kill "${pid}" 2>/dev/null || true
    local i
    for ((i = 0; i < 20; i++)); do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "${pid}" 2>/dev/null; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
  fi
  rm -f "${pidfile}"
}

resolve_miner_bin() {
  if [[ -n "${MINER_BIN:-}" ]]; then
    if [[ ! -x "${MINER_BIN}" ]]; then
      echo "error: MINER_BIN is not executable: ${MINER_BIN}" >&2
      exit 1
    fi
    echo "${MINER_BIN}"
    return
  fi

  local miner_dir="${QUANTUS_MINER_DIR:-${DEFAULT_MINER_DIR}}"
  if [[ ! -d "${miner_dir}" ]]; then
    echo "error: quantus-miner not found at ${miner_dir}" >&2
    echo "Set QUANTUS_MINER_DIR or MINER_BIN." >&2
    exit 1
  fi

  local bin="${miner_dir}/target/release/quantus-miner"
  if [[ -x "${bin}" ]]; then
    echo "${bin}"
    return
  fi

  require_cmd cargo
  echo "Building quantus-miner (release) in ${miner_dir} ..." >&2
  (
    cd "${miner_dir}"
    cargo build -p miner-cli --release
  ) >&2
  if [[ ! -x "${bin}" ]]; then
    echo "error: expected binary missing: ${bin}" >&2
    exit 1
  fi
  echo "${bin}"
}

detect_node_target() {
  local os arch
  case "$(uname -s)" in
    Linux) os="linux" ;;
    Darwin) os="macos" ;;
    *)
      echo "error: unsupported OS for node download: $(uname -s)" >&2
      exit 1
      ;;
  esac
  case "$(uname -m)" in
    x86_64 | amd64)
      if [[ "${os}" == "linux" ]]; then
        arch="x86_64-unknown-linux-gnu"
      else
        arch="x86_64-apple-darwin"
      fi
      ;;
    arm64 | aarch64)
      if [[ "${os}" == "linux" ]]; then
        arch="aarch64-unknown-linux-gnu"
      else
        arch="aarch64-apple-darwin"
      fi
      ;;
    *)
      echo "error: unsupported arch for node download: $(uname -m)" >&2
      exit 1
      ;;
  esac
  echo "${arch}"
}

download_node_binary() {
  local dest="$1"
  local target
  target="$(detect_node_target)"
  echo "Fetching latest quantus-node release (${target}) ..." >&2

  local release_json tag asset_url tmp
  release_json="$(curl -fsSL https://api.github.com/repos/Quantus-Network/chain/releases/latest)"
  tag="$(echo "${release_json}" | grep -o '"tag_name": "[^"]*"' | head -n 1 | cut -d'"' -f4)"
  if [[ -z "${tag}" ]]; then
    echo "error: could not determine latest chain release tag" >&2
    exit 1
  fi

  asset_url="https://github.com/Quantus-Network/chain/releases/download/${tag}/quantus-node-${tag}-${target}.tar.gz"
  echo "Downloading ${asset_url} ..." >&2
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '${tmp}'" RETURN
  if ! curl -fL "${asset_url}" -o "${tmp}/node.tar.gz"; then
    echo "error: failed to download quantus-node for ${target}" >&2
    echo "Set QUANTUS_NODE_BIN to a local binary, or build from the chain repo." >&2
    exit 1
  fi
  tar -xzf "${tmp}/node.tar.gz" -C "${tmp}"
  if [[ ! -f "${tmp}/quantus-node" ]]; then
    echo "error: archive did not contain quantus-node" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${dest}")"
  mv "${tmp}/quantus-node" "${dest}"
  chmod +x "${dest}"
  echo "Installed quantus-node -> ${dest}" >&2
}

resolve_node_bin() {
  if [[ -n "${QUANTUS_NODE_BIN:-}" ]]; then
    if [[ ! -x "${QUANTUS_NODE_BIN}" ]]; then
      echo "error: QUANTUS_NODE_BIN is not executable: ${QUANTUS_NODE_BIN}" >&2
      exit 1
    fi
    echo "${QUANTUS_NODE_BIN}"
    return
  fi

  if command -v quantus-node >/dev/null 2>&1; then
    command -v quantus-node
    return
  fi

  local chain_dir="${QUANTUS_CHAIN_DIR:-${DEFAULT_CHAIN_DIR}}"
  if [[ -x "${chain_dir}/target/release/quantus-node" ]]; then
    echo "${chain_dir}/target/release/quantus-node"
    return
  fi

  local cached="${BIN_DIR}/quantus-node"
  if [[ -x "${cached}" ]]; then
    echo "${cached}"
    return
  fi

  download_node_binary "${cached}"
  echo "${cached}"
}

ensure_rewards() {
  local rewards="${REWARDS_INNER_HASH:-}"
  if [[ -z "${rewards}" || "${rewards}" == "0xyour_inner_hash_here" ]]; then
    echo "error: set REWARDS_INNER_HASH in ${ENV_FILE} or the environment" >&2
    echo "Generate with (after node binary is available):" >&2
    echo "  quantus-node key quantus --scheme wormhole" >&2
    exit 1
  fi
}

ensure_node_key() {
  local node_bin="$1"
  local key_file="${RUN_DIR}/node-keys/key_node"
  mkdir -p "$(dirname "${key_file}")"
  if [[ ! -f "${key_file}" ]]; then
    echo "Generating node key at ${key_file} ..."
    "${node_bin}" key generate-node-key --file "${key_file}"
  fi
  echo "${key_file}"
}

start_node_native() {
  ensure_rewards
  local node_bin
  node_bin="$(resolve_node_bin)"
  local key_file
  key_file="$(ensure_node_key "${node_bin}")"

  local chain="${CHAIN:-planck}"
  local node_name="${NODE_NAME:-gpu-bench-node}"
  local p2p_port="${P2P_PORT:-30333}"
  local rpc_port="${RPC_PORT:-9944}"
  local prom_port="${PROMETHEUS_PORT:-9615}"
  local miner_listen_port="${HOST_MINER_LISTEN_PORT:-9833}"
  local base_path="${RUN_DIR}/node-data"
  mkdir -p "${base_path}"

  stop_pidfile "${RUN_DIR}/node.pid" "node"

  local log_file="${RUN_DIR}/node.log"
  echo "Starting native quantus-node: ${node_bin}"
  echo "  --miner-listen-port ${miner_listen_port} --chain ${chain}"

  nohup "${node_bin}" \
    --validator \
    --base-path "${base_path}" \
    --chain "${chain}" \
    --node-key-file "${key_file}" \
    --rewards-inner-hash "${REWARDS_INNER_HASH}" \
    --name "${node_name}" \
    --port "${p2p_port}" \
    --rpc-port "${rpc_port}" \
    --prometheus-port "${prom_port}" \
    --prometheus-external \
    --miner-listen-port "${miner_listen_port}" \
    --wasm-execution compiled \
    --db-cache 2048 \
    --rpc-cors all \
    --max-blocks-per-request 64 \
    >"${log_file}" 2>&1 &

  local pid=$!
  echo "${pid}" >"${RUN_DIR}/node.pid"
  echo "native" >"${RUN_DIR}/node.mode"
  echo "${node_bin}" >"${RUN_DIR}/node.bin"
  echo "Node pid ${pid}; miner QUIC on 127.0.0.1:${miner_listen_port}/udp"
  echo "Logs: ${log_file}"
}

start_node_docker() {
  ensure_rewards

  if docker ps -a --format '{{.Names}}' | grep -qx "${NODE_CONTAINER_NAME}"; then
    echo "Removing existing container ${NODE_CONTAINER_NAME} ..."
    docker rm -f "${NODE_CONTAINER_NAME}" >/dev/null
  fi

  local node_version="${NODE_VERSION:-latest}"
  local chain="${CHAIN:-planck}"
  local node_name="${NODE_NAME:-gpu-bench-node}"
  local p2p_port="${P2P_PORT:-30333}"
  local rpc_port="${RPC_PORT:-9944}"
  local prom_port="${PROMETHEUS_PORT:-9615}"
  local miner_listen_port="${HOST_MINER_LISTEN_PORT:-9833}"

  mkdir -p "${RUN_DIR}/node-keys" "${RUN_DIR}/node-data"

  echo "Starting quantus-node via Docker (${NODE_CONTAINER_NAME}) ..."
  docker run -d \
    --name "${NODE_CONTAINER_NAME}" \
    --restart unless-stopped \
    --platform linux/amd64 \
    -v "${SCRIPT_DIR}/init-node.sh:/init-node.sh:ro" \
    -v "${RUN_DIR}/node-keys:/node-keys" \
    -v "${RUN_DIR}/node-data:/var/lib/quantus" \
    -p "${p2p_port}:30333" \
    -p "${rpc_port}:9944" \
    -p "${prom_port}:9615" \
    -p "${miner_listen_port}:9833/udp" \
    --entrypoint /init-node.sh \
    "ghcr.io/quantus-network/quantus-node:${node_version}" \
    --validator \
    --base-path /var/lib/quantus \
    --chain "${chain}" \
    --node-key-file /node-keys/key_node \
    --rewards-inner-hash "${REWARDS_INNER_HASH}" \
    --name "${node_name}" \
    --wasm-execution compiled \
    --db-cache 2048 \
    --rpc-cors all \
    --prometheus-external \
    --miner-listen-port 9833 \
    >/dev/null

  echo "${NODE_CONTAINER_NAME}" >"${RUN_DIR}/node.container"
  echo "docker" >"${RUN_DIR}/node.mode"
  echo "Node listening for miners on 127.0.0.1:${miner_listen_port}/udp"
}

wait_for_miner_port() {
  local port="$1"
  local attempts=30
  local i
  for ((i = 1; i <= attempts; i++)); do
    if curl -sf "http://127.0.0.1:${port}/metrics" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "warning: miner metrics not ready on :${port} after ${attempts}s (continuing)" >&2
}

start_miner() {
  local miner_bin
  miner_bin="$(resolve_miner_bin)"

  local gpu_devices="${GPU_DEVICES:-1}"
  local metrics_port="${METRICS_PORT:-9900}"
  local miner_listen_port="${HOST_MINER_LISTEN_PORT:-9833}"
  local miner_log="${MINER_LOG:-info}"

  stop_pidfile "${RUN_DIR}/miner.pid" "miner"

  mkdir -p "${RUN_DIR}"
  local log_file="${RUN_DIR}/miner.log"
  echo "Starting GPU miner: ${miner_bin}"
  echo "  --node-addr 127.0.0.1:${miner_listen_port} --gpu-devices ${gpu_devices} --cpu-workers 0"

  RUST_LOG="${miner_log}" nohup "${miner_bin}" serve \
    --node-addr "127.0.0.1:${miner_listen_port}" \
    --gpu-devices "${gpu_devices}" \
    --cpu-workers 0 \
    --metrics-port "${metrics_port}" \
    >"${log_file}" 2>&1 &

  local pid=$!
  echo "${pid}" >"${RUN_DIR}/miner.pid"
  echo "${miner_bin}" >"${RUN_DIR}/miner.bin"
  echo "${metrics_port}" >"${RUN_DIR}/metrics.port"

  wait_for_miner_port "${metrics_port}"
  echo "Miner pid ${pid}; metrics http://127.0.0.1:${metrics_port}/metrics"
  echo "Logs: ${log_file}"
}

do_fetch_node() {
  load_env
  require_cmd curl
  require_cmd tar
  mkdir -p "${BIN_DIR}"
  local bin
  bin="$(resolve_node_bin)"
  echo "quantus-node ready: ${bin}"
  echo "Generate rewards hash with: ./setup.sh wormhole"
}

do_wormhole() {
  load_env
  require_cmd curl
  require_cmd tar
  mkdir -p "${BIN_DIR}"
  local bin
  bin="$(resolve_node_bin)"
  echo "Using ${bin}" >&2
  echo "Save the inner_hash into .env as REWARDS_INNER_HASH" >&2
  echo >&2
  "${bin}" key quantus --scheme wormhole
}

do_start() {
  load_env
  check_prereqs
  mkdir -p "${RUN_DIR}"

  if [[ "${NODE_MODE}" == "docker" ]]; then
    start_node_docker
  else
    start_node_native
  fi

  sleep 2
  start_miner
  echo
  echo "Stack is up (same host). Next:"
  echo "  ./record.sh --provider <name> --cost-per-hour <usd>"
  echo "Or hardware-only (no node): ./record.sh --benchmark --provider <name> --cost-per-hour <usd>"
}

do_stop() {
  load_env
  stop_pidfile "${RUN_DIR}/miner.pid" "miner"
  stop_pidfile "${RUN_DIR}/node.pid" "node"

  local container="${NODE_CONTAINER_NAME}"
  if [[ -f "${RUN_DIR}/node.container" ]]; then
    container="$(cat "${RUN_DIR}/node.container")"
  fi
  if command -v docker >/dev/null 2>&1 && docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "${container}"; then
    echo "Removing node container ${container} ..."
    docker rm -f "${container}" >/dev/null
  fi
  rm -f "${RUN_DIR}/node.container" "${RUN_DIR}/node.mode"
  echo "Stopped."
}

do_status() {
  load_env
  echo "Run dir: ${RUN_DIR}"
  local mode="unknown"
  if [[ -f "${RUN_DIR}/node.mode" ]]; then
    mode="$(cat "${RUN_DIR}/node.mode")"
  fi
  echo "Node mode: ${mode}"

  if [[ -f "${RUN_DIR}/node.pid" ]]; then
    local pid
    pid="$(cat "${RUN_DIR}/node.pid")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Node: running (pid ${pid})"
    else
      echo "Node: not running (stale pid ${pid})"
    fi
  elif [[ -f "${RUN_DIR}/node.container" ]]; then
    local c
    c="$(cat "${RUN_DIR}/node.container")"
    if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' | grep -qx "${c}"; then
      echo "Node: running (docker ${c})"
    else
      echo "Node: not running (recorded docker ${c})"
    fi
  else
    echo "Node: not started via setup.sh"
  fi

  if [[ -f "${RUN_DIR}/miner.pid" ]]; then
    local pid
    pid="$(cat "${RUN_DIR}/miner.pid")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Miner: running (pid ${pid})"
    else
      echo "Miner: not running (stale pid ${pid})"
    fi
  else
    echo "Miner: not started via setup.sh"
  fi

  local metrics_port="${METRICS_PORT:-9900}"
  if [[ -f "${RUN_DIR}/metrics.port" ]]; then
    metrics_port="$(cat "${RUN_DIR}/metrics.port")"
  fi
  if curl -sf "http://127.0.0.1:${metrics_port}/metrics" >/dev/null 2>&1; then
    echo "Metrics: ok on :${metrics_port}"
  else
    echo "Metrics: not reachable on :${metrics_port}"
  fi
}

cmd="start"
while [[ $# -gt 0 ]]; do
  case "$1" in
    start | stop | status | fetch-node | wormhole)
      cmd="$1"
      shift
      ;;
    --docker)
      NODE_MODE="docker"
      shift
      ;;
    -h | --help | help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${cmd}" in
  start) do_start ;;
  stop) do_stop ;;
  status) do_status ;;
  fetch-node) do_fetch_node ;;
  wormhole) do_wormhole ;;
esac

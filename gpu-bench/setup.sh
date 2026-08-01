#!/usr/bin/env bash
# Start (or stop) a Quantus node + native GPU miner on this host for benchmarking.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="${SCRIPT_DIR}/.run"
ENV_FILE="${SCRIPT_DIR}/.env"

NODE_CONTAINER_NAME="${NODE_CONTAINER_NAME:-quantus-gpu-bench-node}"
DEFAULT_MINER_DIR="${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage: ./setup.sh [start|stop|status]

  start   Build/start native GPU miner + Docker node (default)
  stop    Stop miner and remove node container
  status  Show run state

Environment (see .env.example):
  REWARDS_INNER_HASH   Required for start
  QUANTUS_MINER_DIR    Path to quantus-miner checkout (default: parent of gpu-bench)
  MINER_BIN            Prebuilt quantus-miner binary (skips cargo build)
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
  require_cmd docker
  require_cmd nvidia-smi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "error: nvidia-smi failed — NVIDIA driver required for this toolkit" >&2
    exit 1
  fi
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

  require_cmd cargo
  echo "Building quantus-miner (release) in ${miner_dir} ..." >&2
  (
    cd "${miner_dir}"
    cargo build -p miner-cli --release
  ) >&2
  local bin="${miner_dir}/target/release/quantus-miner"
  if [[ ! -x "${bin}" ]]; then
    echo "error: expected binary missing: ${bin}" >&2
    exit 1
  fi
  echo "${bin}"
}

start_node() {
  local rewards="${REWARDS_INNER_HASH:-}"
  if [[ -z "${rewards}" || "${rewards}" == "0xyour_inner_hash_here" ]]; then
    echo "error: set REWARDS_INNER_HASH in ${ENV_FILE} or the environment" >&2
    echo "Generate with: docker run --rm ghcr.io/quantus-network/quantus-node:latest key quantus --scheme wormhole" >&2
    exit 1
  fi

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

  echo "Starting quantus-node (${NODE_CONTAINER_NAME}) ..."
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
    --rewards-inner-hash "${rewards}" \
    --name "${node_name}" \
    --wasm-execution compiled \
    --db-cache 2048 \
    --rpc-cors all \
    --prometheus-external \
    --miner-listen-port 9833 \
    >/dev/null

  echo "${NODE_CONTAINER_NAME}" >"${RUN_DIR}/node.container"
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

  if [[ -f "${RUN_DIR}/miner.pid" ]]; then
    local old_pid
    old_pid="$(cat "${RUN_DIR}/miner.pid")"
    if kill -0 "${old_pid}" 2>/dev/null; then
      echo "Stopping previous miner (pid ${old_pid}) ..."
      kill "${old_pid}" 2>/dev/null || true
      local i
      for ((i = 0; i < 10; i++)); do
        kill -0 "${old_pid}" 2>/dev/null || break
        sleep 0.5
      done
      if kill -0 "${old_pid}" 2>/dev/null; then
        kill -9 "${old_pid}" 2>/dev/null || true
      fi
    fi
  fi

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

do_start() {
  load_env
  check_prereqs
  mkdir -p "${RUN_DIR}"
  start_node
  # Brief pause so UDP listener is up before miner connects
  sleep 2
  start_miner
  echo
  echo "Stack is up. Record a spreadsheet row with:"
  echo "  ./record.sh --provider <name> --cost-per-hour <usd>"
}

do_stop() {
  load_env
  if [[ -f "${RUN_DIR}/miner.pid" ]]; then
    local pid
    pid="$(cat "${RUN_DIR}/miner.pid")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Stopping miner (pid ${pid}) ..."
      kill "${pid}" 2>/dev/null || true
      # Give it a moment; force if needed
      sleep 1
      if kill -0 "${pid}" 2>/dev/null; then
        kill -9 "${pid}" 2>/dev/null || true
      fi
    fi
    rm -f "${RUN_DIR}/miner.pid"
  fi

  local container="${NODE_CONTAINER_NAME}"
  if [[ -f "${RUN_DIR}/node.container" ]]; then
    container="$(cat "${RUN_DIR}/node.container")"
  fi
  if docker ps -a --format '{{.Names}}' | grep -qx "${container}"; then
    echo "Removing node container ${container} ..."
    docker rm -f "${container}" >/dev/null
  fi
  rm -f "${RUN_DIR}/node.container"
  echo "Stopped."
}

do_status() {
  load_env
  echo "Run dir: ${RUN_DIR}"
  if [[ -f "${RUN_DIR}/node.container" ]]; then
    local c
    c="$(cat "${RUN_DIR}/node.container")"
    if docker ps --format '{{.Names}}' | grep -qx "${c}"; then
      echo "Node: running (${c})"
    else
      echo "Node: not running (recorded ${c})"
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

cmd="${1:-start}"
case "${cmd}" in
  start) do_start ;;
  stop) do_stop ;;
  status) do_status ;;
  -h | --help | help) usage ;;
  *)
    echo "error: unknown command: ${cmd}" >&2
    usage >&2
    exit 1
    ;;
esac

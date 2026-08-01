#!/usr/bin/env bash
# Sample GPU miner performance and append one row to results.csv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${SCRIPT_DIR}/.run"
ENV_FILE="${SCRIPT_DIR}/.env"
RESULTS_CSV="${SCRIPT_DIR}/results.csv"
CSV_HEADER="timestamp,cloud_provider,gpu_model,vram_mb,sm_count,driver_version,hashrate,gpu_utilization_pct,cost_per_hour,efficiency,sample_seconds,notes"
DEFAULT_MINER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="live"
PROVIDER=""
COST_PER_HOUR=""
DURATION=60
NOTES=""
DRY_RUN=0
GPU_DEVICES_FLAG=""

usage() {
  cat <<'EOF'
Usage: ./record.sh [options]

Options:
  --live                 Sample a running miner (default; requires setup.sh)
  --benchmark            Run quantus-miner benchmark (no node required)
  --provider NAME        Cloud provider label (e.g. vast.ai, runpod)
  --cost-per-hour USD    Hourly cost in USD (e.g. 0.35)
  --duration SECONDS     Sample / benchmark window (default: 60)
  --gpu-devices N        GPUs for --benchmark (default: GPU_DEVICES or 1)
  --notes TEXT           Optional notes column
  --dry-run              Print the CSV row but do not append
  -h, --help             Show this help

Examples:
  ./record.sh --provider vast.ai --cost-per-hour 0.35
  ./record.sh --benchmark --provider runpod --cost-per-hour 0.42 --duration 60
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

csv_escape() {
  local s="${1:-}"
  if [[ "${s}" == *","* || "${s}" == *"\""* || "${s}" == *$'\n'* ]]; then
    s="${s//\"/\"\"}"
    printf '"%s"' "${s}"
  else
    printf '%s' "${s}"
  fi
}

ensure_results_header() {
  if [[ ! -f "${RESULTS_CSV}" ]]; then
    printf '%s\n' "${CSV_HEADER}" >"${RESULTS_CSV}"
    return
  fi
  if [[ ! -s "${RESULTS_CSV}" ]]; then
    printf '%s\n' "${CSV_HEADER}" >"${RESULTS_CSV}"
  fi
}

query_gpu_static() {
  # name, memory.total [MiB], multiprocessor_count, driver_version
  local line
  if ! line="$(nvidia-smi --query-gpu=name,memory.total,multiprocessor_count,driver_version \
    --format=csv,noheader,nounits 2>/dev/null | head -n 1)"; then
    line=""
  fi
  if [[ -z "${line}" ]]; then
    # Some drivers (e.g. 580.x) do not support multiprocessor_count
    line="$(nvidia-smi --query-gpu=name,memory.total,driver_version \
      --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
    GPU_MODEL="$(echo "${line}" | awk -F', ' '{print $1}')"
    VRAM_MB="$(echo "${line}" | awk -F', ' '{print $2}' | tr -d ' ')"
    SM_COUNT=""
    DRIVER_VERSION="$(echo "${line}" | awk -F', ' '{print $3}' | tr -d ' ')"
    return
  fi
  GPU_MODEL="$(echo "${line}" | awk -F', ' '{print $1}')"
  VRAM_MB="$(echo "${line}" | awk -F', ' '{print $2}' | tr -d ' ')"
  SM_COUNT="$(echo "${line}" | awk -F', ' '{print $3}' | tr -d ' ')"
  DRIVER_VERSION="$(echo "${line}" | awk -F', ' '{print $4}' | tr -d ' ')"
}

read_gpu_util() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | head -n 1 | tr -d ' '
}

read_prometheus_hashrate() {
  local port="$1"
  local metrics
  if ! metrics="$(curl -sf "http://127.0.0.1:${port}/metrics")"; then
    echo ""
    return
  fi
  local gpu_hr total_hr
  gpu_hr="$(echo "${metrics}" | awk '/^miner_gpu_hash_rate[[:space:]]/{print $2; exit}')"
  if [[ -n "${gpu_hr}" ]]; then
    echo "${gpu_hr}"
    return
  fi
  total_hr="$(echo "${metrics}" | awk '/^miner_hash_rate[[:space:]]/{print $2; exit}')"
  echo "${total_hr}"
}

average_list() {
  # stdin: one number per line; prints average or empty
  awk '
    NF && $1+0 == $1 {
      sum += $1
      n++
    }
    END {
      if (n > 0) printf "%.6f", sum / n
    }
  '
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
  local bin="${miner_dir}/target/release/quantus-miner"
  if [[ -x "${bin}" ]]; then
    echo "${bin}"
    return
  fi

  if [[ ! -d "${miner_dir}" ]]; then
    echo "error: quantus-miner not found at ${miner_dir}" >&2
    exit 1
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

parse_benchmark_hashrate() {
  # Prefer Total hashes / Total time for a numeric H/s.
  local out="$1"
  local total_hashes total_time
  total_hashes="$(echo "${out}" | awk -F': ' '/^Total hashes:/{gsub(/[^0-9]/,"",$2); print $2; exit}')"
  total_time="$(echo "${out}" | awk -F': ' '/^Total time:/{gsub(/s$/,"",$2); print $2; exit}')"
  if [[ -n "${total_hashes}" && -n "${total_time}" ]]; then
    awk -v h="${total_hashes}" -v t="${total_time}" 'BEGIN {
      if (t+0 > 0) printf "%.6f", h / t
    }'
    return
  fi
  echo ""
}

sample_util_during() {
  # Background util sampler; writes samples to $1 for $2 seconds every ~2s
  local out_file="$1"
  local seconds="$2"
  local end=$((SECONDS + seconds))
  : >"${out_file}"
  while (( SECONDS < end )); do
    local u
    u="$(read_gpu_util || true)"
    if [[ -n "${u}" ]]; then
      echo "${u}" >>"${out_file}"
    fi
    sleep 2
  done
}

prompt_if_empty() {
  local var_name="$1"
  local prompt="$2"
  local current="${!var_name:-}"
  if [[ -n "${current}" ]]; then
    return
  fi
  if [[ ! -t 0 ]]; then
    echo "error: ${var_name} required (pass flag; stdin is not a TTY)" >&2
    exit 1
  fi
  local value
  read -r -p "${prompt}: " value
  printf -v "${var_name}" '%s' "${value}"
}

emit_row() {
  local timestamp="$1"
  local hashrate="$2"
  local util_avg="$3"
  local efficiency="$4"

  local row
  row="$(csv_escape "${timestamp}"),$(csv_escape "${PROVIDER}"),$(csv_escape "${GPU_MODEL}"),$(csv_escape "${VRAM_MB}"),$(csv_escape "${SM_COUNT}"),$(csv_escape "${DRIVER_VERSION}"),$(csv_escape "${hashrate}"),$(csv_escape "${util_avg}"),$(csv_escape "${COST_PER_HOUR}"),$(csv_escape "${efficiency}"),$(csv_escape "${DURATION}"),$(csv_escape "${NOTES}")"

  echo "${CSV_HEADER}"
  echo "${row}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "(dry-run: not written to ${RESULTS_CSV})" >&2
    return
  fi

  ensure_results_header
  echo "${row}" >>"${RESULTS_CSV}"
  echo "Appended row to ${RESULTS_CSV}" >&2
}

run_live() {
  require_cmd curl
  local metrics_port="${METRICS_PORT:-9900}"
  if [[ -f "${RUN_DIR}/metrics.port" ]]; then
    metrics_port="$(cat "${RUN_DIR}/metrics.port")"
  fi

  if ! curl -sf "http://127.0.0.1:${metrics_port}/metrics" >/dev/null; then
    echo "error: miner metrics not reachable at http://127.0.0.1:${metrics_port}/metrics" >&2
    echo "Start the stack with ./setup.sh, or use --benchmark." >&2
    exit 1
  fi

  echo "Sampling live miner for ${DURATION}s (metrics :${metrics_port}) ..." >&2
  local hr_file util_file
  hr_file="$(mktemp)"
  util_file="$(mktemp)"
  trap 'rm -f "${hr_file}" "${util_file}"' RETURN

  local end=$((SECONDS + DURATION))
  while (( SECONDS < end )); do
    local hr util
    hr="$(read_prometheus_hashrate "${metrics_port}")"
    util="$(read_gpu_util || true)"
    if [[ -n "${hr}" ]]; then
      echo "${hr}" >>"${hr_file}"
    fi
    if [[ -n "${util}" ]]; then
      echo "${util}" >>"${util_file}"
    fi
    sleep 2
  done

  local hashrate util_avg
  hashrate="$(average_list <"${hr_file}")"
  util_avg="$(average_list <"${util_file}")"
  if [[ -z "${hashrate}" ]]; then
    echo "error: no hashrate samples collected from :${metrics_port}" >&2
    exit 1
  fi
  # Round util for spreadsheet readability
  if [[ -n "${util_avg}" ]]; then
    util_avg="$(awk -v u="${util_avg}" 'BEGIN { printf "%.2f", u }')"
  fi
  hashrate="$(awk -v h="${hashrate}" 'BEGIN { printf "%.6f", h }')"

  local efficiency
  efficiency="$(awk -v h="${hashrate}" -v c="${COST_PER_HOUR}" 'BEGIN {
    if (c+0 > 0) printf "%.6f", h / c
  }')"

  local ts
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  emit_row "${ts}" "${hashrate}" "${util_avg}" "${efficiency}"
}

run_benchmark() {
  local miner_bin
  miner_bin="$(resolve_miner_bin)"
  local gpu_devices="${GPU_DEVICES_FLAG:-${GPU_DEVICES:-1}}"

  local util_file bench_log
  util_file="$(mktemp)"
  bench_log="$(mktemp)"
  trap 'rm -f "${util_file}" "${bench_log}"' RETURN

  echo "Running GPU benchmark for ${DURATION}s (gpu-devices=${gpu_devices}) ..." >&2
  sample_util_during "${util_file}" "${DURATION}" &
  local sampler_pid=$!

  set +e
  "${miner_bin}" benchmark \
    --cpu-workers 0 \
    --gpu-devices "${gpu_devices}" \
    --duration "${DURATION}" \
    >"${bench_log}" 2>&1
  local bench_rc=$?
  set -e

  wait "${sampler_pid}" 2>/dev/null || true

  if [[ "${bench_rc}" -ne 0 ]]; then
    echo "error: benchmark failed (exit ${bench_rc}). Output:" >&2
    cat "${bench_log}" >&2
    exit 1
  fi

  cat "${bench_log}" >&2

  local hashrate util_avg
  hashrate="$(parse_benchmark_hashrate "$(cat "${bench_log}")")"
  if [[ -z "${hashrate}" ]]; then
    echo "error: could not parse hashrate from benchmark output" >&2
    exit 1
  fi
  util_avg="$(average_list <"${util_file}")"
  if [[ -n "${util_avg}" ]]; then
    util_avg="$(awk -v u="${util_avg}" 'BEGIN { printf "%.2f", u }')"
  fi

  local efficiency
  efficiency="$(awk -v h="${hashrate}" -v c="${COST_PER_HOUR}" 'BEGIN {
    if (c+0 > 0) printf "%.6f", h / c
  }')"

  local ts
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  emit_row "${ts}" "${hashrate}" "${util_avg}" "${efficiency}"
}

# --- args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --live)
      MODE="live"
      shift
      ;;
    --benchmark)
      MODE="benchmark"
      shift
      ;;
    --provider)
      PROVIDER="${2:-}"
      shift 2
      ;;
    --cost-per-hour)
      COST_PER_HOUR="${2:-}"
      shift 2
      ;;
    --duration)
      DURATION="${2:-}"
      shift 2
      ;;
    --gpu-devices)
      GPU_DEVICES_FLAG="${2:-}"
      shift 2
      ;;
    --notes)
      NOTES="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

load_env
require_cmd nvidia-smi
if ! nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi failed — NVIDIA driver required" >&2
  exit 1
fi

if ! [[ "${DURATION}" =~ ^[0-9]+$ ]] || [[ "${DURATION}" -lt 1 ]]; then
  echo "error: --duration must be a positive integer" >&2
  exit 1
fi

prompt_if_empty PROVIDER "Cloud provider (e.g. vast.ai)"
prompt_if_empty COST_PER_HOUR "Cost per hour USD (e.g. 0.35)"

if ! awk -v c="${COST_PER_HOUR}" 'BEGIN { exit !(c+0 > 0) }'; then
  echo "error: --cost-per-hour must be a positive number" >&2
  exit 1
fi

query_gpu_static
echo "GPU: ${GPU_MODEL} | VRAM: ${VRAM_MB} MiB | SMs: ${SM_COUNT:-n/a} | driver: ${DRIVER_VERSION}" >&2

case "${MODE}" in
  live) run_live ;;
  benchmark) run_benchmark ;;
  *)
    echo "error: unknown mode ${MODE}" >&2
    exit 1
    ;;
esac

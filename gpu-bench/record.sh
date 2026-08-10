#!/usr/bin/env bash
# Sample GPU miner performance and append one row to results.csv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${SCRIPT_DIR}/.run"
ENV_FILE="${SCRIPT_DIR}/.env"
RESULTS_CSV="${RESULTS_CSV:-${SCRIPT_DIR}/results.csv}"
CSV_HEADER="timestamp,cloud_provider,gpu_model,vram_mb,sm_count,driver_version,hashrate,gpu_utilization_pct,cost_per_hour,cost_per_sec,hash_per_dollar,sample_seconds,wind_up_ms,busy_ms,wind_down_ms,notes"
DEFAULT_MINER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="live"
PROVIDER=""
COST_PER_HOUR=""
DURATION=60
NOTES=""
DRY_RUN=0
GPU_DEVICES_FLAG=""
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-}"
# Space-separated list; when set with --benchmark, runs one row per size.
BATCH_SIZES="${BATCH_SIZES:-}"
JOB_INTERVAL="${JOB_INTERVAL:-0}"
JOB_JITTER="${JOB_JITTER:-0.2}"
DIFFICULTY="${DIFFICULTY:-}"

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
  --gpu-batch-size N     Single GPU batch size for --benchmark
  --batch-sizes "N N"    Sweep several batch sizes (one CSV row each)
  --job-interval SECONDS Simulated NewJob period (0 = sustained; default 0)
  --job-jitter FRAC      ±fraction of job-interval (default 0.2; 0 = metronomic)
  --difficulty DEC|max   Difficulty for job simulation
  --notes TEXT           Optional notes column
  --dry-run              Print the CSV row but do not append
  -h, --help             Show this help

Examples:
  ./record.sh --provider vast.ai --cost-per-hour 0.35
  ./record.sh --benchmark --provider runpod --cost-per-hour 0.42 --duration 30
  ./record.sh --benchmark --job-interval 2 --batch-sizes "262144 524288 1000000" \
      --provider runpod --cost-per-hour 0.39
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
  if [[ ! -f "${RESULTS_CSV}" || ! -s "${RESULTS_CSV}" ]]; then
    printf '%s\n' "${CSV_HEADER}" >"${RESULTS_CSV}"
    return
  fi
  local first
  first="$(head -n 1 "${RESULTS_CSV}")"
  if [[ "${first}" == "${CSV_HEADER}" ]]; then
    return
  fi
  # Migrate older schemas by inserting empty phase columns before notes.
  local tmp
  tmp="$(mktemp)"
  python3 - "${RESULTS_CSV}" "${tmp}" "${CSV_HEADER}" <<'PY'
import csv, sys
src, dst, new_header = sys.argv[1], sys.argv[2], sys.argv[3]
new_fields = new_header.split(",")
with open(src, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=new_fields, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in new_fields})
PY
  mv "${tmp}" "${RESULTS_CSV}"
  echo "Migrated ${RESULTS_CSV} header → include wind_up_ms,busy_ms,wind_down_ms" >&2
}

# Best-effort SM / multiprocessor count. Many cloud drivers omit
# --query-gpu=multiprocessor_count (fails the whole CSV row), so we try
# several sources after name/VRAM are already known.
query_sm_count() {
  local name="${1:-}"
  local sm=""

  sm="$(nvidia-smi --query-gpu=multiprocessor_count --format=csv,noheader,nounits 2>/dev/null \
    | head -n 1 | tr -d '[:space:]')"
  if [[ "${sm}" =~ ^[0-9]+$ ]]; then
    echo "${sm}"
    return
  fi

  sm="$(nvidia-smi -q 2>/dev/null \
    | awk -F: 'tolower($0) ~ /multiprocessor count/ {
        gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit
      }')"
  if [[ "${sm}" =~ ^[0-9]+$ ]]; then
    echo "${sm}"
    return
  fi

  sm="$(nvidia-smi -q -x 2>/dev/null \
    | sed -n 's/.*<multiprocessor_count>\([0-9][0-9]*\)<\/multiprocessor_count>.*/\1/p' \
    | head -n 1)"
  if [[ "${sm}" =~ ^[0-9]+$ ]]; then
    echo "${sm}"
    return
  fi

  # Static fallback for common RunPod SKUs (architecture SM counts).
  case "${name}" in
    *"RTX 3070"*) echo 46 ;;
    *"RTX 3080 Ti"*) echo 80 ;;
    *"RTX 3080"*) echo 68 ;;
    *"RTX 3090 Ti"*) echo 84 ;;
    *"RTX 3090"*) echo 82 ;;
    *"RTX 4070 Ti"*) echo 60 ;;
    *"RTX 4080 SUPER"*) echo 80 ;;
    *"RTX 4080"*) echo 76 ;;
    *"RTX 4090"*) echo 128 ;;
    *"RTX 5080"*) echo 84 ;;
    *"RTX 5090"*) echo 170 ;;
    *"RTX A2000"*) echo 26 ;;
    *"RTX A4000"*) echo 48 ;;
    *"RTX A4500"*) echo 56 ;;
    *"RTX A5000"*) echo 64 ;;
    *"RTX A6000"*) echo 84 ;;
    *"RTX 2000 Ada"*) echo 22 ;;
    *"RTX 4000 Ada"*|*"RTX 4000 SFF Ada"*) echo 48 ;;
    *"RTX 5000 Ada"*) echo 100 ;;
    *"RTX 6000 Ada"*) echo 142 ;;
    *"RTX PRO 4000"*) echo 48 ;;
    *"RTX PRO 4500"*) echo 80 ;;
    *"RTX PRO 5000"*) echo 140 ;;
    *"RTX PRO 6000"*) echo 188 ;;
    *"NVIDIA L4"|*" L4") echo 60 ;;
    *"L40S"*) echo 142 ;;
    *"L40"*) echo 142 ;;
    *"NVIDIA A40"|*" A40") echo 84 ;;
    *"A100"*) echo 108 ;;
    *"H100 NVL"*) echo 132 ;;
    *"H100"*) echo 132 ;;
    *"H200"*) echo 132 ;;
    *"B200"*) echo 160 ;;
    *"B300"*) echo 160 ;;
    *"V100"*) echo 80 ;;
    *) echo "" ;;
  esac
}

query_gpu_static() {
  # Query name/VRAM/driver without multiprocessor_count — some drivers (e.g. 580.x)
  # reject that field and, under pipefail, abort before any fallback. SM count is
  # filled separately via query_sm_count (nvidia-smi / -q / static table).
  local line
  SM_COUNT=""
  if ! line="$(nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader,nounits 2>/dev/null | head -n 1)" || [[ -z "${line}" ]]; then
    echo "error: failed to query GPU via nvidia-smi" >&2
    exit 1
  fi
  GPU_MODEL="$(echo "${line}" | awk -F', ' '{print $1}')"
  VRAM_MB="$(echo "${line}" | awk -F', ' '{print $2}' | tr -d ' ')"
  DRIVER_VERSION="$(echo "${line}" | awk -F', ' '{print $3}' | tr -d ' ')"
  SM_COUNT="$(query_sm_count "${GPU_MODEL}")"
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

# Parse PHASE_MS lines from benchmark stdout. Averages across devices.
# Prints three lines: wind_up_ms, busy_ms, wind_down_ms (empty if absent).
parse_phase_timings() {
  local out="$1"
  echo "${out}" | awk '
    /^PHASE_MS / {
      wu = bu = wd = "";
      for (i = 1; i <= NF; i++) {
        split($i, a, "=");
        if (a[1] == "wind_up") wu = a[2];
        else if (a[1] == "busy") bu = a[2];
        else if (a[1] == "wind_down") wd = a[2];
      }
      if (wu != "" && bu != "" && wd != "") {
        sum_wu += wu + 0; sum_bu += bu + 0; sum_wd += wd + 0; n++;
      }
    }
    END {
      if (n > 0) {
        printf "%.2f\n%.2f\n%.2f\n", sum_wu / n, sum_bu / n, sum_wd / n;
      }
    }
  '
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

# From cost_per_hour ($/hr) and hashrate (H/s):
#   cost_per_sec     = cost_per_hour / 3600
#   hash_per_dollar  = hashrate / cost_per_sec    (hashes per $)
compute_cost_metrics() {
  local hashrate="$1"
  local cost_per_hour="$2"
  awk -v h="${hashrate}" -v c="${cost_per_hour}" 'BEGIN {
    if (c+0 <= 0) { print ""; print ""; exit }
    cps = c / 3600
    hpd = h / cps
    printf "%.10f\n%.6f\n", cps, hpd
  }'
}

emit_row() {
  local timestamp="$1"
  local hashrate="$2"
  local util_avg="$3"
  local cost_per_sec="$4"
  local hash_per_dollar="$5"
  local wind_up_ms="${6:-}"
  local busy_ms="${7:-}"
  local wind_down_ms="${8:-}"

  local row
  row="$(csv_escape "${timestamp}"),$(csv_escape "${PROVIDER}"),$(csv_escape "${GPU_MODEL}"),$(csv_escape "${VRAM_MB}"),$(csv_escape "${SM_COUNT}"),$(csv_escape "${DRIVER_VERSION}"),$(csv_escape "${hashrate}"),$(csv_escape "${util_avg}"),$(csv_escape "${COST_PER_HOUR}"),$(csv_escape "${cost_per_sec}"),$(csv_escape "${hash_per_dollar}"),$(csv_escape "${DURATION}"),$(csv_escape "${wind_up_ms}"),$(csv_escape "${busy_ms}"),$(csv_escape "${wind_down_ms}"),$(csv_escape "${NOTES}")"

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

  local cost_per_sec hash_per_dollar
  {
    read -r cost_per_sec
    read -r hash_per_dollar
  } < <(compute_cost_metrics "${hashrate}" "${COST_PER_HOUR}")

  local ts
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  # Live Prometheus path has no phase timings.
  emit_row "${ts}" "${hashrate}" "${util_avg}" "${cost_per_sec}" "${hash_per_dollar}" "" "" ""
}

run_benchmark_once() {
  local miner_bin="$1"
  local gpu_devices="$2"
  local batch_size="$3"
  local notes_extra="$4"

  local util_file bench_log
  util_file="$(mktemp)"
  bench_log="$(mktemp)"

  local batch_label="default"
  if [[ -n "${batch_size}" ]]; then
    batch_label="${batch_size}"
  fi

  local bench_cmd=(
    "${miner_bin}" benchmark
    --cpu-workers 0
    --gpu-devices "${gpu_devices}"
    --duration "${DURATION}"
  )
  if [[ -n "${batch_size}" ]]; then
    bench_cmd+=(--gpu-batch-size "${batch_size}")
  fi
  if awk -v j="${JOB_INTERVAL}" 'BEGIN { exit !(j+0 > 0) }'; then
    bench_cmd+=(--job-interval "${JOB_INTERVAL}" --job-jitter "${JOB_JITTER}")
  fi
  if [[ -n "${DIFFICULTY}" ]]; then
    bench_cmd+=(--difficulty "${DIFFICULTY}")
  fi

  echo "Running GPU benchmark for ${DURATION}s (gpu-devices=${gpu_devices}, batch=${batch_label}, job_interval=${JOB_INTERVAL}, job_jitter=${JOB_JITTER}) ..." >&2
  sample_util_during "${util_file}" "${DURATION}" &
  local sampler_pid=$!

  set +e
  "${bench_cmd[@]}" >"${bench_log}" 2>&1
  local bench_rc=$?
  set -e

  wait "${sampler_pid}" 2>/dev/null || true

  if [[ "${bench_rc}" -ne 0 ]]; then
    echo "error: benchmark failed (exit ${bench_rc}, batch=${batch_label}). Output:" >&2
    cat "${bench_log}" >&2
    rm -f "${util_file}" "${bench_log}"
    return 1
  fi

  cat "${bench_log}" >&2

  local hashrate util_avg
  hashrate="$(parse_benchmark_hashrate "$(cat "${bench_log}")")"
  if [[ -z "${hashrate}" ]]; then
    echo "error: could not parse hashrate from benchmark output (batch=${batch_label})" >&2
    rm -f "${util_file}" "${bench_log}"
    return 1
  fi
  util_avg="$(average_list <"${util_file}")"
  if [[ -n "${util_avg}" ]]; then
    util_avg="$(awk -v u="${util_avg}" 'BEGIN { printf "%.2f", u }')"
  fi

  local cost_per_sec hash_per_dollar
  {
    read -r cost_per_sec
    read -r hash_per_dollar
  } < <(compute_cost_metrics "${hashrate}" "${COST_PER_HOUR}")

  local wind_up_ms="" busy_ms="" wind_down_ms=""
  local phase_out
  phase_out="$(parse_phase_timings "$(cat "${bench_log}")" || true)"
  if [[ -n "${phase_out}" ]]; then
    {
      read -r wind_up_ms
      read -r busy_ms
      read -r wind_down_ms
    } <<<"${phase_out}"
  fi

  local saved_notes="${NOTES}"
  if [[ -n "${notes_extra}" ]]; then
    if [[ -n "${NOTES}" ]]; then
      NOTES="${NOTES};${notes_extra}"
    else
      NOTES="${notes_extra}"
    fi
  fi

  local ts
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  emit_row "${ts}" "${hashrate}" "${util_avg}" "${cost_per_sec}" "${hash_per_dollar}" \
    "${wind_up_ms}" "${busy_ms}" "${wind_down_ms}"
  NOTES="${saved_notes}"

  rm -f "${util_file}" "${bench_log}"
  return 0
}

run_benchmark() {
  local miner_bin
  miner_bin="$(resolve_miner_bin)"
  local gpu_devices="${GPU_DEVICES_FLAG:-${GPU_DEVICES:-1}}"

  local sizes=()
  if [[ -n "${BATCH_SIZES}" ]]; then
    # shellcheck disable=SC2206
    sizes=(${BATCH_SIZES})
  elif [[ -n "${GPU_BATCH_SIZE}" ]]; then
    sizes=("${GPU_BATCH_SIZE}")
  else
    sizes=("")
  fi

  local bs note_extra
  local any_ok=0
  for bs in "${sizes[@]}"; do
    note_extra=""
    if [[ -n "${bs}" ]]; then
      note_extra="batch=${bs}"
    fi
    if awk -v j="${JOB_INTERVAL}" 'BEGIN { exit !(j+0 > 0) }'; then
      if [[ -n "${note_extra}" ]]; then
        note_extra="${note_extra};job_interval=${JOB_INTERVAL};job_jitter=${JOB_JITTER}"
      else
        note_extra="job_interval=${JOB_INTERVAL};job_jitter=${JOB_JITTER}"
      fi
      if [[ -n "${DIFFICULTY}" ]]; then
        note_extra="${note_extra};difficulty=${DIFFICULTY}"
      fi
    fi
    if run_benchmark_once "${miner_bin}" "${gpu_devices}" "${bs}" "${note_extra}"; then
      any_ok=1
    fi
  done

  if [[ "${any_ok}" -ne 1 ]]; then
    echo "error: all benchmark runs failed" >&2
    exit 1
  fi
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
    --gpu-batch-size)
      GPU_BATCH_SIZE="${2:-}"
      shift 2
      ;;
    --batch-sizes)
      BATCH_SIZES="${2:-}"
      shift 2
      ;;
    --job-interval)
      JOB_INTERVAL="${2:-}"
      shift 2
      ;;
    --job-jitter)
      JOB_JITTER="${2:-}"
      shift 2
      ;;
    --difficulty)
      DIFFICULTY="${2:-}"
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

#!/usr/bin/env bash
# A/B GPU batch sizes on a pod (or any host with a built miner + NVIDIA GPU).
# Samples nvidia-smi util during each run and prints a comparison table.
#
# Usage (on pod, after miner is built):
#   cd /workspace/quantus-gpu-bench
#   ./batch-tune.sh
#   ./batch-tune.sh --sizes "1000000 4194304 16777216" --duration 30
#
# Or from laptop via runpod-shell (see comments at bottom of this file).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-/workspace/quantus-gpu-bench}"
BIN_DIR="${BIN_DIR:-${WORK_DIR}/bin}"
OUT_CSV="${OUT_CSV:-${WORK_DIR}/batch-tune.csv}"
SIZES="${SIZES:-1000000 4194304 8388608 16777216}"
DURATION="${DURATION:-30}"
GPU_DEVICES="${GPU_DEVICES:-1}"
CPU_WORKERS="${CPU_WORKERS:-0}"
MINER_BIN="${MINER_BIN:-}"

usage() {
  cat <<'EOF'
Usage: ./batch-tune.sh [options]

  --sizes "N N N"     batch sizes to try (default: 1M 4M 8M 16M)
  --duration SECONDS  per-size benchmark window (default: 30)
  --gpu-devices N     default 1
  --cpu-workers N     default 0 (GPU-only)
  --miner-bin PATH    override miner binary
  --out PATH          results CSV (default: $WORK_DIR/batch-tune.csv)
  --help

Requires a miner built with the gpu_chunk=batch_size CLI fix (illuzen/gpu-bench).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sizes) SIZES="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --gpu-devices) GPU_DEVICES="$2"; shift 2 ;;
    --cpu-workers) CPU_WORKERS="$2"; shift 2 ;;
    --miner-bin) MINER_BIN="$2"; shift 2 ;;
    --out) OUT_CSV="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

resolve_miner() {
  if [[ -n "${MINER_BIN}" && -x "${MINER_BIN}" ]]; then
    echo "${MINER_BIN}"
    return
  fi
  for cand in \
    "${BIN_DIR}/quantus-miner" \
    "${WORK_DIR}/quantus-miner/target/release/quantus-miner" \
    "${SCRIPT_DIR}/../target/release/quantus-miner" \
    "$(command -v quantus-miner 2>/dev/null || true)"; do
    if [[ -n "${cand}" && -x "${cand}" ]]; then
      echo "${cand}"
      return
    fi
  done
  echo "error: quantus-miner binary not found. Run remote-run.sh once, or set --miner-bin" >&2
  exit 1
}

sample_util() {
  # Average GPU util % over ~DURATION seconds (1 Hz).
  local secs="$1"
  local sum=0 count=0 u
  for ((i = 0; i < secs; i++)); do
    u="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || echo 0)"
    [[ "${u}" =~ ^[0-9]+$ ]] || u=0
    sum=$((sum + u))
    count=$((count + 1))
    sleep 1
  done
  if [[ "${count}" -eq 0 ]]; then
    echo 0
  else
    echo $((sum / count))
  fi
}

parse_rate_hs() {
  # Parse "Average rate: 1.23M H/s" / "45.67K H/s" / "1234 H/s" from benchmark stdout.
  local line unit num
  line="$(grep -E 'Average rate:' "$1" | tail -n1 || true)"
  [[ -n "${line}" ]] || { echo 0; return; }
  num="$(echo "${line}" | sed -E 's/.*Average rate:[[:space:]]*([0-9.]+)([KMkm]?).*H\/s.*/\1/')"
  unit="$(echo "${line}" | sed -E 's/.*Average rate:[[:space:]]*[0-9.]+([KMkm]?).*H\/s.*/\1/')"
  case "${unit}" in
    M|m) awk -v n="${num}" 'BEGIN { printf "%.0f", n * 1000000 }' ;;
    K|k) awk -v n="${num}" 'BEGIN { printf "%.0f", n * 1000 }' ;;
    *) awk -v n="${num}" 'BEGIN { printf "%.0f", n }' ;;
  esac
}

MINER_BIN="$(resolve_miner)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | sed 's/,/;/g' || echo unknown)"
echo "miner:  ${MINER_BIN}"
echo "gpu:    ${GPU_NAME}"
echo "sizes:  ${SIZES}"
echo "window: ${DURATION}s  gpu_devices=${GPU_DEVICES} cpu_workers=${CPU_WORKERS}"
echo

mkdir -p "$(dirname "${OUT_CSV}")"
if [[ ! -f "${OUT_CSV}" ]]; then
  echo "timestamp,gpu_name,batch_size,duration_s,hashrate_hs,avg_util_pct,notes" >"${OUT_CSV}"
fi

printf "%-12s  %-14s  %-10s\n" "batch_size" "hashrate" "avg_util%"
printf "%-12s  %-14s  %-10s\n" "----------" "--------" "---------"

for bs in ${SIZES}; do
  log="$(mktemp)"
  util_log="$(mktemp)"
  # Sample util in background while benchmark runs.
  (
    # skip first ~3s of warmup
    sleep 3
    sample_util $((DURATION > 5 ? DURATION - 3 : DURATION))
  ) >"${util_log}" &
  util_pid=$!

  set +e
  "${MINER_BIN}" benchmark \
    --gpu-devices "${GPU_DEVICES}" \
    --cpu-workers "${CPU_WORKERS}" \
    --gpu-batch-size "${bs}" \
    --duration "${DURATION}" \
    >"${log}" 2>&1
  rc=$?
  set -e

  wait "${util_pid}" 2>/dev/null || true
  util="$(cat "${util_log}" 2>/dev/null || echo 0)"
  rate="$(parse_rate_hs "${log}")"

  if [[ "${rc}" -ne 0 || "${rate}" == "0" ]]; then
    echo "--- failed batch_size=${bs} (rc=${rc}) ---" >&2
    tail -n 40 "${log}" >&2 || true
    notes="FAILED"
    rate_disp="FAIL"
  else
    notes="ok"
    if (( rate >= 1000000 )); then
      rate_disp="$(awk -v n="${rate}" 'BEGIN { printf "%.2fM" , n/1000000 }')"
    elif (( rate >= 1000 )); then
      rate_disp="$(awk -v n="${rate}" 'BEGIN { printf "%.2fK" , n/1000 }')"
    else
      rate_disp="${rate}"
    fi
  fi

  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "${ts},${GPU_NAME},${bs},${DURATION},${rate},${util},${notes}" >>"${OUT_CSV}"
  printf "%-12s  %-14s  %-10s\n" "${bs}" "${rate_disp}" "${util}%"

  rm -f "${log}" "${util_log}"
done

echo
echo "wrote ${OUT_CSV}"
echo "Best util/rate usually wins — if 16M ≈ 1M, dispatch shape is not the bottleneck."

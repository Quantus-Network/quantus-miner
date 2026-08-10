#!/usr/bin/env bash
# Orchestrate Clore.ai rentals via API: pick offers for a GPU model, create
# order → SSH → remote-run.sh (miner build + benchmark sweep → CSV) → scp →
# cancel. Bad hosts (no NVIDIA GL/Vulkan userspace mounted) are screened with a
# 30-second probe and skipped for the next candidate, so a doomed host costs the
# creation fee (~$0.10) instead of a 15-minute build.
#
# Clore notes vs RunPod:
#   - Marketplace prices are per DAY; cost_per_hour is derived (price/24).
#   - Payment currency is USD-Blockchain (account must hold USD balance).
#   - create_order returns only {"code":0}; the order id + SSH endpoint are
#     resolved by polling my_orders for the rented server id.
#   - Some hosts mount only the CUDA compute userspace. Vulkan is unfixable
#     there (exact-version driver extract still fails vkCreateInstance), so we
#     probe for libGLX_nvidia.so.0 right after SSH and re-rent elsewhere.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_BASE="${CLORE_API_BASE:-https://api.clore.ai/v1}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/sweep-out}"
RESULTS_CSV="${RESULTS_CSV:-${SCRIPT_DIR}/results.csv}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519}"
IMAGE_NAME="${IMAGE_NAME:-cloreai/jupyter:ubuntu24.04-v2}"
GPU_COUNT="${GPU_COUNT:-1}"          # rig size to rent (Nx GPU offers)
ISOLATED="${ISOLATED:-1}"            # rigs >1 GPU: also run a 1-GPU pass
MAX_PRICE_PER_DAY="${MAX_PRICE_PER_DAY:-}"
MIN_RELIABILITY="${MIN_RELIABILITY:-0.97}"
MIN_RATING="${MIN_RATING:-4}"
HOST_RETRIES="${HOST_RETRIES:-4}"    # candidate offers to try per GPU model
DURATION="${DURATION:-30}"
BATCH_SIZES="${BATCH_SIZES:-262144 524288 1000000 4194304}"
JOB_INTERVAL="${JOB_INTERVAL:-2}"
JOB_JITTER="${JOB_JITTER:-0.2}"
DIFFICULTY="${DIFFICULTY:-}"
MINER_SOURCE="${MINER_SOURCE:-git}"
MINER_REPO="${MINER_REPO:-https://github.com/Quantus-Network/quantus-miner.git}"
MINER_BRANCH="${MINER_BRANCH:-illuzen/gpu-bench}"
KEEP_ON_FAILURE="${KEEP_ON_FAILURE:-0}"
SSH_WAIT_SECONDS="${SSH_WAIT_SECONDS:-360}"
REMOTE_DIR="/root/quantus-gpu-bench"
EXIT_COMPUTE_ONLY=42

usage() {
  cat <<'EOF'
Usage: ./clore-sweep.sh "GPU NAME" ["GPU NAME" ...]
       ./clore-sweep.sh --list "GPU NAME"        # print matching offers, rent nothing
       ./clore-sweep.sh --server ID --price-per-day USD

GPU names match the marketplace listing minus the count prefix, e.g.
"NVIDIA GeForce RTX 4090" (or a substring: "RTX 4090").

Environment:
  CLORE_API_KEY       Required (clore.ai dashboard → settings → API), or a
                      token file at ~/.config/clore/token
  GPU_COUNT           Rig size to rent, default 1 (e.g. 2 = "2x ..." offers)
  ISOLATED            1 (default): rigs >1 GPU also run a --gpu-devices 1 pass
  MAX_PRICE_PER_DAY   Skip offers above this USD/day price
  MIN_RELIABILITY     Offer filter, default 0.97
  MIN_RATING          Offer filter, default 4
  HOST_RETRIES        Candidate offers to try per GPU model (default 4)
  IMAGE_NAME          Docker image (default cloreai/jupyter:ubuntu24.04-v2)
  SSH_KEY             Default ~/.ssh/id_ed25519 (pubkey is sent with the order)
  DURATION/BATCH_SIZES/JOB_INTERVAL/JOB_JITTER/DIFFICULTY
                      Passed through to remote-run.sh
  MINER_SOURCE/REPO/BRANCH
                      Miner build source (default git @ illuzen/gpu-bench)
  RESULTS_CSV         Collaborative dataset to append (default ./results.csv)
  KEEP_ON_FAILURE     1 = leave the rental up after a benchmark failure

Requires: curl, ssh, scp, python3.
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    exit 1
  fi
}

api() {
  curl -fsS -H "auth: ${CLORE_API_KEY}" -H "Content-type: application/json" "$@"
}

# Print candidate offers "server_id price_per_day", cheapest first.
find_candidates() {
  local gpu_name="$1"
  local mp
  mp="$(mktemp)"
  # shellcheck disable=SC2064
  trap "rm -f '${mp}'" RETURN
  api "${API_BASE}/marketplace" -o "${mp}"
  python3 - "${mp}" "$gpu_name" "${GPU_COUNT}" \
    "${MIN_RELIABILITY}" "${MIN_RATING}" "${MAX_PRICE_PER_DAY:-inf}" <<'PY'
import json, re, sys
mp, name, count, min_rel, min_rate, max_price = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]),
    float(sys.argv[6]))
d = json.load(open(mp))
cands = []
for s in d.get("servers", []):
    if s.get("rented"):
        continue
    if "USD-Blockchain" not in s.get("allowed_coins", []):
        continue
    m = re.match(r"(\d+)x (.+)", s["specs"].get("gpu", ""))
    if not m or int(m.group(1)) != count or name.lower() not in m.group(2).lower():
        continue
    usd = s["price"].get("on_demand", {}).get("USD-Blockchain")
    if not usd or usd > max_price:
        continue
    if (s.get("reliability") or 0) < min_rel:
        continue
    if (s.get("rating", {}).get("avg") or 0) < min_rate:
        continue
    cands.append((usd, s["id"]))
cands.sort()
for usd, sid in cands:
    print(sid, usd)
PY
}

ORDER_ID=""

cancel_order() {
  [[ -z "${ORDER_ID}" ]] && return 0
  local i r
  for i in 1 2 3 4 5; do
    r="$(api -X POST -d "{\"id\":${ORDER_ID}}" "${API_BASE}/cancel_order" 2>/dev/null)" || r=""
    if [[ "${r}" == '{"code":0}' ]]; then
      echo "cancelled order ${ORDER_ID}" >&2
      ORDER_ID=""
      return 0
    fi
    sleep 45 # API is rate-limited (~1 req/s); 429s here are common
  done
  echo "WARNING: could not cancel order ${ORDER_ID} — cancel it manually!" >&2
  ORDER_ID=""
  return 1
}
trap cancel_order EXIT

create_order() {
  local server_id="$1"
  local body
  body="$(python3 - "${server_id}" "${IMAGE_NAME}" "${SSH_KEY}.pub" <<'PY'
import json, pathlib, secrets, sys
print(json.dumps({
    "currency": "USD-Blockchain",
    "image": sys.argv[2],
    "renting_server": int(sys.argv[1]),
    "type": "on-demand",
    "ports": {"22": "tcp"},
    "env": {"NVIDIA_DRIVER_CAPABILITIES": "all"},
    "ssh_key": pathlib.Path(sys.argv[3]).read_text().strip(),
    "jupyter_token": secrets.token_hex(8)}))
PY
)"
  local resp
  resp="$(api -X POST -d "${body}" "${API_BASE}/create_order" 2>/dev/null)" || resp=""
  [[ "${resp}" == '{"code":0}' ]]
}

# Poll my_orders until the order for server_id has an SSH port.
# Prints "order_id host port".
resolve_ssh() {
  local server_id="$1"
  local i info
  for ((i = 0; i < 30; i++)); do
    sleep $((15 + RANDOM % 10))
    info="$(api "${API_BASE}/my_orders" 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for o in d.get("orders", []):
    if o.get("si") == int(sys.argv[1]):
        ports = dict(p.split(":") for p in o.get("tcp_ports", []))
        if o.get("pub_cluster") and ports.get("22"):
            print(o["id"], o["pub_cluster"][0], ports["22"])
        break
' "${server_id}")" || info=""
    if [[ -n "${info}" ]]; then
      echo "${info}"
      return 0
    fi
  done
  return 1
}

run_one_attempt() {
  local server_id="$1" price_per_day="$2" gpu_label="$3"
  local pph
  pph="$(python3 -c "print(round(${price_per_day}/24, 4))")"
  echo "renting server ${server_id} (\$${price_per_day}/day = \$${pph}/hr) ..." >&2

  if ! create_order "${server_id}"; then
    echo "create_order failed (likely rented out from under us)" >&2
    return 3
  fi

  local info host port
  if ! info="$(resolve_ssh "${server_id}")"; then
    echo "order never exposed an SSH endpoint" >&2
    ORDER_ID="$(api "${API_BASE}/my_orders" | python3 -c "
import json, sys
for o in json.load(sys.stdin).get('orders', []):
    if o.get('si') == ${server_id}:
        print(o['id'])" || true)"
    cancel_order
    return 3
  fi
  read -r ORDER_ID host port <<<"${info}"
  echo "order ${ORDER_ID}: ssh root@${host} -p ${port}" >&2

  local ssh_opts=(-i "${SSH_KEY}" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -p "${port}")
  local deadline=$((SECONDS + SSH_WAIT_SECONDS)) up=0
  while ((SECONDS < deadline)); do
    if ssh "${ssh_opts[@]}" "root@${host}" true 2>/dev/null; then
      up=1
      break
    fi
    sleep 15
  done
  if [[ "${up}" != 1 ]]; then
    echo "SSH never came up" >&2
    cancel_order
    return 3
  fi

  # Vulkan screening probe: no GL userspace mounted → unfixable, next host.
  if ! ssh "${ssh_opts[@]}" "root@${host}" \
    'ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 /usr/lib64/libGLX_nvidia.so.0 2>/dev/null | head -1 | grep -q .'; then
    echo "probe: compute-only host (no libGLX_nvidia mounted) — skipping" >&2
    cancel_order
    return "${EXIT_COMPUTE_ONLY}"
  fi
  echo "probe ok: NVIDIA GL userspace mounted" >&2

  local gpus_note="gpus=${GPU_COUNT}/${GPU_COUNT}"
  local remote_rc=0
  scp -q -O -i "${SSH_KEY}" -o StrictHostKeyChecking=accept-new -P "${port}" \
    "${SCRIPT_DIR}/remote-run.sh" "${SCRIPT_DIR}/record.sh" "root@${host}:/root/" || remote_rc=1

  if [[ "${remote_rc}" -eq 0 ]]; then
    ssh "${ssh_opts[@]}" "root@${host}" \
      "export DEBIAN_FRONTEND=noninteractive; apt-get update -qq >/dev/null 2>&1; \
       apt-get install -y -qq curl ca-certificates git python3 >/dev/null 2>&1; echo prep-ok" \
      || remote_rc=1
  fi

  if [[ "${remote_rc}" -eq 0 ]]; then
    local runs="${GPU_COUNT}"
    [[ "${GPU_COUNT}" -gt 1 && "${ISOLATED}" == "1" ]] && runs="${GPU_COUNT} 1"
    local g
    for g in ${runs}; do
      echo "--- remote-run --gpu-devices ${g}" >&2
      set +e
      ssh "${ssh_opts[@]}" "root@${host}" \
        "cd /root && WORK_DIR='${REMOTE_DIR}' ./remote-run.sh \
           --provider clore --cost-per-hour '${pph}' \
           --duration '${DURATION}' --batch-sizes '${BATCH_SIZES}' \
           --job-interval '${JOB_INTERVAL}' --job-jitter '${JOB_JITTER}' \
           ${DIFFICULTY:+--difficulty '${DIFFICULTY}'} \
           --gpu-devices '${g}' \
           --miner-source '${MINER_SOURCE}' \
           --miner-repo '${MINER_REPO}' \
           --miner-branch '${MINER_BRANCH}' \
           --notes 'clore_server=${server_id};gpus=${g}/${GPU_COUNT};miner=${MINER_SOURCE}@${MINER_BRANCH}'"
      remote_rc=$?
      set -e
      [[ "${remote_rc}" -ne 0 ]] && break
    done
  fi

  if [[ "${remote_rc}" -eq 0 ]]; then
    local local_row="${OUT_DIR}/row-clore-${ORDER_ID}.csv"
    scp -q -O -i "${SSH_KEY}" -P "${port}" "root@${host}:${REMOTE_DIR}/results.csv" "${local_row}"
    tail -n +2 "${local_row}" >>"${RESULTS_CSV}"
    echo "Appended results to ${RESULTS_CSV}" >&2
    cancel_order
    return 0
  fi

  echo "error: remote benchmark failed on server ${server_id} (rc=${remote_rc})" >&2
  if [[ "${KEEP_ON_FAILURE}" == "1" ]]; then
    echo "KEEP_ON_FAILURE=1 — rental stays up: ssh -i ${SSH_KEY} -p ${port} root@${host}" >&2
    echo "  (cancel it yourself: order ${ORDER_ID})" >&2
    ORDER_ID=""
    return "${remote_rc}"
  fi
  cancel_order
  return "${remote_rc}"
}

run_one() {
  local gpu_name="$1"
  echo "== ${gpu_name} (${GPU_COUNT}x): searching offers ..." >&2
  local cands
  cands="$(find_candidates "${gpu_name}")"
  if [[ -z "${cands}" ]]; then
    echo "error: no available offers match '${gpu_name}' with current filters" >&2
    return 1
  fi
  local tried=0 sid ppd rc=1
  while read -r sid ppd; do
    ((tried += 1))
    if ((tried > HOST_RETRIES)); then
      break
    fi
    set +e
    run_one_attempt "${sid}" "${ppd}" "${gpu_name}"
    rc=$?
    set -e
    if [[ "${rc}" -eq 0 ]]; then
      return 0
    fi
    echo "candidate ${tried}/${HOST_RETRIES} failed (rc=${rc}); trying next offer" >&2
  done <<<"${cands}"
  echo "error: exhausted candidates for ${gpu_name}" >&2
  return "${rc}"
}

main() {
  require_cmd curl
  require_cmd ssh
  require_cmd scp
  require_cmd python3

  if [[ -z "${CLORE_API_KEY:-}" && -f "${HOME}/.config/clore/token" ]]; then
    CLORE_API_KEY="$(cat "${HOME}/.config/clore/token")"
  fi
  if [[ -z "${CLORE_API_KEY:-}" ]]; then
    echo "error: set CLORE_API_KEY (or ~/.config/clore/token)" >&2
    exit 1
  fi
  if [[ ! -f "${SSH_KEY}.pub" ]]; then
    echo "error: SSH public key not found: ${SSH_KEY}.pub" >&2
    exit 1
  fi
  mkdir -p "${OUT_DIR}"

  local explicit_server="" explicit_price="" list_only=0
  local targets=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --server) explicit_server="${2:-}"; shift 2 ;;
      --price-per-day) explicit_price="${2:-}"; shift 2 ;;
      --list) list_only=1; shift ;;
      -h | --help) usage; exit 0 ;;
      *) targets+=("$1"); shift ;;
    esac
  done

  if [[ "${list_only}" == 1 ]]; then
    local t
    for t in "${targets[@]}"; do
      echo "== ${t} (${GPU_COUNT}x) — server_id price_per_day (cheapest first):"
      find_candidates "${t}"
    done
    exit 0
  fi

  if [[ -n "${explicit_server}" ]]; then
    if [[ -z "${explicit_price}" ]]; then
      echo "error: --server needs --price-per-day (for the cost columns)" >&2
      exit 1
    fi
    run_one_attempt "${explicit_server}" "${explicit_price}" "server-${explicit_server}"
    exit $?
  fi

  if [[ "${#targets[@]}" -eq 0 ]]; then
    usage >&2
    exit 1
  fi

  local t failed=0
  for t in "${targets[@]}"; do
    run_one "${t}" || failed=1
  done
  exit "${failed}"
}

main "$@"

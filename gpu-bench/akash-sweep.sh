#!/usr/bin/env bash
# Orchestrate Akash Network GPU rentals via the Console Managed Wallet API:
#   create SDL deployment → wait for bids → lease cheapest → resolve SSH
#   (forwarded port 22) → remote-run.sh → scp CSV → close deployment.
#
# Auth: Console API key (Settings → API Keys). Billing is USD on the managed
# wallet (credit card); on-chain bid prices are converted to $/hr for the CSV.
#
# Unlike Clore/RunPod there is no offer catalog — --list creates a short-lived
# deployment, prints bids, and closes without leasing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_BASE="${AKASH_API_BASE:-https://console-api.akash.network}"
PROXY_BASE="${AKASH_PROVIDER_PROXY:-https://console.akash.network/provider-proxy-mainnet}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/sweep-out}"
RESULTS_CSV="${RESULTS_CSV:-${SCRIPT_DIR}/results.csv}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519}"
# CUDA devel on Ubuntu 24.04 (matches remote-run GLIBC needs). Providers must
# expose the NVIDIA container runtime.
IMAGE_NAME="${IMAGE_NAME:-nvidia/cuda:12.6.3-devel-ubuntu24.04}"
GPU_COUNT="${GPU_COUNT:-1}"
CPU_UNITS="${CPU_UNITS:-8}"
MEMORY_SIZE="${MEMORY_SIZE:-32Gi}"
STORAGE_SIZE="${STORAGE_SIZE:-50Gi}"
# Max bid price in SDL (chain micro-units per block). High enough to attract
# GPU bids; the actual lease price is the provider's bid.
MAX_PRICE_AMOUNT="${MAX_PRICE_AMOUNT:-1000000}"
# Console managed wallet uses uact (not legacy uakt).
PRICE_DENOM="${PRICE_DENOM:-uact}"
DEPOSIT_USD="${DEPOSIT_USD:-5}"
AKT_USD="${AKT_USD:-}"                 # optional; fetched if empty
BLOCK_TIME_SEC="${BLOCK_TIME_SEC:-6}"
BID_WAIT_SECONDS="${BID_WAIT_SECONDS:-90}"
SSH_WAIT_SECONDS="${SSH_WAIT_SECONDS:-600}"
HOST_RETRIES="${HOST_RETRIES:-3}"
DURATION="${DURATION:-30}"
BATCH_SIZES="${BATCH_SIZES:-262144 524288 1000000 4194304}"
JOB_INTERVAL="${JOB_INTERVAL:-2}"
JOB_JITTER="${JOB_JITTER:-0.2}"
DIFFICULTY="${DIFFICULTY:-}"
MINER_SOURCE="${MINER_SOURCE:-git}"
MINER_REPO="${MINER_REPO:-https://github.com/Quantus-Network/quantus-miner.git}"
MINER_BRANCH="${MINER_BRANCH:-illuzen/gpu-bench}"
KEEP_ON_FAILURE="${KEEP_ON_FAILURE:-0}"
REMOTE_DIR="/root/quantus-gpu-bench"
SERVICE_NAME="gpu"
EXIT_COMPUTE_ONLY=42

DSEQ=""
SKIP_PROVIDERS=()

usage() {
  cat <<'EOF'
Usage: ./akash-sweep.sh "GPU MODEL" ["GPU MODEL" ...]
       ./akash-sweep.sh --list "rtx4090"     # create deploy, print bids, close
       ./akash-sweep.sh --model rtx4090      # same as positional
       ./akash-sweep.sh --gpus-file gpus.all.txt
       ./akash-sweep.sh --list --gpus-file gpus.all.txt   # bids only

GPU models are Akash SDL names (lowercase, no spaces): rtx4090, rtx5080,
rtx3090, a100, h100, p4, …  Human / RunPod strings ("NVIDIA GeForce RTX 4090")
are normalized. --gpus-file skips blank/# lines, normalizes, and dedupes.

Note: gpus.all.txt is a RunPod inventory. Many names have no Akash bids
(0-bid models are skipped; each attempt still spends a small deposit). Prefer
--list first, or use a shorter akash-oriented list.

Environment:
  AKASH_API_KEY       Required (console.akash.network → Settings → API Keys),
                      or a token file at ~/.config/akash/api_key
  DEPOSIT_USD         Escrow deposit per deployment (default 5; min 0.5)
  MAX_PRICE_AMOUNT    SDL max price per block in PRICE_DENOM (default 1000000)
  PRICE_DENOM         uact (default; Console managed wallet) or uusdc
  AKT_USD             USD per AKT for $/hr conversion (auto-fetched if unset)
  IMAGE_NAME          Container image (default nvidia/cuda:12.6.3-devel-ubuntu24.04)
  GPU_COUNT           GPUs in the SDL request (default 1)
  CPU_UNITS/MEMORY_SIZE/STORAGE_SIZE
                      Compute profile (defaults 8 / 32Gi / 50Gi)
  HOST_RETRIES        Redeploy attempts if host is bad (default 3)
  SSH_KEY             Default ~/.ssh/id_ed25519 (pubkey injected into the pod)
  DURATION/BATCH_SIZES/JOB_INTERVAL/JOB_JITTER/DIFFICULTY
                      Passed through to remote-run.sh
  MINER_SOURCE/REPO/BRANCH
  RESULTS_CSV         Collaborative dataset (default ./results.csv)
  KEEP_ON_FAILURE=1   Leave deployment up after a failed benchmark

Requires: curl, ssh, scp, python3.
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    exit 1
  fi
}

# usage: api METHOD PATH [curl -d args…]
# On HTTP error, prints body to stderr and returns non-zero (no silent -f).
api() {
  local method="$1" path="$2"
  shift 2
  local tmp http
  tmp="$(mktemp)"
  http="$(curl -sS -X "${method}" \
    -H "x-api-key: ${AKASH_API_KEY}" \
    -H "Content-Type: application/json" \
    -o "${tmp}" -w "%{http_code}" \
    "${API_BASE}${path}" "$@")" || true
  if [[ "${http}" -lt 200 || "${http}" -ge 300 ]]; then
    echo "error: ${method} ${path} → HTTP ${http}" >&2
    sed 's/^/  /' "${tmp}" >&2 || true
    rm -f "${tmp}"
    return 1
  fi
  cat "${tmp}"
  rm -f "${tmp}"
}

close_deployment() {
  [[ -z "${DSEQ}" ]] && return 0
  local dseq="${DSEQ}"
  DSEQ=""
  echo "closing deployment dseq=${dseq} ..." >&2
  api DELETE "/v1/deployments/${dseq}" >/dev/null 2>&1 \
    || echo "WARNING: failed to close dseq=${dseq} — close it in the Console" >&2
}
trap close_deployment EXIT

normalize_model() {
  python3 -c '
import re, sys
s = sys.argv[1].lower()
for junk in ("nvidia ", "geforce ", "tesla ", "generation", "sff "):
    s = s.replace(junk, " ")
raw = re.sub(r"[^a-z0-9]+", "", s)
# Map RunPod-ish labels onto common Akash SDL model ids.
# Longer needles first (a4000 before a40, l40s before l40, …).
families = (
    ("b300", "b300"),
    ("b200", "b200"),
    ("h200", "h200"),
    ("h100", "h100"),
    ("a100", "a100"),
    ("a6000", "a6000"),
    ("a5000", "a5000"),
    ("a4500", "a4500"),
    ("a4000", "a4000"),
    ("a2000", "a2000"),
    ("a40", "a40"),
    ("l40s", "l40s"),
    ("l40", "l40"),
    ("l4", "l4"),
    ("v100", "v100"),
    ("t4", "t4"),
    ("p40", "p40"),
    ("p4", "p4"),
)
for needle, name in families:
    if needle in raw:
        print(name)
        raise SystemExit
# Consumer / pro GeForce-style: keep rtx4090, rtx5080, …
if raw.startswith("rtx") or re.match(r"^\d{4}", raw):
    print(raw)
    raise SystemExit
print(raw)
' "$1"
}

# Load targets from a gpus*.txt (RunPod-style or Akash ids). Prints normalized ids.
load_gpus_file() {
  local file="$1"
  python3 - "$file" <<'PY'
import re, sys
from pathlib import Path

def normalize(name: str) -> str:
    s = name.lower()
    for junk in ("nvidia ", "geforce ", "tesla ", "generation", "sff "):
        s = s.replace(junk, " ")
    raw = re.sub(r"[^a-z0-9]+", "", s)
    for needle, out in (
        ("b300", "b300"), ("b200", "b200"), ("h200", "h200"), ("h100", "h100"),
        ("a100", "a100"), ("a6000", "a6000"), ("a5000", "a5000"),
        ("a4500", "a4500"), ("a4000", "a4000"), ("a2000", "a2000"), ("a40", "a40"),
        ("l40s", "l40s"), ("l40", "l40"), ("l4", "l4"),
        ("v100", "v100"), ("t4", "t4"), ("p40", "p40"), ("p4", "p4"),
    ):
        if needle in raw:
            return out
    return raw

seen = set()
for line in Path(sys.argv[1]).read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    m = normalize(line)
    if not m or m in seen:
        continue
    seen.add(m)
    print(m)
PY
}

resolve_akt_usd() {
  if [[ -n "${AKT_USD}" ]]; then
    echo "${AKT_USD}"
    return
  fi
  local px
  px="$(curl -fsS "https://api.coingecko.com/api/v3/simple/price?ids=akash-network&vs_currencies=usd" 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["akash-network"]["usd"])' 2>/dev/null)" || px=""
  if [[ -z "${px}" ]]; then
    # A made-up rate would silently poison every $/hr and hash_per_dollar row.
    echo "error: could not fetch AKT/USD price; set AKT_USD=<rate> and retry" >&2
    return 1
  fi
  echo "${px}"
}

# Build SDL JSON body for POST /v1/deployments. Prints the request JSON.
build_deploy_body() {
  local model="$1"
  python3 - "${model}" "${IMAGE_NAME}" "${SSH_KEY}.pub" \
    "${GPU_COUNT}" "${CPU_UNITS}" "${MEMORY_SIZE}" "${STORAGE_SIZE}" \
    "${MAX_PRICE_AMOUNT}" "${PRICE_DENOM}" "${DEPOSIT_USD}" "${SERVICE_NAME}" <<'PY'
import base64, json, pathlib, sys, textwrap

(
    model, image, pubkey_path, gpu_count, cpu, mem, storage,
    max_price, denom, deposit, svc,
) = sys.argv[1:12]
pubkey = pathlib.Path(pubkey_path).read_text().strip()
# Base64 avoids YAML quoting issues with spaces in ssh-ed25519 keys.
pubkey_b64 = base64.b64encode(pubkey.encode()).decode("ascii")

# Keep the entrypoint as one shell string — Console SDL validation is picky
# about multiline args blocks.
boot = (
    "set -e; export DEBIAN_FRONTEND=noninteractive; "
    "apt-get update -qq; "
    "apt-get install -y -qq openssh-server curl ca-certificates git python3 >/dev/null; "
    "mkdir -p /var/run/sshd /root/.ssh; "
    "echo \"$SSH_PUBKEY_B64\" | base64 -d > /root/.ssh/authorized_keys; "
    "chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; "
    "sed -i 's/^#\\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config; "
    "sed -i 's/^#\\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config; "
    "ssh-keygen -A >/dev/null; "
    "exec /usr/sbin/sshd -D -e"
)

sdl = textwrap.dedent(f"""\
version: "2.0"
services:
  {svc}:
    image: {image}
    env:
      - SSH_PUBKEY_B64={pubkey_b64}
      - NVIDIA_DRIVER_CAPABILITIES=all
      - NVIDIA_VISIBLE_DEVICES=all
    command:
      - /bin/bash
      - -lc
      - {json.dumps(boot)}
    expose:
      - port: 22
        as: 22
        to:
          - global: true
profiles:
  compute:
    {svc}:
      resources:
        cpu:
          units: {float(cpu):.1f}
        memory:
          size: {mem}
        storage:
          size: {storage}
        gpu:
          units: {int(gpu_count)}
          attributes:
            vendor:
              nvidia:
                - model: {model}
  placement:
    dcloud:
      pricing:
        {svc}:
          denom: {denom}
          amount: {int(max_price)}
deployment:
  {svc}:
    dcloud:
      profile: {svc}
      count: 1
""")
# json.dumps(boot) already added quotes; strip them for YAML scalar embedding
# (we embedded the JSON string including quotes as the YAML list item — valid).
print(json.dumps({"data": {"sdl": sdl, "deposit": float(deposit)}}))
PY
}

create_deployment() {
  local model="$1"
  local body resp manifest_file
  body="$(build_deploy_body "${model}")"
  # Save last request for debugging 400s
  printf '%s\n' "${body}" >"${OUT_DIR}/.last-deploy-request.json"
  python3 -c 'import json,sys; open(sys.argv[1],"w").write(json.load(open(sys.argv[2]))["data"]["sdl"])' \
    "${OUT_DIR}/.last-deploy.sdl.yaml" "${OUT_DIR}/.last-deploy-request.json"

  if ! resp="$(api POST /v1/deployments -d @"${OUT_DIR}/.last-deploy-request.json")"; then
    echo "error: create deployment failed for model=${model}" >&2
    echo "  SDL: ${OUT_DIR}/.last-deploy.sdl.yaml" >&2
    echo "  request: ${OUT_DIR}/.last-deploy-request.json" >&2
    return 1
  fi
  manifest_file="${OUT_DIR}/.manifest-${model}.json"
  python3 -c '
import json, sys
d = json.load(sys.stdin)["data"]
open(sys.argv[1], "w").write(json.dumps(d.get("manifest", "")))
print(d["dseq"])
' "${manifest_file}" <<<"${resp}"
}

# Rank open bids from a JSON file. Prints: usd_per_hour provider gseq oseq denom amount
# NOTE: must not use `python3 - <<'PY'` with JSON on stdin — `-` already consumes stdin.
format_bids() {
  local akt_usd="$1"
  local json_file="$2"
  python3 - "${akt_usd}" "${BLOCK_TIME_SEC}" "${SKIP_PROVIDERS[*]:-}" "${json_file}" <<'PY'
import json, sys
akt_usd = float(sys.argv[1])
block_s = float(sys.argv[2])
skip = set(sys.argv[3].split()) if sys.argv[3].strip() else set()
blocks_per_hour = 3600.0 / block_s
try:
    data = json.load(open(sys.argv[4])).get("data") or []
except Exception as e:
    print(f"warning: could not parse bids JSON: {e}", file=sys.stderr)
    data = []
rows = []
for entry in data:
    bid = entry.get("bid") or entry
    bid_id = bid.get("id") or {}
    provider = bid_id.get("provider", "")
    if not provider or provider in skip:
        continue
    state = (bid.get("state") or "open").lower()
    if state not in ("open", "active", ""):
        continue
    price = bid.get("price") or {}
    denom = price.get("denom", "")
    amount = float(price.get("amount") or 0)
    per_hour_tokens = (amount / 1e6) * blocks_per_hour
    if denom in ("uakt", "uact") or "uakt" in denom or denom.endswith("uact"):
        usd = per_hour_tokens * akt_usd
    else:
        # uusdc / IBC-USDC micro-units → USD
        usd = per_hour_tokens
    rows.append((usd, provider, bid_id.get("gseq", 1), bid_id.get("oseq", 1), denom, amount))
rows.sort()
for r in rows:
    print(f"{r[0]:.6f} {r[1]} {r[2]} {r[3]} {r[4]} {r[5]:.0f}")
PY
}

wait_bids() {
  local dseq="$1"
  local akt_usd="$2"
  local deadline=$((SECONDS + BID_WAIT_SECONDS))
  local bids_file ranked n
  bids_file="$(mktemp)"
  # shellcheck disable=SC2064
  trap "rm -f '${bids_file}'" RETURN
  while ((SECONDS < deadline)); do
    if ! api GET "/v1/bids?dseq=${dseq}" >"${bids_file}" 2>/dev/null; then
      echo '{"data":[]}' >"${bids_file}"
    fi
    n="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("data") or []))' "${bids_file}" 2>/dev/null || echo 0)"
    echo "  bids so far: ${n}" >&2
    ranked="$(format_bids "${akt_usd}" "${bids_file}")"
    if [[ -n "${ranked}" ]]; then
      echo "${ranked}"
      return 0
    fi
    sleep 3
  done
  return 1
}

create_lease() {
  local dseq="$1" gseq="$2" oseq="$3" provider="$4" model="$5"
  local body
  body="$(python3 - "${OUT_DIR}/.manifest-${model}.json" "${dseq}" "${gseq}" "${oseq}" "${provider}" <<'PY'
import json, sys
raw = open(sys.argv[1]).read()
try:
    manifest = json.loads(raw)
except Exception:
    manifest = raw.strip().strip('"')
print(json.dumps({
    "manifest": manifest,
    "leases": [{
        "dseq": sys.argv[2],
        "gseq": int(sys.argv[3]),
        "oseq": int(sys.argv[4]),
        "provider": sys.argv[5],
    }],
}))
PY
)"
  api POST /v1/leases -d "${body}" >/dev/null
}

# Resolve SSH host:port from deployment status (or provider-proxy fallback).
# Prints "host port".
# Parse host/port for SERVICE_NAME out of a deployment or lease-status JSON file.
parse_ssh_from_json() {
  local json_file="$1"
  python3 - "${SERVICE_NAME}" "${json_file}" <<'PY'
import json, sys
svc, path = sys.argv[1], sys.argv[2]
try:
    root = json.load(open(path))
except Exception:
    sys.exit(0)

def emit_from_status(st):
    fps = (st.get("forwarded_ports") or {}).get(svc) or []
    for fp in fps:
        host = fp.get("host") or fp.get("Host")
        port = fp.get("externalPort") or fp.get("ExternalPort") or fp.get("port")
        inner = fp.get("port") or fp.get("Port") or 22
        if host and port and int(inner) == 22:
            print(host, int(port))
            return True
    if fps:
        fp = fps[0]
        host = fp.get("host") or fp.get("Host")
        port = fp.get("externalPort") or fp.get("ExternalPort")
        if host and port:
            print(host, int(port))
            return True
    for ip in (st.get("ips") or {}).get(svc) or []:
        host = ip.get("IP") or ip.get("ip")
        port = ip.get("ExternalPort") or ip.get("externalPort") or ip.get("Port") or 22
        if host:
            print(host, int(port))
            return True
    return False

d = root.get("data") or root
# Full deployment object
for lease in d.get("leases") or []:
    if emit_from_status(lease.get("status") or {}):
        sys.exit(0)
# Bare lease status
if emit_from_status(d if isinstance(d, dict) else {}):
    sys.exit(0)
if emit_from_status(root if isinstance(root, dict) else {}):
    sys.exit(0)
PY
}

resolve_ssh() {
  local dseq="$1" provider="$2" gseq="$3" oseq="$4"
  local deadline=$((SECONDS + SSH_WAIT_SECONDS))
  local info="" status_file
  status_file="$(mktemp)"
  # shellcheck disable=SC2064
  trap "rm -f '${status_file}'" RETURN
  while ((SECONDS < deadline)); do
    if api GET "/v1/deployments/${dseq}" >"${status_file}" 2>/dev/null; then
      info="$(parse_ssh_from_json "${status_file}" || true)"
      if [[ -n "${info}" ]]; then
        echo "${info}"
        return 0
      fi
    fi

    # Fallback: provider-proxy / direct provider status
    if fetch_status_via_proxy "${dseq}" "${provider}" "${gseq}" "${oseq}" >"${status_file}" 2>/dev/null; then
      info="$(parse_ssh_from_json "${status_file}" || true)"
      if [[ -n "${info}" ]]; then
        echo "${info}"
        return 0
      fi
    fi
    sleep 10
  done
  return 1
}

fetch_status_via_proxy() {
  local dseq="$1" provider="$2" gseq="$3" oseq="$4"
  local host_uri jwt body
  host_uri="$(curl -fsS "${API_BASE}/v1/providers/${provider}" \
    -H "x-api-key: ${AKASH_API_KEY}" 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("data") or d).get("hostUri") or (d.get("data") or d).get("host_uri") or "")' 2>/dev/null)" || host_uri=""
  if [[ -z "${host_uri}" ]]; then
    # network console providers endpoint variants
    host_uri="$(curl -fsS "https://api.cloudmos.io/v1/providers/${provider}" 2>/dev/null \
      | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("hostUri") or d.get("host_uri") or "")' 2>/dev/null)" || host_uri=""
  fi
  [[ -z "${host_uri}" ]] && return 1

  jwt="$(api POST /v1/create-jwt-token -d '{"data":{"ttl":1800,"leases":{"access":"scoped","scope":["status","logs","shell"]}}}' 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["token"])' 2>/dev/null)" || jwt=""
  [[ -z "${jwt}" ]] && return 1

  local url="${host_uri%/}/lease/${dseq}/${gseq}/${oseq}/status"
  # Prefer provider-proxy (handles mTLS / self-signed). Fall back to curl -k.
  body="$(python3 -c 'import json,sys; print(json.dumps({"method":"GET","url":sys.argv[1],"providerAddress":sys.argv[2],"auth":{"type":"jwt","token":sys.argv[3]}}))' \
    "${url}" "${provider}" "${jwt}")"
  if curl -fsS -X POST "${PROXY_BASE}/" \
      -H "x-api-key: ${AKASH_API_KEY}" \
      -H "Content-Type: application/json" \
      -d "${body}" 2>/dev/null; then
    return 0
  fi
  curl -fsSk -H "Authorization: Bearer ${jwt}" "${url}" 2>/dev/null || return 1
}

run_one_attempt() {
  local model="$1"
  local akt_usd="$2"
  echo "deploying SDL for model=${model} (deposit=\$${DEPOSIT_USD}) ..." >&2

  local dseq
  if ! dseq="$(create_deployment "${model}")"; then
    return 1
  fi
  DSEQ="${dseq}"
  echo "dseq=${dseq} — waiting up to ${BID_WAIT_SECONDS}s for bids ..." >&2

  local bids
  if ! bids="$(wait_bids "${dseq}" "${akt_usd}")" || [[ -z "${bids}" ]]; then
    echo "error: no bids for model=${model} (try raising MAX_PRICE_AMOUNT or check inventory)" >&2
    close_deployment
    # Distinct rc: no inventory — retrying would just burn more deposits.
    return 4
  fi

  local pph provider gseq oseq denom amount
  read -r pph provider gseq oseq denom amount <<<"$(echo "${bids}" | head -n1)"
  echo "accepting cheapest bid: \$${pph}/hr  provider=${provider}  (${denom} ${amount}/block)" >&2

  if ! create_lease "${dseq}" "${gseq}" "${oseq}" "${provider}" "${model}"; then
    echo "error: create lease failed" >&2
    close_deployment
    return 3
  fi

  # Top up a bit more escrow for the build+bench window
  api POST /v1/deposit-deployment \
    -d "{\"data\":{\"dseq\":\"${dseq}\",\"deposit\":${DEPOSIT_USD}}}" >/dev/null 2>&1 || true

  local host port ssh_info
  # `read` from a here-string returns 0 even on empty input, so check the
  # resolve_ssh output itself instead of the read status.
  if ! ssh_info="$(resolve_ssh "${dseq}" "${provider}" "${gseq}" "${oseq}")" ||
    [[ -z "${ssh_info}" ]]; then
    echo "error: never got an SSH forwarded port" >&2
    SKIP_PROVIDERS+=("${provider}")
    close_deployment
    return 3
  fi
  read -r host port <<<"${ssh_info}"
  echo "ssh root@${host} -p ${port}" >&2

  local ssh_opts=(-i "${SSH_KEY}" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -p "${port}")
  local deadline=$((SECONDS + SSH_WAIT_SECONDS)) up=0
  while ((SECONDS < deadline)); do
    if ssh "${ssh_opts[@]}" "root@${host}" true 2>/dev/null; then
      up=1
      break
    fi
    sleep 10
  done
  if [[ "${up}" != 1 ]]; then
    echo "error: SSH never came up" >&2
    SKIP_PROVIDERS+=("${provider}")
    close_deployment
    return 3
  fi

  if ! ssh "${ssh_opts[@]}" "root@${host}" \
    'nvidia-smi >/dev/null 2>&1'; then
    echo "probe: nvidia-smi failed — skipping provider" >&2
    SKIP_PROVIDERS+=("${provider}")
    close_deployment
    return "${EXIT_COMPUTE_ONLY}"
  fi

  # Vulkan screening (same idea as Clore)
  if ! ssh "${ssh_opts[@]}" "root@${host}" \
    'ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 /usr/lib64/libGLX_nvidia.so.0 2>/dev/null | head -1 | grep -q .' \
    && ! ssh "${ssh_opts[@]}" "root@${host}" \
      'ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.* 2>/dev/null | head -1 | grep -q .'; then
    echo "probe: no libGLX_nvidia — trying remote-run Vulkan setup anyway" >&2
  else
    echo "probe ok: NVIDIA GL userspace present" >&2
  fi

  local remote_rc=0
  scp -q -O -i "${SSH_KEY}" -o StrictHostKeyChecking=accept-new -P "${port}" \
    "${SCRIPT_DIR}/remote-run.sh" "${SCRIPT_DIR}/record.sh" "root@${host}:/root/" || remote_rc=1

  if [[ "${remote_rc}" -eq 0 ]]; then
    set +e
    ssh "${ssh_opts[@]}" "root@${host}" \
      "cd /root && WORK_DIR='${REMOTE_DIR}' ./remote-run.sh \
         --provider akash --cost-per-hour '${pph}' \
         --duration '${DURATION}' --batch-sizes '${BATCH_SIZES}' \
         --job-interval '${JOB_INTERVAL}' --job-jitter '${JOB_JITTER}' \
         ${DIFFICULTY:+--difficulty '${DIFFICULTY}'} \
         --gpu-devices '${GPU_COUNT}' \
         --miner-source '${MINER_SOURCE}' \
         --miner-repo '${MINER_REPO}' \
         --miner-branch '${MINER_BRANCH}' \
         --notes 'akash_dseq=${dseq};provider=${provider};model=${model};miner=${MINER_SOURCE}@${MINER_BRANCH}'"
    remote_rc=$?
    set -e
  fi

  if [[ "${remote_rc}" -eq 0 ]]; then
    local local_row="${OUT_DIR}/row-akash-${dseq}.csv"
    if ! scp -q -O -i "${SSH_KEY}" -o StrictHostKeyChecking=accept-new -P "${port}" \
      "root@${host}:${REMOTE_DIR}/results.csv" "${local_row}" </dev/null ||
      [[ ! -s "${local_row}" ]]; then
      echo "error: could not download results.csv (dseq ${dseq}); benchmark data lost" >&2
      close_deployment
      return 1
    fi
    # Drop util<50 rows at append time (matches dataset policy)
    local append_rc=0
    python3 - "${local_row}" "${RESULTS_CSV}" <<'PY' || append_rc=1
import csv, sys
src, dst = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src, newline="")))
# Ensure destination has a header
import os
if not os.path.exists(dst) or os.path.getsize(dst) == 0:
    # copy header from source
    with open(src, newline="") as f:
        header = f.readline()
    open(dst, "w").write(header)
dst_fields = next(csv.reader(open(dst)))
with open(dst, "a", newline="") as out:
    w = csv.DictWriter(out, fieldnames=dst_fields, extrasaction="ignore", lineterminator="\n")
    for r in rows:
        try:
            util = float(r.get("gpu_utilization_pct") or "")
        except ValueError:
            continue
        if util < 50:
            print(f"skip append util={util}", file=sys.stderr)
            continue
        # fill ideal if missing
        if not (r.get("ideal_hash_per_dollar") or "").strip():
            try:
                hpd = float(r["hash_per_dollar"])
                r["ideal_hash_per_dollar"] = f"{hpd / (util / 100.0):.6f}"
            except Exception:
                pass
        w.writerow({k: r.get(k, "") for k in dst_fields})
print(f"appended from {src}", file=sys.stderr)
PY
    if [[ "${append_rc}" -ne 0 ]]; then
      echo "error: failed to append ${local_row} to ${RESULTS_CSV}" >&2
      close_deployment
      return 1
    fi
    echo "Appended results to ${RESULTS_CSV}" >&2
    close_deployment
    return 0
  fi

  echo "error: remote benchmark failed (rc=${remote_rc})" >&2
  SKIP_PROVIDERS+=("${provider}")
  if [[ "${KEEP_ON_FAILURE}" == "1" ]]; then
    echo "KEEP_ON_FAILURE=1 — deployment stays up: ssh -i ${SSH_KEY} -p ${port} root@${host}" >&2
    echo "  dseq=${dseq}  (close yourself or unset KEEP and re-run close)" >&2
    DSEQ=""
    return "${remote_rc}"
  fi
  close_deployment
  return "${remote_rc}"
}

list_bids() {
  local model="$1" akt_usd="$2"
  echo "== ${model}: creating temporary deployment to discover bids ..." >&2
  local dseq
  if ! dseq="$(create_deployment "${model}")"; then
    return 1
  fi
  DSEQ="${dseq}"
  echo "dseq=${dseq}" >&2
  local bids
  if ! bids="$(wait_bids "${dseq}" "${akt_usd}")" || [[ -z "${bids}" ]]; then
    echo "(no bids within ${BID_WAIT_SECONDS}s)" >&2
    close_deployment
    return 1
  fi
  echo "usd_per_hour  provider  gseq  oseq  denom  amount_per_block"
  echo "${bids}"
  close_deployment
}

run_one() {
  local model="$1" akt_usd="$2"
  local tried=0 rc=1
  SKIP_PROVIDERS=()
  while ((tried < HOST_RETRIES)); do
    ((tried += 1)) || true
    echo "== ${model}: attempt ${tried}/${HOST_RETRIES}" >&2
    set +e
    run_one_attempt "${model}" "${akt_usd}"
    rc=$?
    set -e
    if [[ "${rc}" -eq 0 ]]; then
      return 0
    fi
    if [[ "${rc}" -eq 4 ]]; then
      echo "no bids for ${model}; not retrying" >&2
      return "${rc}"
    fi
    echo "attempt ${tried} failed (rc=${rc}); retrying with skip=${SKIP_PROVIDERS[*]:-}" >&2
  done
  return "${rc}"
}

main() {
  require_cmd curl
  require_cmd ssh
  require_cmd scp
  require_cmd python3

  if [[ -z "${AKASH_API_KEY:-}" && -f "${HOME}/.config/akash/api_key" ]]; then
    AKASH_API_KEY="$(cat "${HOME}/.config/akash/api_key")"
  fi
  if [[ -z "${AKASH_API_KEY:-}" ]]; then
    echo "error: set AKASH_API_KEY (or ~/.config/akash/api_key)" >&2
    exit 1
  fi
  if [[ ! -f "${SSH_KEY}.pub" ]]; then
    echo "error: SSH public key not found: ${SSH_KEY}.pub" >&2
    exit 1
  fi
  mkdir -p "${OUT_DIR}"

  local list_only=0
  local targets=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --list) list_only=1; shift ;;
      --model) targets+=("$(normalize_model "${2:-}")"); shift 2 ;;
      --gpus-file)
        if [[ ! -f "${2:-}" ]]; then
          echo "error: --gpus-file not found: ${2:-}" >&2
          exit 1
        fi
        while IFS= read -r line || [[ -n "${line}" ]]; do
          [[ -n "${line}" ]] && targets+=("${line}")
        done < <(load_gpus_file "$2")
        shift 2
        ;;
      -h | --help) usage; exit 0 ;;
      *) targets+=("$(normalize_model "$1")"); shift ;;
    esac
  done

  # Dedupe while preserving order
  if [[ "${#targets[@]}" -gt 0 ]]; then
    local deduped=()
    deduped=()
    while IFS= read -r line; do
      [[ -n "${line}" ]] && deduped+=("${line}")
    done < <(printf '%s\n' "${targets[@]}" | awk 'NF && !seen[$0]++')
    targets=("${deduped[@]}")
  fi

  if [[ "${#targets[@]}" -eq 0 ]]; then
    usage >&2
    exit 1
  fi

  local akt_usd
  akt_usd="$(resolve_akt_usd)"
  echo "AKT_USD=${akt_usd}  deposit=\$${DEPOSIT_USD}  image=${IMAGE_NAME}" >&2
  echo "models (${#targets[@]}): ${targets[*]}" >&2

  local t failed=0
  for t in "${targets[@]}"; do
    if [[ "${list_only}" == 1 ]]; then
      list_bids "${t}" "${akt_usd}" || failed=1
    else
      run_one "${t}" "${akt_usd}" || failed=1
    fi
  done
  exit "${failed}"
}

main "$@"

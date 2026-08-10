#!/usr/bin/env bash
# Orchestrate RunPod Pods via REST API: for each GPU type, create → SSH →
# remote-run.sh (miner build + benchmark batch-size sweep → CSV) → scp → delete.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_BASE="${RUNPOD_API_BASE:-https://rest.runpod.io/v1}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/sweep-out}"
# Shared collaborative dataset (tracked in git). Per-pod temps stay in OUT_DIR.
RESULTS_CSV="${RESULTS_CSV:-${SCRIPT_DIR}/results.csv}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519}"
SSH_USER="${SSH_USER:-root}"
CLOUD_TYPE="${CLOUD_TYPE:-COMMUNITY}"
# Prefer runpod/base (CUDA + SSH, no PyTorch). Drivers come from the host.
# Use a current Hub tag — stale tags leave Pods RUNNING with runtime=null forever.
# Ubuntu 24.04 base is fine for miner builds; keep NVIDIA_DRIVER_CAPABILITIES=all.
IMAGE_NAME="${IMAGE_NAME:-runpod/base:1.1.0-cuda1281-ubuntu2404}"
TEMPLATE_ID="${TEMPLATE_ID:-}"
# Community hosts often reject large disks ("machine does not have the resources").
# Benchmark path has no node/chain data — only cargo target/ during miner build (~few GB).
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-20}"
VOLUME_GB="${VOLUME_GB:-0}"
DURATION="${DURATION:-30}"
GPU_DEVICES="${GPU_DEVICES:-1}"
BATCH_SIZES="${BATCH_SIZES:-262144 524288 1000000 4194304}"
JOB_INTERVAL="${JOB_INTERVAL:-2}"
JOB_JITTER="${JOB_JITTER:-0.2}"
DIFFICULTY="${DIFFICULTY:-}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/quantus-gpu-bench}"
MINER_SOURCE="${MINER_SOURCE:-git}"
MINER_REPO="${MINER_REPO:-https://github.com/Quantus-Network/quantus-miner.git}"
MINER_BRANCH="${MINER_BRANCH:-illuzen/gpu-bench}"
KEEP_ON_FAILURE="${KEEP_ON_FAILURE:-0}"
CREATE_RETRIES="${CREATE_RETRIES:-5}"
CREATE_RETRY_SLEEP="${CREATE_RETRY_SLEEP:-15}"
# Retry a new Pod when remote-run exits 42 (compute-only / no Vulkan host).
HOST_RETRIES="${HOST_RETRIES:-3}"
EXIT_COMPUTE_ONLY=42
# If Community create keeps failing capacity, retry on Secure Cloud.
FALLBACK_SECURE="${FALLBACK_SECURE:-1}"
# Proxy SSH (podId@ssh.runpod.io) is off by default — needs a RunPod-account
# SSH key and is easy to false-trigger while the image is still extracting.
# Direct TCP (after GraphQL shows port 22) is the reliable path.
ALLOW_PROXY_SSH="${ALLOW_PROXY_SSH:-0}"
# Image pull/extract on Community can take several minutes.
SSH_WAIT_SECONDS="${SSH_WAIT_SECONDS:-900}"

usage() {
  cat <<'EOF'
Usage: ./runpod-sweep.sh [gpuTypeId ...]

Environment:
  RUNPOD_API_KEY     Required (https://www.runpod.io/console/user/settings)
  SSH_KEY            Default ~/.ssh/id_ed25519 (public key must be in RunPod)
  CLOUD_TYPE         COMMUNITY (default) or SECURE
  IMAGE_NAME         Docker image (default: runpod/base … ubuntu2404 for GLIBC)
  TEMPLATE_ID        Optional RunPod template id (skips IMAGE_NAME)
  DURATION           Benchmark seconds per batch size (default 30)
  BATCH_SIZES        Space-separated gpu-batch-size list (default 256K 512K 1M 4M)
  JOB_INTERVAL       Simulated NewJob seconds (default 2; 0 = sustained)
  JOB_JITTER         ±fraction of JOB_INTERVAL (default 0.2; 0 = metronomic)
  DIFFICULTY         Optional decimal or max for job simulation
  RESULTS_CSV        Collaborative dataset to append (default ./results.csv)
  OUT_DIR            Per-pod temp rows (default ./sweep-out, gitignored)
  KEEP_ON_FAILURE=1  Do not delete Pod if remote-run fails
  HOST_RETRIES       New Pods to try if host lacks Vulkan libs (default 3)
  MINER_SOURCE       git (default, clone+build) or release
  MINER_BRANCH       git ref to build (default: illuzen/gpu-bench)
  MINER_REPO         git URL (default: Quantus-Network/quantus-miner)
  CONTAINER_DISK_GB  default 20 (cargo build; no node disk)

GPU type ids are RunPod strings, e.g.:
  "NVIDIA GeForce RTX 4090"
  "NVIDIA GeForce RTX 3090"
  "NVIDIA RTX A5000"

Or pass a file via:
  ./runpod-sweep.sh --gpus-file gpus.txt

See README for creating a RunPod template.
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command not found: $1" >&2
    exit 1
  fi
}

# Last failed API response body (for callers to classify errors).
API_LAST_ERROR_BODY=""
API_LAST_HTTP_CODE=""

# Prints response body to stdout. Exits non-zero on HTTP >= 400.
api() {
  local method="$1"
  local path="$2"
  local data="${3:-}"
  local tmp code
  API_LAST_ERROR_BODY=""
  API_LAST_HTTP_CODE=""
  tmp="$(mktemp)"
  local args=(
    -sS
    -X "${method}"
    -H "Authorization: Bearer ${RUNPOD_API_KEY}"
    -H "Content-Type: application/json"
    -o "${tmp}"
    -w "%{http_code}"
  )
  if [[ -n "${data}" ]]; then
    args+=(-d "${data}")
  fi
  code="$(curl "${args[@]}" "${API_BASE}${path}" || true)"
  API_LAST_HTTP_CODE="${code}"
  if [[ -z "${code}" || "${code}" == "000" ]]; then
    API_LAST_ERROR_BODY="$(cat "${tmp}" 2>/dev/null || true)"
    echo "error: RunPod API request failed (network): ${method} ${path}" >&2
    echo "${API_LAST_ERROR_BODY}" >&2 || true
    rm -f "${tmp}"
    return 1
  fi
  if [[ "${code}" -ge 400 ]]; then
    API_LAST_ERROR_BODY="$(cat "${tmp}" 2>/dev/null || true)"
    echo "error: RunPod API ${method} ${path} -> HTTP ${code}" >&2
    if [[ "${VERBOSE:-0}" == "1" && -n "${data}" ]]; then
      echo "request body:" >&2
      echo "${data}" >&2
    fi
    echo "response: ${API_LAST_ERROR_BODY}" >&2
    echo >&2
    rm -f "${tmp}"
    return 1
  fi
  cat "${tmp}"
  rm -f "${tmp}"
}

# True if last API error means "no stock for this GPU" — skip the model.
is_no_capacity_error() {
  echo "${API_LAST_ERROR_BODY}" | grep -qiE \
    'no instances currently available|no .*available|out of capacity|insufficient.*capacity'
}

# True if gpuTypeIds value is not in the current REST API enum.
is_invalid_gpu_type_error() {
  echo "${API_LAST_ERROR_BODY}" | grep -qiE \
    'gpuTypeIds/items/enum|value must be one of'
}

# Extract a field from a Pod JSON object. Fails clearly if body is an error list/object.
json_pod_field() {
  local body="$1"
  local field="$2"
  BODY="${body}" FIELD="${field}" python3 - <<'PY'
import json, os, sys
raw = os.environ["BODY"]
field = os.environ["FIELD"]
try:
    data = json.loads(raw)
except json.JSONDecodeError as e:
    print(f"error: invalid JSON from RunPod: {e}", file=sys.stderr)
    print(raw[:2000], file=sys.stderr)
    sys.exit(1)
if isinstance(data, list):
    print("error: RunPod returned a list (usually an API error), not a Pod object:", file=sys.stderr)
    print(json.dumps(data, indent=2)[:4000], file=sys.stderr)
    sys.exit(1)
if not isinstance(data, dict):
    print(f"error: unexpected RunPod JSON type: {type(data).__name__}", file=sys.stderr)
    print(raw[:2000], file=sys.stderr)
    sys.exit(1)
if field == "ssh_port":
    m = data.get("portMappings") or {}
    val = m.get("22") or m.get(22) or ""
    if not val:
        for p in (data.get("runtime") or {}).get("ports") or []:
            if str(p.get("privatePort")) == "22" and p.get("publicPort"):
                val = p.get("publicPort")
                break
elif field == "publicIp":
    val = data.get("publicIp") or ""
    if not val:
        for p in (data.get("runtime") or {}).get("ports") or []:
            if str(p.get("privatePort")) == "22" and p.get("ip"):
                val = p.get("ip")
                break
elif field == "cost":
    val = data.get("costPerHr") or data.get("adjustedCostPerHr") or ""
else:
    val = data.get(field) or ""
print(val if val is not None else "")
PY
}

# SSH helpers. MODE=direct|proxy. For proxy, HOST is pod id and PORT is unused.
ssh_cmd() {
  local mode="$1"
  local host="$2"
  local port="$3"
  shift 3
  if [[ -z "${host}" ]]; then
    echo "error: ssh_cmd called with empty host (mode=${mode})" >&2
    return 2
  fi
  local ssh_opts=(
    -i "${SSH_KEY}"
    -o StrictHostKeyChecking=no
    -o UserKnownHostsFile=/dev/null
    -o GlobalKnownHostsFile=/dev/null
    -o ConnectTimeout=20
    -o BatchMode=yes
    -o PreferredAuthentications=publickey
    -o LogLevel=ERROR
  )
  if [[ "${mode}" == "direct" ]]; then
    if [[ -z "${port}" ]]; then
      echo "error: direct ssh_cmd missing port" >&2
      return 2
    fi
    ssh "${ssh_opts[@]}" -p "${port}" "${SSH_USER}@${host}" "$@"
  else
    # Basic/proxy SSH — no scp, but command + stdin/stdout pipes work.
    ssh "${ssh_opts[@]}" "${host}@ssh.runpod.io" "$@"
  fi
}

# Upload local files into REMOTE_DIR (works for direct and proxy SSH).
ssh_upload_files() {
  local mode="$1"
  local host="$2"
  local port="$3"
  shift 3
  ssh_cmd "${mode}" "${host}" "${port}" "mkdir -p '${REMOTE_DIR}'"
  local f
  for f in "$@"; do
    local base
    base="$(basename "${f}")"
    ssh_cmd "${mode}" "${host}" "${port}" "cat > '${REMOTE_DIR}/${base}'" <"${f}"
  done
}

ssh_download() {
  local mode="$1"
  local host="$2"
  local port="$3"
  local remote_path="$4"
  local local_path="$5"
  ssh_cmd "${mode}" "${host}" "${port}" "cat '${remote_path}'" >"${local_path}"
}

# When sourced by runpod-shell.sh, only helpers below are used.
build_pod_payload() {
  local gpu_type="$1"
  local name="$2"
  local disk_gb="$3"
  local volume_gb="$4"
  local cloud_type="$5"
  GPU_TYPE="${gpu_type}" \
  CLOUD_TYPE="${cloud_type}" \
  IMAGE_NAME="${IMAGE_NAME}" \
  TEMPLATE_ID="${TEMPLATE_ID}" \
  POD_NAME="${name}" \
  CONTAINER_DISK_GB="${disk_gb}" \
  VOLUME_GB="${volume_gb}" \
  python3 - <<'PY'
import json, os
volume = int(os.environ["VOLUME_GB"])
# graphics+utility so NVIDIA Container Toolkit mounts Vulkan/GL libs
# (compute-only pods only get CUDA → WGPU sees llvmpipe and fails).
body = {
  "name": os.environ["POD_NAME"],
  "cloudType": os.environ["CLOUD_TYPE"],
  "computeType": "GPU",
  "gpuTypeIds": [os.environ["GPU_TYPE"]],
  "gpuCount": 1,
  "gpuTypePriority": "availability",
  "supportPublicIp": True,
  "containerDiskInGb": int(os.environ["CONTAINER_DISK_GB"]),
  "volumeMountPath": "/workspace",
  "ports": ["22/tcp"],
  "volumeInGb": volume,
  "env": {
    "NVIDIA_DRIVER_CAPABILITIES": "all",
    "NVIDIA_VISIBLE_DEVICES": "all",
  },
}
template = os.environ.get("TEMPLATE_ID") or ""
if template:
  body["templateId"] = template
else:
  body["imageName"] = os.environ["IMAGE_NAME"]
print(json.dumps(body))
PY
}

create_pod_with_cloud() {
  local gpu_type="$1"
  local cloud_type="$2"
  local name="qbench-$(echo "${gpu_type}" | tr -c 'A-Za-z0-9' '-' | cut -c1-40)-$(date +%s)"
  local disk="${CONTAINER_DISK_GB}"
  local volume="${VOLUME_GB}"
  local attempt payload body_file api_rc

  body_file="$(mktemp)"
  for ((attempt = 1; attempt <= CREATE_RETRIES; attempt++)); do
    payload="$(build_pod_payload "${gpu_type}" "${name}-${attempt}" "${disk}" "${volume}" "${cloud_type}")"
    echo "Creating Pod for ${gpu_type} on ${cloud_type} (attempt ${attempt}/${CREATE_RETRIES}, disk=${disk}G volume=${volume}G) ..." >&2
    if [[ "${VERBOSE:-0}" == "1" ]]; then
      echo "payload: ${payload}" >&2
    fi
    # Call api in this shell (not $(api …)) so API_LAST_ERROR_BODY is preserved.
    set +e
    api POST /pods "${payload}" >"${body_file}"
    api_rc=$?
    set -e
    if [[ "${api_rc}" -eq 0 ]]; then
      cat "${body_file}"
      rm -f "${body_file}"
      return 0
    fi
    # No stock — shrinking disk will not help; stop this cloud tier.
    if is_no_capacity_error; then
      echo "no capacity for ${gpu_type} on ${cloud_type}" >&2
      rm -f "${body_file}"
      return 2
    fi
    # Stale gpuTypeId vs current RunPod schema — skip (update gpus.all.txt).
    if is_invalid_gpu_type_error; then
      echo "skip: invalid gpuTypeId for API schema: ${gpu_type}" >&2
      echo "tip: refresh gpu-bench/gpus.all.txt from the RunPod /pods enum" >&2
      rm -f "${body_file}"
      return 2
    fi
    if [[ "${volume}" -gt 0 ]]; then
      volume=0
    elif [[ "${disk}" -gt 15 ]]; then
      disk=15
    elif [[ "${disk}" -gt 10 ]]; then
      disk=10
    fi
    if [[ "${attempt}" -lt "${CREATE_RETRIES}" ]]; then
      echo "retrying in ${CREATE_RETRY_SLEEP}s (next disk=${disk}G volume=${volume}G) ..." >&2
      sleep "${CREATE_RETRY_SLEEP}"
    fi
  done
  rm -f "${body_file}"
  return 1
}

# 0 = created (JSON on stdout), 2 = skip (no capacity on any tried tier), 1 = other failure
create_pod() {
  local gpu_type="$1"
  local rc=0

  set +e
  create_pod_with_cloud "${gpu_type}" "${CLOUD_TYPE}"
  rc=$?
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    return 0
  fi

  # Community out of stock → try Secure (same as resource exhaustion).
  if [[ "${FALLBACK_SECURE}" == "1" && "${CLOUD_TYPE}" == "COMMUNITY" ]]; then
    echo "Community unavailable for ${gpu_type}; trying SECURE cloud ..." >&2
    set +e
    create_pod_with_cloud "${gpu_type}" "SECURE"
    rc=$?
    set -e
    if [[ "${rc}" -eq 0 ]]; then
      return 0
    fi
  fi

  if [[ "${rc}" -eq 2 ]]; then
    return 2
  fi
  echo "error: exhausted create retries for ${gpu_type}" >&2
  return 1
}

graphql_pod_ssh() {
  # Prints one line to stdout: "IP PORT" when TCP/22 is mapped, else nothing.
  # Quiet unless VERBOSE=1.
  local pod_id="$1"
  local query payload
  query="$(printf 'query { pod(input: {podId: "%s"}) { id desiredStatus costPerHr runtime { ports { ip isIpPublic privatePort publicPort type } } } }' "${pod_id}")"
  payload="$(POD_QUERY="${query}" python3 - <<'PY'
import json, os
print(json.dumps({"query": os.environ["POD_QUERY"]}))
PY
)"
  local resp
  resp="$(curl -sS -X POST \
    -H "content-type: application/json" \
    --url "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
    -d "${payload}" || true)"
  BODY="${resp}" VERBOSE="${VERBOSE:-0}" python3 - <<'PY'
import json, os, sys
raw = os.environ["BODY"]
verbose = os.environ.get("VERBOSE") == "1"
try:
    data = json.loads(raw)
except Exception as e:
    if verbose:
        print(f"graphql parse error: {e}", file=sys.stderr)
    sys.exit(0)
if data.get("errors"):
    if verbose:
        print("graphql errors:", json.dumps(data["errors"])[:500], file=sys.stderr)
    sys.exit(0)
pod = ((data.get("data") or {}).get("pod")) or {}
ports = ((pod.get("runtime") or {}).get("ports")) or []
best = None
for p in ports:
    if int(p.get("privatePort") or 0) != 22:
        continue
    if not p.get("ip") or not p.get("publicPort"):
        continue
    if p.get("isIpPublic"):
        best = p
        break
    if best is None:
        best = p
if best:
    print(f"{best['ip']} {best['publicPort']}")
elif verbose:
    print(f"  graphql: status={pod.get('desiredStatus')} ports={len(ports)}", file=sys.stderr)
PY
}

# Prints one stdout line: ready|<mode>|<host>|<port>|<cost>
# Do not print anything else to stdout (stderr is OK).
wait_ssh() {
  local pod_id="$1"
  if [[ -z "${pod_id}" ]]; then
    echo "error: wait_ssh: empty pod id" >&2
    return 1
  fi
  local deadline=$((SECONDS + SSH_WAIT_SECONDS))
  local last_heartbeat=0
  local spun_at=$SECONDS

  echo "  waiting for image extract + SSH (up to ${SSH_WAIT_SECONDS}s) ..." >&2

  while (( SECONDS < deadline )); do
    local info ip port status cost g_ip g_port elapsed
    elapsed=$((SECONDS - spun_at))
    if ! info="$(api GET "/pods/${pod_id}?includeMachine=true")"; then
      sleep 10
      continue
    fi
    ip="$(json_pod_field "${info}" publicIp)" || return 1
    port="$(json_pod_field "${info}" ssh_port)" || return 1
    status="$(json_pod_field "${info}" desiredStatus)" || return 1
    cost="$(json_pod_field "${info}" cost)" || return 1
    [[ -n "${cost}" ]] || cost="0"

    g_ip=""
    g_port=""
    if read -r g_ip g_port <<<"$(graphql_pod_ssh "${pod_id}")"; then
      if [[ -n "${g_ip}" && -n "${g_port}" ]]; then
        ip="${g_ip}"
        port="${g_port}"
      fi
    fi

    # Only attempt SSH after port mappings exist (container finished starting).
    if [[ "${status}" == "RUNNING" && -n "${ip}" && -n "${port}" ]]; then
      echo "  ports up (${elapsed}s) — probing ${SSH_USER}@${ip}:${port}" >&2
      if ssh_cmd direct "${ip}" "${port}" nvidia-smi >/dev/null 2>&1; then
        printf 'ready|direct|%s|%s|%s\n' "${ip}" "${port}" "${cost}"
        return 0
      fi
      echo "  SSH daemon not ready yet, retrying ..." >&2
    elif (( elapsed - last_heartbeat >= 30 )); then
      last_heartbeat=$elapsed
      if [[ -n "${ip}" && -n "${port}" ]]; then
        echo "  still waiting ${elapsed}s (tcp=${ip}:${port}, ssh not ready)" >&2
      else
        echo "  still waiting ${elapsed}s (image extract / ports pending)" >&2
      fi
    fi

    # Optional proxy — only after ports exist, and only if explicitly enabled.
    if [[ "${ALLOW_PROXY_SSH}" == "1" && "${status}" == "RUNNING" && -n "${ip}" && -n "${port}" ]]; then
      if ssh_cmd proxy "${pod_id}" "22" nvidia-smi >/dev/null 2>&1; then
        printf 'ready|proxy|%s|22|%s\n' "${pod_id}" "${cost}"
        return 0
      fi
    fi

    sleep 10
  done
  echo "error: timed out after ${SSH_WAIT_SECONDS}s waiting for Pod ${pod_id}" >&2
  echo "tip: console logs still extracting? increase SSH_WAIT_SECONDS=1200" >&2
  return 1
}

delete_pod() {
  local pod_id="$1"
  echo "Deleting Pod ${pod_id} ..." >&2
  api DELETE "/pods/${pod_id}" >/dev/null || true
}

run_one_attempt() {
  local gpu_type="$1"
  local create_json pod_id create_rc=0
  set +e
  create_json="$(create_pod "${gpu_type}")"
  create_rc=$?
  set -e
  if [[ "${create_rc}" -eq 2 ]]; then
    echo "skip: no instances available for ${gpu_type}" >&2
    return 2
  fi
  if [[ "${create_rc}" -ne 0 ]]; then
    echo "error: failed to create Pod for ${gpu_type}" >&2
    return 1
  fi
  if ! pod_id="$(json_pod_field "${create_json}" id)"; then
    echo "error: create Pod response missing id:" >&2
    echo "${create_json}" >&2
    return 1
  fi
  if [[ -z "${pod_id}" ]]; then
    echo "error: empty Pod id. Full response:" >&2
    echo "${create_json}" >&2
    return 1
  fi
  echo "Pod id: ${pod_id}" >&2

  local ready_line tag mode host port cost
  if ! ready_line="$(wait_ssh "${pod_id}")"; then
    delete_pod "${pod_id}"
    return 1
  fi
  IFS='|' read -r tag mode host port cost <<<"${ready_line}"
  if [[ "${tag}" != "ready" || -z "${mode}" || -z "${host}" || -z "${port}" ]]; then
    echo "error: bad wait_ssh result: ${ready_line}" >&2
    delete_pod "${pod_id}"
    return 1
  fi
  if [[ -z "${cost}" ]]; then
    cost="0"
  fi
  if [[ "${mode}" == "direct" ]]; then
    echo "SSH ${SSH_USER}@${host} -p ${port} (costPerHr=${cost})" >&2
  else
    echo "SSH ${host}@ssh.runpod.io (costPerHr=${cost})" >&2
  fi

  ssh_upload_files "${mode}" "${host}" "${port}" \
    "${SCRIPT_DIR}/remote-run.sh" \
    "${SCRIPT_DIR}/record.sh"

  local remote_rc=0
  set +e
  ssh_cmd "${mode}" "${host}" "${port}" \
    "chmod +x '${REMOTE_DIR}/remote-run.sh' '${REMOTE_DIR}/record.sh' && \
     WORK_DIR='${REMOTE_DIR}' \
     MINER_SOURCE='${MINER_SOURCE}' \
     MINER_REPO='${MINER_REPO}' \
     MINER_BRANCH='${MINER_BRANCH}' \
     FORCE_MINER_BUILD='${FORCE_MINER_BUILD:-0}' \
     '${REMOTE_DIR}/remote-run.sh' \
       --provider runpod \
       --cost-per-hour '${cost}' \
       --duration '${DURATION}' \
       --gpu-devices '${GPU_DEVICES}' \
       --batch-sizes '${BATCH_SIZES}' \
       --job-interval '${JOB_INTERVAL}' \
       --job-jitter '${JOB_JITTER}' \
       ${DIFFICULTY:+--difficulty '${DIFFICULTY}'} \
       --miner-source '${MINER_SOURCE}' \
       --miner-repo '${MINER_REPO}' \
       --miner-branch '${MINER_BRANCH}' \
       --notes 'gpuTypeId=${gpu_type};pod=${pod_id};ssh=${mode};miner=${MINER_SOURCE}@${MINER_BRANCH}'"
  remote_rc=$?
  set -e

  if [[ "${remote_rc}" -eq 0 ]]; then
    local local_row="${OUT_DIR}/row-${pod_id}.csv"
    ssh_download "${mode}" "${host}" "${port}" \
      "${REMOTE_DIR}/results.csv" "${local_row}"
    tail -n +2 "${local_row}" >>"${RESULTS_CSV}"
    echo "Appended results to ${RESULTS_CSV}" >&2
    delete_pod "${pod_id}"
    return 0
  fi

  echo "error: remote-run failed on ${gpu_type} (pod ${pod_id}, rc=${remote_rc})" >&2
  if [[ "${KEEP_ON_FAILURE}" == "1" ]]; then
    echo "KEEP_ON_FAILURE=1 — leaving Pod ${pod_id} up" >&2
    if [[ "${mode}" == "direct" ]]; then
      echo "  ssh -i ${SSH_KEY} -p ${port} ${SSH_USER}@${host}" >&2
    else
      echo "  ssh -i ${SSH_KEY} ${host}@ssh.runpod.io" >&2
    fi
    return "${remote_rc}"
  fi
  delete_pod "${pod_id}"
  # Propagate compute-only so caller can try another host.
  return "${remote_rc}"
}

run_one() {
  local gpu_type="$1"
  local attempt rc=1
  for ((attempt = 1; attempt <= HOST_RETRIES; attempt++)); do
    if [[ "${attempt}" -gt 1 ]]; then
      echo "host retry ${attempt}/${HOST_RETRIES} for ${gpu_type} ..." >&2
    fi
    set +e
    run_one_attempt "${gpu_type}"
    rc=$?
    set -e
    if [[ "${rc}" -eq 0 || "${rc}" -eq 2 ]]; then
      return "${rc}"
    fi
    if [[ "${rc}" -eq "${EXIT_COMPUTE_ONLY}" && "${attempt}" -lt "${HOST_RETRIES}" ]]; then
      echo "compute-only / bad Vulkan host — trying another Pod ..." >&2
      continue
    fi
    return "${rc}"
  done
  return "${rc}"
}

runpod_sweep_main() {
  local GPUS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpus-file)
        while IFS= read -r line || [[ -n "${line}" ]]; do
          [[ -z "${line}" || "${line}" =~ ^# ]] && continue
          GPUS+=("${line}")
        done <"${2:?}"
        shift 2
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        GPUS+=("$1")
        shift
        ;;
    esac
  done

  if [[ "${#GPUS[@]}" -eq 0 ]]; then
    echo "error: pass at least one GPU type id" >&2
    usage >&2
    exit 1
  fi

  require_cmd curl
  require_cmd ssh
  require_cmd python3

  if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
    echo "error: set RUNPOD_API_KEY" >&2
    exit 1
  fi
  if [[ ! -f "${SSH_KEY}" ]]; then
    echo "error: SSH private key not found: ${SSH_KEY}" >&2
    exit 1
  fi

  mkdir -p "${OUT_DIR}"
  if [[ ! -f "${RESULTS_CSV}" ]]; then
    printf '%s\n' \
      "timestamp,cloud_provider,gpu_model,vram_mb,sm_count,driver_version,hashrate,gpu_utilization_pct,cost_per_hour,cost_per_sec,hash_per_dollar,sample_seconds,notes" \
      >"${RESULTS_CSV}"
  fi

  local failed=0
  local gpu
  for gpu in "${GPUS[@]}"; do
    echo "======== ${gpu} ========" >&2
    if ! run_one "${gpu}"; then
      failed=1
    fi
  done

  echo "Dataset CSV: ${RESULTS_CSV}" >&2
  exit "${failed}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  runpod_sweep_main "$@"
fi

#!/usr/bin/env bash
# Spin up one RunPod GPU, upload bench scripts, print SSH — leave it running
# so you can iterate on Vulkan / miner commands without recreate-delete cycles.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runpod-sweep.sh
source "${SCRIPT_DIR}/runpod-sweep.sh"

STATE_FILE="${STATE_FILE:-${SCRIPT_DIR}/sweep-out/last-shell-pod.env}"

usage() {
  cat <<'EOF'
Usage:
  ./runpod-shell.sh "NVIDIA L4"          Create pod, upload scripts, print SSH
  ./runpod-shell.sh --delete [POD_ID]    Terminate pod (default: last created)
  ./runpod-shell.sh --ssh                ssh into last pod
  ./runpod-shell.sh --status             Show last pod id / SSH line

Same env as runpod-sweep.sh (RUNPOD_API_KEY, IMAGE_NAME, CLOUD_TYPE, SSH_KEY, …).

On the pod, typical debug loop:
  cd /workspace/quantus-gpu-bench
  bash -x ./remote-run.sh --cost-per-hour 0.39 --duration 30 --warmup 20

  # or step through (see printed hints)
EOF
}

write_state() {
  local pod_id="$1" mode="$2" host="$3" port="$4" cost="$5" gpu_type="$6"
  mkdir -p "$(dirname "${STATE_FILE}")"
  # Quote values so GPU names with spaces survive `source`.
  {
    printf "POD_ID=%q\n" "${pod_id}"
    printf "SSH_MODE=%q\n" "${mode}"
    printf "SSH_HOST=%q\n" "${host}"
    printf "SSH_PORT=%q\n" "${port}"
    printf "COST_PER_HOUR=%q\n" "${cost}"
    printf "GPU_TYPE=%q\n" "${gpu_type}"
    printf "SSH_KEY=%q\n" "${SSH_KEY}"
    printf "SSH_USER=%q\n" "${SSH_USER}"
    printf "REMOTE_DIR=%q\n" "${REMOTE_DIR}"
  } >"${STATE_FILE}"
}

load_state() {
  if [[ ! -f "${STATE_FILE}" ]]; then
    echo "error: no saved pod state at ${STATE_FILE}" >&2
    echo "Create one with: ./runpod-shell.sh \"NVIDIA L4\"" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${STATE_FILE}"
}

print_ssh_hint() {
  local mode="$1" host="$2" port="$3" cost="$4" pod_id="$5"
  echo
  echo "======== interactive pod ready ========"
  echo "Pod id:     ${pod_id}"
  echo "costPerHr:  ${cost}"
  echo "state file: ${STATE_FILE}"
  echo
  if [[ "${mode}" == "direct" ]]; then
    echo "ssh -i ${SSH_KEY} -p ${port} ${SSH_USER}@${host}"
  else
    echo "ssh -i ${SSH_KEY} ${host}@ssh.runpod.io"
  fi
  echo
  echo "Re-attach:  ./runpod-shell.sh --ssh"
  echo "Tear down:  ./runpod-shell.sh --delete"
  echo
  echo "On the pod:"
  cat <<EOF
  cd ${REMOTE_DIR}
  # builds miner from git (MINER_BRANCH=illuzen/gpu-bench) + release node:
  bash -x ./remote-run.sh --cost-per-hour ${cost} --duration 30 --warmup 20
  # FORCE_MINER_BUILD=1 ./remote-run.sh ...   # after pushing new commits

  # A/B gpu-batch-size (needs pushed CLI fix: range >= batch size):
  # FORCE_MINER_BUILD=1 ./remote-run.sh --cost-per-hour ${cost} --duration 20 --warmup 15
  # ./batch-tune.sh --duration 30

  # Vulkan sanity:
  nvidia-smi
  ls -l /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so* 2>/dev/null || true
  vulkaninfo --summary 2>&1 | head -n 60

  # After editing scripts locally, re-upload from your laptop:
  #   ./runpod-shell.sh --upload
EOF
  echo "======================================="
}

cmd_create() {
  local gpu_type="$1"

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

  echo "======== shell: ${gpu_type} ========" >&2
  local create_json pod_id create_rc=0
  set +e
  create_json="$(create_pod "${gpu_type}")"
  create_rc=$?
  set -e
  if [[ "${create_rc}" -eq 2 ]]; then
    echo "error: no instances available for ${gpu_type} (Community + Secure)." >&2
    echo "tip: try another GPU, or CLOUD_TYPE=SECURE ./runpod-shell.sh \"${gpu_type}\"" >&2
    exit 2
  fi
  if [[ "${create_rc}" -ne 0 ]]; then
    echo "error: failed to create Pod for ${gpu_type}" >&2
    exit 1
  fi
  # create_pod already printed JSON to stdout via create_pod_with_cloud; capture it.
  # (create_json from $(create_pod) only has stdout — errors went to stderr.)
  pod_id="$(json_pod_field "${create_json}" id)"
  if [[ -z "${pod_id}" ]]; then
    echo "error: create Pod response missing id:" >&2
    echo "${create_json}" >&2
    exit 1
  fi
  echo "Pod id: ${pod_id}" >&2

  local ready_line tag mode host port cost
  if ! ready_line="$(wait_ssh "${pod_id}")"; then
    echo "error: SSH never came up — deleting ${pod_id}" >&2
    delete_pod "${pod_id}"
    exit 1
  fi
  IFS='|' read -r tag mode host port cost <<<"${ready_line}"
  if [[ "${tag}" != "ready" || -z "${host}" || -z "${port}" ]]; then
    echo "error: bad wait_ssh result: ${ready_line}" >&2
    delete_pod "${pod_id}"
    exit 1
  fi
  [[ -n "${cost}" ]] || cost="0"

  ssh_upload_files "${mode}" "${host}" "${port}" \
    "${SCRIPT_DIR}/remote-run.sh" \
    "${SCRIPT_DIR}/record.sh"
  ssh_cmd "${mode}" "${host}" "${port}" \
    "chmod +x '${REMOTE_DIR}/remote-run.sh' '${REMOTE_DIR}/record.sh'"

  write_state "${pod_id}" "${mode}" "${host}" "${port}" "${cost}" "${gpu_type}"
  print_ssh_hint "${mode}" "${host}" "${port}" "${cost}" "${pod_id}"
}

cmd_upload() {
  load_state
  echo "Uploading scripts to ${POD_ID} (${REMOTE_DIR}) ..." >&2
  ssh_upload_files "${SSH_MODE}" "${SSH_HOST}" "${SSH_PORT}" \
    "${SCRIPT_DIR}/remote-run.sh" \
    "${SCRIPT_DIR}/record.sh"
  ssh_cmd "${SSH_MODE}" "${SSH_HOST}" "${SSH_PORT}" \
    "chmod +x '${REMOTE_DIR}/remote-run.sh' '${REMOTE_DIR}/record.sh'"
  echo "Done." >&2
}

cmd_ssh() {
  load_state
  local ssh_opts=(
    -i "${SSH_KEY}"
    -o StrictHostKeyChecking=no
    -o UserKnownHostsFile=/dev/null
    -o PreferredAuthentications=publickey
  )
  if [[ "${SSH_MODE}" == "direct" ]]; then
    exec ssh "${ssh_opts[@]}" -p "${SSH_PORT}" "${SSH_USER}@${SSH_HOST}"
  else
    exec ssh "${ssh_opts[@]}" "${SSH_HOST}@ssh.runpod.io"
  fi
}

cmd_status() {
  load_state
  echo "POD_ID=${POD_ID}"
  echo "GPU_TYPE=${GPU_TYPE:-}"
  echo "COST_PER_HOUR=${COST_PER_HOUR}"
  if [[ "${SSH_MODE}" == "direct" ]]; then
    echo "ssh -i ${SSH_KEY} -p ${SSH_PORT} ${SSH_USER}@${SSH_HOST}"
  else
    echo "ssh -i ${SSH_KEY} ${SSH_HOST}@ssh.runpod.io"
  fi
  if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
    local info status
    if info="$(api GET "/pods/${POD_ID}?includeMachine=true" 2>/dev/null)"; then
      status="$(json_pod_field "${info}" desiredStatus || true)"
      echo "desiredStatus=${status}"
    fi
  fi
}

cmd_delete() {
  local pod_id="${1:-}"
  if [[ -z "${pod_id}" ]]; then
    load_state
    pod_id="${POD_ID}"
  fi
  if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
    echo "error: set RUNPOD_API_KEY" >&2
    exit 1
  fi
  delete_pod "${pod_id}"
  if [[ -f "${STATE_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${STATE_FILE}"
    if [[ "${POD_ID:-}" == "${pod_id}" ]]; then
      rm -f "${STATE_FILE}"
    fi
  fi
}

case "${1:-}" in
  -h | --help | "")
    usage
    [[ -n "${1:-}" ]] || exit 1
    exit 0
    ;;
  --delete)
    shift
    cmd_delete "${1:-}"
    ;;
  --ssh)
    cmd_ssh
    ;;
  --upload)
    cmd_upload
    ;;
  --status)
    cmd_status
    ;;
  --*)
    echo "error: unknown option: $1" >&2
    usage >&2
    exit 1
    ;;
  *)
    cmd_create "$1"
    ;;
esac

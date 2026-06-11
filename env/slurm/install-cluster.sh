#!/usr/bin/env bash
# Install or repair the unified flash-dgx Slurm cluster across spark-5da5 + spark-0808.
#
# Run on spark-5da5:
#   cd /home/ricfulop/Desktop/Cursor/flash-modelling
#   sudo env/slurm/install-cluster.sh controller
#   sudo env/slurm/install-cluster.sh compute
#
# The compute step SSHes to dgx-peer as nvidia and configures spark-0808 as a
# Slurm compute node pointing at spark-5da5. You will be prompted for sudo on
# both machines.
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root: sudo $0 controller|compute|status|resume-nodes" >&2
  exit 1
fi

ROLE="${1:-}"
if [[ -z "${ROLE}" ]]; then
  echo "usage: $0 controller|compute|status|resume-nodes" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HOST="$(hostname -s)"
PEER_HOST="spark-0808"
PEER_SSH="dgx-peer"
CONTROLLER_HOST="spark-5da5"
CONTROLLER_ADDR="192.168.100.1"
PEER_ADDR="192.168.100.2"

install_common_files() {
  local gres_src="${SCRIPT_DIR}/gres-${HOST}.conf"
  if [[ ! -f "${gres_src}" ]]; then
    echo "ERROR: missing ${gres_src}" >&2
    exit 1
  fi

  install -d /etc/slurm /var/log/slurm/flash-modelling
  install -m 0644 "${SCRIPT_DIR}/slurm.conf" /etc/slurm/slurm.conf
  install -m 0644 "${gres_src}" /etc/slurm/gres.conf
  install -m 0755 "${SCRIPT_DIR}/prolog.sh" /etc/slurm/prolog.sh
  install -m 0755 "${SCRIPT_DIR}/epilog.sh" /etc/slurm/epilog.sh
  install -m 0755 "${SCRIPT_DIR}/job-run.sh" /etc/slurm/flash-job-run.sh
  install -d /etc/slurm/flash-profiles
  install -m 0755 "${SCRIPT_DIR}/profiles/"*.sh /etc/slurm/flash-profiles/

  grep -q "${CONTROLLER_ADDR}[[:space:]]\+${CONTROLLER_HOST}" /etc/hosts || \
    echo "${CONTROLLER_ADDR} ${CONTROLLER_HOST}" >> /etc/hosts
  grep -q "${PEER_ADDR}[[:space:]]\+${PEER_HOST}" /etc/hosts || \
    echo "${PEER_ADDR} ${PEER_HOST}" >> /etc/hosts
}

ensure_munge() {
  systemctl enable munge >/dev/null 2>&1 || true
  systemctl restart munge
  systemctl is-active --quiet munge
}

maybe_reset_slurm_state() {
  local new_name state_dir="/var/spool/slurm/ctld"
  new_name="$(grep '^ClusterName=' /etc/slurm/slurm.conf | head -1 | cut -d= -f2- | tr -d '[:space:]')"

  systemctl stop slurmctld slurmd 2>/dev/null || true

  if [[ -f "${state_dir}/clustername" ]]; then
    local old_name
    old_name="$(tr -d '[:space:]' < "${state_dir}/clustername")"
    if [[ "${old_name}" != "${new_name}" ]]; then
      echo "Cluster rename ${old_name} -> ${new_name}; clearing Slurm controller/node state"
      rm -rf "${state_dir:?}/"*
      rm -rf /var/spool/slurm/d/*
    fi
  fi
}

install_controller() {
  if [[ "${HOST}" != "${CONTROLLER_HOST}" ]]; then
    echo "Run 'controller' on ${CONTROLLER_HOST}, not ${HOST}" >&2
    exit 1
  fi

  install_common_files
  ensure_munge
  maybe_reset_slurm_state

  systemctl enable slurmctld slurmd >/dev/null 2>&1 || true
  systemctl restart slurmctld
  sleep 2
  systemctl restart slurmd
  sleep 2

  scontrol update NodeName="${CONTROLLER_HOST}" State=RESUME || true
  scontrol update NodeName="${PEER_HOST}" State=RESUME || true

  echo
  echo "Controller installed on ${CONTROLLER_HOST}."
  echo "Next: sudo ${SCRIPT_DIR}/install-cluster.sh compute"
  sinfo -Nel
}

install_compute() {
  if [[ "${HOST}" != "${CONTROLLER_HOST}" ]]; then
    echo "Run 'compute' from ${CONTROLLER_HOST} so it can SSH to ${PEER_SSH}" >&2
    exit 1
  fi

  local munge_key peer_repo="/home/nvidia/Cursor/flash-modelling"
  munge_key="$(cat /etc/munge/munge.key | base64 -w0)"

  echo "Syncing Slurm config to ${PEER_SSH}:${peer_repo}/env/slurm ..."
  ssh -o BatchMode=yes "${PEER_SSH}" "mkdir -p ${peer_repo}/env"
  rsync -a "${SCRIPT_DIR}/" "${PEER_SSH}:${peer_repo}/env/slurm/"

  ssh -o BatchMode=yes "${PEER_SSH}" "sudo bash -s" <<EOF
set -euo pipefail
HOST="\$(hostname -s)"
CONTROLLER_HOST="${CONTROLLER_HOST}"
CONTROLLER_ADDR="${CONTROLLER_ADDR}"
PEER_ADDR="${PEER_ADDR}"
PEER_HOST="${PEER_HOST}"
REPO="${peer_repo}"

install -d /etc/slurm /var/log/slurm/flash-modelling
install -m 0644 "\${REPO}/env/slurm/slurm.conf" /etc/slurm/slurm.conf
install -m 0644 "\${REPO}/env/slurm/gres-\${HOST}.conf" /etc/slurm/gres.conf
install -m 0755 "\${REPO}/env/slurm/prolog.sh" /etc/slurm/prolog.sh
install -m 0755 "\${REPO}/env/slurm/epilog.sh" /etc/slurm/epilog.sh
install -m 0755 "\${REPO}/env/slurm/job-run.sh" /etc/slurm/flash-job-run.sh
install -d /etc/slurm/flash-profiles
install -m 0755 "\${REPO}/env/slurm/profiles/"*.sh /etc/slurm/flash-profiles/

grep -q "\${CONTROLLER_ADDR}[[:space:]]\+\${CONTROLLER_HOST}" /etc/hosts || \
  echo "\${CONTROLLER_ADDR} \${CONTROLLER_HOST}" >> /etc/hosts
grep -q "\${PEER_ADDR}[[:space:]]\+\${PEER_HOST}" /etc/hosts || \
  echo "\${PEER_ADDR} \${PEER_HOST}" >> /etc/hosts

echo '${munge_key}' | base64 -d > /etc/munge/munge.key
chown munge:munge /etc/munge/munge.key
chmod 0400 /etc/munge/munge.key
systemctl enable munge >/dev/null 2>&1 || true
systemctl restart munge
systemctl is-active --quiet munge

systemctl disable --now slurmctld >/dev/null 2>&1 || true
rm -rf /var/spool/slurm/d/*
systemctl enable slurmd >/dev/null 2>&1 || true
systemctl restart slurmd
sleep 2
systemctl is-active --quiet slurmd
echo "Compute-side Slurm configured on \${HOST}"
EOF

  sleep 2
  scontrol update NodeName="${PEER_HOST}" State=RESUME || true
  scontrol reconfigure

  echo
  echo "Compute node ${PEER_HOST} joined cluster ${CONTROLLER_HOST}."
  sinfo -Nel
}

show_status() {
  echo "=== local host: ${HOST} ==="
  systemctl is-active munge slurmctld slurmd 2>/dev/null || true
  sinfo -Nel 2>/dev/null || true
  echo
  echo "=== peer via ${PEER_SSH} ==="
  ssh -o BatchMode=yes "${PEER_SSH}" "hostname -s; systemctl is-active munge slurmd 2>/dev/null || true" || true
}

resume_nodes() {
  scontrol update NodeName="${CONTROLLER_HOST}" State=RESUME || true
  scontrol update NodeName="${PEER_HOST}" State=RESUME || true
  sinfo -Nel
}

case "${ROLE}" in
  controller) install_controller ;;
  compute) install_compute ;;
  status) show_status ;;
  resume-nodes) resume_nodes ;;
  *)
    echo "unknown role: ${ROLE}" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
# Quick health check + smoke test for the flash-dgx Slurm cluster.
#
# Run on spark-5da5 after install:
#   env/slurm/test-cluster.sh
#   env/slurm/test-cluster.sh --submit
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

SUBMIT=false
if [[ "${1:-}" == "--submit" ]]; then
  SUBMIT=true
fi

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

echo "=== flash-dgx Slurm cluster test ==="
echo "host: $(hostname -s)"
echo

echo "[1/4] Controller reachability"
scontrol ping >/dev/null 2>&1 || fail "slurmctld not reachable on $(hostname -s). Run: sudo env/slurm/install-cluster.sh controller"
echo "  OK: slurmctld responding"

echo "[2/4] Node inventory"
sinfo -Nel
idle_nodes="$(sinfo -h -t idle -N | wc -l | tr -d ' ')"
if [[ "${idle_nodes}" -lt 2 ]]; then
  echo "  WARN: expected 2 idle nodes (spark-5da5 + spark-0808); got ${idle_nodes}"
  echo "  If spark-0808 is down/missing, run: sudo env/slurm/install-cluster.sh compute"
else
  echo "  OK: both nodes idle"
fi

echo "[3/4] Peer Slurm daemon (via dgx-peer SSH)"
peer_slurmd="$(ssh -o BatchMode=yes -o ConnectTimeout=4 dgx-peer 'systemctl is-active slurmd 2>/dev/null' || true)"
peer_ctld="$(ssh -o BatchMode=yes -o ConnectTimeout=4 dgx-peer 'systemctl is-active slurmctld 2>/dev/null' || true)"
peer_conf="$(ssh -o BatchMode=yes -o ConnectTimeout=4 dgx-peer "grep '^SlurmctldHost=' /etc/slurm/slurm.conf 2>/dev/null" || true)"
echo "  peer slurmd: ${peer_slurmd:-unknown}"
echo "  peer slurmctld (should be inactive): ${peer_ctld:-unknown}"
echo "  peer config: ${peer_conf:-missing}"
if [[ "${peer_ctld}" == "active" ]]; then
  echo "  WARN: dgx-peer still runs its own controller; run: sudo env/slurm/install-cluster.sh compute"
fi
if [[ "${peer_conf}" != *"spark-5da5"* ]]; then
  echo "  WARN: dgx-peer not pointed at spark-5da5 controller"
fi

if [[ "${SUBMIT}" != true ]]; then
  echo
  echo "Dry run only. Re-run with --submit to launch 30s smoke jobs on each node."
  exit 0
fi

echo "[4/4] Submit smoke jobs (30s sleep, one per node)"
job_local="$(sbatch --parsable --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 \
  --time=00:02:00 --nodelist=spark-5da5 --job-name=slurm-smoke-local \
  --output=runs/slurm-smoke-local-%j.out --error=runs/slurm-smoke-local-%j.err \
  --wrap 'hostname; nvidia-smi -L; sleep 30')"
job_peer="$(sbatch --parsable --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 \
  --time=00:02:00 --nodelist=spark-0808 --job-name=slurm-smoke-peer \
  --output=runs/slurm-smoke-peer-%j.out --error=runs/slurm-smoke-peer-%j.err \
  --wrap 'hostname; nvidia-smi -L; sleep 30')"
echo "  submitted local job ${job_local} on spark-5da5"
echo "  submitted peer job ${job_peer} on spark-0808"
echo "  watch: squeue -u \"\$USER\""

for _ in $(seq 1 60); do
  pending="$(squeue -h -j "${job_local},${job_peer}" 2>/dev/null | wc -l | tr -d ' ')"
  [[ "${pending}" == "0" ]] && break
  sleep 2
done

sacct -j "${job_local},${job_peer}" --format=JobID,JobName,NodeList,State,ExitCode -n || true
echo
echo "Logs:"
ls -1 runs/slurm-smoke-*-"${job_local}".out runs/slurm-smoke-*-"${job_peer}".out 2>/dev/null || true

#!/usr/bin/env bash
# Slurm epilog: record exit status for flash-modelling jobs.
set -euo pipefail

LOG_DIR="${FLASH_SLURM_LOG_DIR:-/var/log/slurm/flash-modelling}"
mkdir -p "${LOG_DIR}" 2>/dev/null || true

{
  echo "=== epilog $(date -Is) ==="
  echo "job_id=${SLURM_JOB_ID:-} exit_code=${SLURM_JOB_EXIT_CODE:-unknown}"
  echo "node=${SLURMD_NODENAME:-${SLURM_NODELIST:-}}"
} >> "${LOG_DIR}/job-${SLURM_JOB_ID:-unknown}.log" 2>/dev/null || true

exit 0

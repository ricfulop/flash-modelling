#!/usr/bin/env bash
# Slurm prolog: lightweight job-start logging for flash-modelling.
set -euo pipefail

LOG_DIR="${FLASH_SLURM_LOG_DIR:-/var/log/slurm/flash-modelling}"
mkdir -p "${LOG_DIR}" 2>/dev/null || true

{
  echo "=== prolog $(date -Is) ==="
  echo "job_id=${SLURM_JOB_ID:-} job_name=${SLURM_JOB_NAME:-} user=${SLURM_JOB_USER:-}"
  echo "node=${SLURMD_NODENAME:-${SLURM_NODELIST:-}} partition=${SLURM_JOB_PARTITION:-}"
  echo "profile=${FLASH_PROFILE:-unset} gres=${SLURM_JOB_GRES:-}"
  echo "cmd=${SLURM_JOB_COMMAND:-}"
} >> "${LOG_DIR}/job-${SLURM_JOB_ID:-unknown}.log" 2>/dev/null || true

exit 0

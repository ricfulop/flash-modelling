#!/usr/bin/env bash
# QE 7.4.1 CPU toolchain (smoke tests and input validation).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

if [[ -f "${CURSOR_ROOT}/use_qe_cpu.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CURSOR_ROOT}/use_qe_cpu.sh" >/dev/null
else
  echo "ERROR: missing ${CURSOR_ROOT}/use_qe_cpu.sh" >&2
  exit 1
fi

export FLASH_PROFILE=qe-cpu
echo "Slurm profile: qe-cpu on ${CURSOR_ROOT}"

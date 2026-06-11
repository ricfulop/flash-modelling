#!/usr/bin/env bash
# BigDFT CPU micromamba environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

if [[ -f "${CURSOR_ROOT}/use_bigdft.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CURSOR_ROOT}/use_bigdft.sh" >/dev/null
else
  echo "ERROR: missing ${CURSOR_ROOT}/use_bigdft.sh" >&2
  exit 1
fi

export FLASH_PROFILE=bigdft
echo "Slurm profile: bigdft on ${CURSOR_ROOT}"

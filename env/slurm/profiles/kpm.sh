#!/usr/bin/env bash
# Torch / MACE / KPM stack for GPU transport sweeps.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [[ -f "${FLASH_MODELLING_ROOT}/env/activate_dgx.sh" ]]; then
  # shellcheck disable=SC1091
  source "${FLASH_MODELLING_ROOT}/env/activate_dgx.sh"
else
  echo "WARNING: missing ${FLASH_MODELLING_ROOT}/env/activate_dgx.sh" >&2
fi

export FLASH_PROFILE=kpm
echo "Slurm profile: kpm (torch/MACE) on ${FLASH_MODELLING_ROOT}"

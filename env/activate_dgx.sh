#!/usr/bin/env bash
# Torch/MACE stack (Stages 1,2,3,5,6). BigDFT (Stage 4) uses its OWN env — see below.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
else
  echo "WARNING: ${REPO_ROOT}/.venv is missing; run 'uv sync' from the repo root." >&2
fi

echo "torch/MACE env ready. GPU=$CUDA_VISIBLE_DEVICES"
echo "NOTE: BigDFT runs in its own micromamba env via use_bigdft.sh (shell-out from dft_bigdft.py)."

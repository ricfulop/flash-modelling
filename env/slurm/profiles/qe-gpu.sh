#!/usr/bin/env bash
# QE 7.5 GPU + NVHPC CUDA (primary confirmatory label path on spark-5da5).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

QE_ROOT="${CURSOR_ROOT}/.local-qe"
NVHPC="${QE_ROOT}/nvhpc"
ARCH="Linux_aarch64"
VERSION="25.11"
CUDA_VER="13.0"
NVHPC_BASE="${NVHPC}/${ARCH}/${VERSION}"

if [[ ! -d "${NVHPC_BASE}" ]]; then
  echo "ERROR: NVHPC tree missing at ${NVHPC_BASE}" >&2
  exit 1
fi

export PATH="${QE_ROOT}/install/qe-7.5-gpu/bin:${QE_ROOT}/src/q-e-qe-7.5/bin:${PATH}"
export LD_LIBRARY_PATH="${NVHPC_BASE}/cuda/${CUDA_VER}/lib64:${NVHPC_BASE}/math_libs/${CUDA_VER}/lib64:${LD_LIBRARY_PATH:-}"
export NVCOMPILERS="${NVHPC}"
export NVHPC_CUDA_HOME="${NVHPC_BASE}/cuda/${CUDA_VER}"
export NVCOMPILER_COMM_LIBS_HOME="${NVHPC_BASE}/comm_libs/${CUDA_VER}"

export FLASH_PROFILE=qe-gpu
echo "Slurm profile: qe-gpu (QE 7.5 + NVHPC) on ${CURSOR_ROOT}"

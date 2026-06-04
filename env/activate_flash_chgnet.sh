#!/usr/bin/env bash
# Source this file from the repo root to enter the CHGNet environment.
MICROMAMBA="${MICROMAMBA:-/home/ricfulop/Desktop/Cursor/.local-bigdft-opencl/bin/micromamba}"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/ricfulop/Desktop/Cursor/.local-flash-chgnet}"
export MAMBA_PKGS_DIRS="${MAMBA_PKGS_DIRS:-${MAMBA_ROOT_PREFIX}/pkgs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${MAMBA_ROOT_PREFIX}/cache}"

if [ ! -x "${MICROMAMBA}" ]; then
  echo "ERROR: micromamba executable not found at ${MICROMAMBA}" >&2
  return 1 2>/dev/null || exit 1
fi

eval "$("${MICROMAMBA}" shell hook --shell bash)"
micromamba activate flash_chgnet
echo "flash_chgnet env ready. $(python --version 2>&1)"

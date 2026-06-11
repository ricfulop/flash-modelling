#!/usr/bin/env bash
# Run a command under a flash-modelling Slurm environment profile.
#
# Usage:
#   FLASH_PROFILE=qe-gpu env/slurm/job-run.sh python3 scripts/27_phase2_qe_gpu_labels.py ...
#   env/slurm/job-run.sh kpm python3 scripts/14_large_lattice_kpm_showcase.py ...
#
# Profiles: kpm | bigdft | qe-cpu | qe-gpu
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <profile> <command...>" >&2
  echo "   or: FLASH_PROFILE=<profile> $0 <command...>" >&2
  exit 2
fi

PROFILE="${FLASH_PROFILE:-}"
if [[ -z "${PROFILE}" ]]; then
  PROFILE="$1"
  shift
fi

PROFILE_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profiles/${PROFILE}.sh"
if [[ ! -f "${PROFILE_SCRIPT}" ]]; then
  echo "ERROR: unknown profile '${PROFILE}' (expected kpm|bigdft|qe-cpu|qe-gpu)" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${PROFILE_SCRIPT}"

cd "${FLASH_MODELLING_ROOT}"
exec "$@"

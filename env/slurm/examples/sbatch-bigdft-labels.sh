#!/usr/bin/env bash
#SBATCH --job-name=bigdft-labels
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err
#
# Example:
#   sbatch env/slurm/examples/sbatch-bigdft-labels.sh \
#     --root runs/autonomous_electrodefect_20260607_041730/geometry_repair_convergence_gate \
#     --max-jobs 1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export FLASH_PROFILE=bigdft
exec /etc/slurm/flash-job-run.sh bigdft python3 scripts/18_phase2_bigdft_labels.py "$@"

#!/usr/bin/env bash
#SBATCH --job-name=kpm-sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err
#SBATCH --nodelist=spark-0808
#
# Example:
#   sbatch env/slurm/examples/sbatch-kpm-sweep.sh \
#     --morphology random --frac 0.35 --label slurm_smoke
#
# KPM sweeps are a good default workload for spark-0808 while spark-5da5
# runs confirmatory QE GPU labels.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export FLASH_PROFILE=kpm
exec /etc/slurm/flash-job-run.sh kpm python3 scripts/14_large_lattice_kpm_showcase.py "$@"

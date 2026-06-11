#!/usr/bin/env bash
#SBATCH --job-name=qe-gpu-labels
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.err
#SBATCH --nodelist=spark-5da5
#
# Example:
#   sbatch env/slurm/examples/sbatch-qe-gpu-labels.sh \
#     --root runs/autonomous_electrodefect_20260607_041730/geometry_repair_convergence_gate \
#     --max-seconds 7200
#
# Note: QE 7.5 GPU + NVHPC is currently installed only on spark-5da5.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export FLASH_PROFILE=qe-gpu
exec /etc/slurm/flash-job-run.sh qe-gpu python3 scripts/27_phase2_qe_gpu_labels.py "$@"

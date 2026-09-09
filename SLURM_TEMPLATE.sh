#!/usr/bin/env bash
#SBATCH --job-name=stitch_umb
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=1-10

set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Create jobs.txt with one umbrella_window.py command per line.
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs.txt)
echo "$CMD"
eval "$CMD"

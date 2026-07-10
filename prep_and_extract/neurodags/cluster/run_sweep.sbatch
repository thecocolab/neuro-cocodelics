#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --job-name=nrd_zapsweep
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
module load StdEnv/2023 python/3.11
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source "/scratch/${USER}/envs/neurodags/bin/activate"
export MPLBACKEND=Agg OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "/scratch/${USER}/cocodelics_neurodags"
mkdir -p logs
echo "===== node $(hostname) job ${SLURM_JOB_ID} ====="
python sweep_zapline.py
echo "===== DONE ====="

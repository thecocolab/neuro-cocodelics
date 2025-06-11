#!/bin/bash
#SBATCH --account=def-kjerbi          
#SBATCH --job-name=lsd_cpu
#SBATCH --time=0-01:30               # 90 min is plenty for 5‑fold CV
#SBATCH --cpus-per-task=6            # match YAML n_jobs
#SBATCH --mem=24G                    # 4 GB × 6 cores (cluster guideline)
#SBATCH --output=logs/%x_%j.out

# 1) software stack
module load StdEnv/2023 python/3.11            # Narval/Béluga CPU modules
source ~/envs/coco/bin/activate                # your virtualenv

# 2) avoid thread oversubscription: one BLAS/OpenMP thread per worker
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 3) run the analysis
srun python examples/lsd_eye_closed.py \
     --config machine-learning/configs/lsd_closed_baseline.yml 
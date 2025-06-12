#!/bin/bash
#SBATCH --account=rrg-shahabkb          
#SBATCH --job-name=psil_single_sensor_all_features
#SBATCH --time=11:59:00               
#SBATCH --cpus-per-task=32           # match YAML n_jobs
#SBATCH --ntasks=1                     # one task with multiple threads
#SBATCH --mem-per-cpu=1G                    # 4 GB × 6 cores (cluster guideline)
#SBATCH --output=logs/psil_%x_%j.out

# 1) software stack
module load StdEnv/2023 python/3.11            # Narval/Béluga CPU modules
source $HOME/neuro-cocodelics/.venv_neurococodelics/bin/activate            # your virtualenv

# 2) avoid thread oversubscription: one BLAS/OpenMP thread per worker
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 3) run the analysis
srun python $HOME/neuro-cocodelics/machine-learning/run_ml.py \
    --config $HOME/neuro-cocodelics/machine-learning/configs/config_psil_single_sensor_all_features.yml
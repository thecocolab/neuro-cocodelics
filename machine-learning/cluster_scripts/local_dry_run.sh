#!/bin/bash
# Local dry-run script for testing before SLURM submission

echo "Starting local dry-run for LSD analysis..."
echo "=========================================="

# Activate environment
source ~/envs/coco/bin/activate

# Set thread limits to match SLURM job
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6  
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_MAX_THREADS=6

echo "Environment configured:"
echo "  - Virtual environment: $(which python)"
echo "  - Thread limits: OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  - Config file: machine-learning/configs/lsd_closed_baseline.yml"
echo ""

# Run the analysis with logging
echo "Running analysis..."
python examples/lsd_eye_closed.py \
       --config machine-learning/configs/lsd_closed_baseline.yml \
       2>&1 | tee local_test.log

echo ""
echo "Dry-run completed! Check local_test.log for full output."
echo "If successful, you can submit with: sbatch lsd_cpu.sbatch" 
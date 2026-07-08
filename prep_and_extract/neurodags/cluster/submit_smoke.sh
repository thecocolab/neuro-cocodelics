#!/bin/bash
# Submit the 1-subject-per-dataset smoke test. Run on the fir login node,
# from the cluster working dir, AFTER build_env.sh has created the venv.
#
#   ssh fir 'cd /scratch/$USER/cocodelics_neurodags && bash cluster/submit_smoke.sh'
#
set -euo pipefail
cd "/scratch/${USER}/cocodelics_neurodags"
mkdir -p logs outputs
sbatch cluster/run_smoke.sbatch

#!/bin/bash
# Standalone bidsification of ONE cocodelics dataset (raw /project -> BIDS shared /project).
#   sbatch --export=DATASET=ketamine cluster/run_bidsify.sbatch
# DATASET in: lsd | ketamine | perampanel | psilocybin | tiagabine
# (working dir on cluster: /scratch/$USER/cocodelics_bids ; the staged bidsification/ dir)
#
#SBATCH --account=def-kjerbi
#SBATCH --job-name=cocodelics_bidsify
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G      # FieldTrip .mat inspection accumulates; 36 files peaked ~16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

module load StdEnv/2023 python/3.11
# numpy C-ext loads correctly on the compute node only with the cvmfs PYTHONPATH dropped.
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source "/scratch/${USER}/envs/cocodelics-bids/bin/activate"
export XDG_CACHE_HOME="/scratch/${USER}/.cache"
export MPLBACKEND=Agg

cd "/scratch/${USER}/cocodelics_bids"
mkdir -p logs
DS="${DATASET:?set DATASET=lsd|ketamine|perampanel|psilocybin|tiagabine}"

echo "===== node $(hostname)  job ${SLURM_JOB_ID}  dataset ${DS} ====="
python -c "import mne, mne_bids, sovabids, scipy, pandas; print('imports OK')"

case "$DS" in
  lsd)
    echo "== LSD step 2: prepare (Jordan-Venkatesh corrections -> meg_bids.csv) =="
    python MEG_LSD/2bidsifying_preparation.py
    echo "== LSD step 3: write BIDS =="
    python MEG_LSD/3bidsifying.py ;;
  ketamine)    python MEG_ketamine/1bidsifying.py ;;
  perampanel)  python MEG_perampanel/1bidsifying.py ;;
  psilocybin)  python MEG_psilocybin/1bidsifying.py ;;
  tiagabine)   python MEG_tiagabine/1bidsifying.py ;;
  *) echo "unknown DATASET '$DS'"; exit 2 ;;
esac

# Summary count — scope to OUR yorguin-owned bids/ dir (fully traversable). `|| true` so a
# stray find non-zero can't trip `set -e` and mark an otherwise-successful job FAILED.
OUT="/home/yorguin/projects/rrg-kjerbi/shared/neuro-cocodelics/bids"
count=$(find "$OUT" -name '*_meg.fif' 2>/dev/null | wc -l || true)
echo "===== output: ${count} sub-*_meg.fif under bids/ ====="
echo "===== DONE ${DS} (${SLURM_JOB_ID}) ====="

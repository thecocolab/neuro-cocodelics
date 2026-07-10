#!/bin/bash
# DFA long-window (120 s / 60 s overlap) SMOKE — 1 subject/dataset. Runs only the DFA
# chain (reuses cached PickedRaw), verifies PrepDur120Ov60 + the DFA feature + aggregates.
#   sbatch --exclude=fc30557 cluster/run_dfa_smoke.sbatch
#SBATCH --account=def-kjerbi
#SBATCH --job-name=nrd_dfa_smoke
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
module load StdEnv/2023 python/3.11
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source "/scratch/${USER}/envs/neurodags/bin/activate"
export XDG_CACHE_HOME="/scratch/${USER}/.cache" MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "/scratch/${USER}/cocodelics_neurodags"
mkdir -p logs outputs
PIPE=pipeline_cocodelics.yml
DS=datasets_cocodelics.yml
JSONL="logs/run_${SLURM_JOB_ID}.jsonl"

CHAIN=(PickedRaw PrepDur120Ov60 Ep_DetrendedFluctuationDur120
       detrendedFluctuationMeanEpochsDur120 detrendedFluctuationSDEpochsDur120)
DF=(detrendedFluctuationMeanEpochsDur120 detrendedFluctuationSDEpochsDur120)
RUN_ARGS=(); for x in "${CHAIN[@]}"; do RUN_ARGS+=(--derivative "$x"); done
DF_ARGS=();  for x in "${DF[@]}";    do DF_ARGS+=(--include-derivative "$x"); done

echo "===== node $(hostname) job ${SLURM_JOB_ID} ====="
python -c "import neurodags, antropy, mne, xarray, netCDF4; print('imports OK', neurodags.__version__)"
echo "===== RUN DFA chain, 1 file/dataset ====="
neurodags --log-level INFO --log-file "$JSONL" \
  run "$PIPE" -d "$DS" --max-files-per-dataset 1 "${RUN_ARGS[@]}" --n-jobs "${SLURM_CPUS_PER_TASK}"
echo "===== dataframe (wide) ====="
neurodags --log-level WARNING dataframe "$PIPE" -d "$DS" --max-files-per-dataset 1 \
  "${DF_ARGS[@]}" --format wide --output "outputs/feats_dfa_smoke.csv" --n-jobs "${SLURM_CPUS_PER_TASK}"
echo "===== DONE ${SLURM_JOB_ID} ====="

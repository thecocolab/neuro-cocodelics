#!/bin/bash
# neurodags cocodelics feature extraction — SMOKE TEST: 1 subject per dataset.
# Submit from the login node:  sbatch cluster/run_smoke.sbatch
# (working dir on cluster: /scratch/$USER/cocodelics_neurodags)
#
#SBATCH --account=def-kjerbi
#SBATCH --job-name=nrd_cocodelics_smoke
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4        # each joblib worker loads a full CTF MEG file (~4-6 GB)
#SBATCH --mem=64G                # 8 workers @ 32G OOM'd (MaxRSS 33.5G); 4 workers fit here
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

module load StdEnv/2023 python/3.11
# `module load python` injects PYTHONPATH=/cvmfs/.../site-packages, whose setuptools +
# arch-specific numpy leak into the venv and break numpy's C-extension import on the
# compute node. Drop it so the venv is fully self-contained (its own PyPI numpy).
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source "/scratch/${USER}/envs/neurodags/bin/activate"

# keep caches/plots off $HOME
export XDG_CACHE_HOME="/scratch/${USER}/.cache"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1  # joblib parallelizes over files

cd "/scratch/${USER}/cocodelics_neurodags"
mkdir -p logs outputs

PIPE=pipeline_cocodelics.yml
DS=datasets_cocodelics.yml            # the real shared-BIDS config (1 file/dataset via --max-files)
JSONL="logs/run_${SLURM_JOB_ID}.jsonl"

# CLASSICAL battery only for this smoke — the EXPERIMENTAL phi/IIT nodes (Ep_Atoms) are slow
# and huge on real 271-channel data; validate them separately. Passing these --derivative
# flags makes neurodags run only them + their dependencies (prep, per-epoch, aggregation).
CLASSICAL=(
  lzivComplexityMeanEpochs higuchiFdMeanEpochs higuchiFdVarEpochs
  katzFdMeanEpochs katzFdSDEpochs petrosianFdMeanEpochs svdEntropyMeanEpochs
  numZerocrossMeanEpochs permEntropyMeanEpochs spectralEntropyMeanEpochs
  hjorthMeanEpochs detrendedFluctuationMeanEpochs
)
RUN_ARGS=();  for d in "${CLASSICAL[@]}"; do RUN_ARGS+=(--derivative "$d"); done
DF_ARGS=();   for d in "${CLASSICAL[@]}"; do DF_ARGS+=(--include-derivative "$d"); done

echo "===== node: $(hostname)  job: ${SLURM_JOB_ID} ====="
python -c "import neurodags, antropy, neurokit2, fooof, mne, xarray, netCDF4; print('imports OK', neurodags.__version__)"

echo "===== RUN classical, 1 file/dataset, on the shared BIDS ====="
neurodags --log-level INFO --log-file "$JSONL" \
  run "$PIPE" -d "$DS" --max-files-per-dataset 1 "${RUN_ARGS[@]}" --n-jobs "${SLURM_CPUS_PER_TASK}"

echo "===== dataframe (wide) ====="
neurodags --log-level WARNING dataframe "$PIPE" -d "$DS" --max-files-per-dataset 1 \
  "${DF_ARGS[@]}" --format wide --output "outputs/feats_wide_smoke.csv" --n-jobs "${SLURM_CPUS_PER_TASK}"

echo "===== bridge -> coco-pipe aggregate columns ====="
python to_cocopipe_columns.py outputs/feats_wide_smoke.csv "outputs/aggregate_smoke_raw.csv" || true

echo "===== DONE ${SLURM_JOB_ID} ====="

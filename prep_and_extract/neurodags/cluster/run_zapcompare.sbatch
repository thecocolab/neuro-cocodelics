#!/bin/bash
# ZapLine A/B comparison — SMOKE (1 subject/dataset). Runs BOTH denoising variants
# (…Fixed = adaptive off, …Adaptive = adaptive on; both line_freq=50 n_harmonics=2)
# through the full classical battery so 50/100 Hz residual can be compared at the
# feature level, plus a PrepSpectrum QC figure per variant.
#   sbatch --exclude=fc30557 cluster/run_zapcompare.sbatch
# (working dir on cluster: /scratch/$USER/cocodelics_neurodags)
#
#SBATCH --account=def-kjerbi
#SBATCH --job-name=nrd_zapcompare
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4        # each joblib worker loads a full CTF MEG file (~4-6 GB)
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

module load StdEnv/2023 python/3.11
unset PYTHONPATH                 # cvmfs setuptools/numpy leak breaks the venv's C-ext on compute
export PYTHONNOUSERSITE=1
source "/scratch/${USER}/envs/neurodags/bin/activate"

export XDG_CACHE_HOME="/scratch/${USER}/.cache"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "/scratch/${USER}/cocodelics_neurodags"
mkdir -p logs outputs

PIPE=pipeline_cocodelics_zapcompare.yml
DS=datasets_cocodelics.yml
JSONL="logs/run_${SLURM_JOB_ID}.jsonl"

# `--derivative X` runs ONLY X (deps NOT auto-added), so we must enumerate the WHOLE
# DerivativeList. Derive both the run set and the for_dataframe set straight from the
# pipeline yaml so this stays correct if the generator adds/removes features.
mapfile -t RUN < <(python -c "import yaml;d=yaml.safe_load(open('$PIPE'));print('\n'.join(d['DerivativeList']))")
mapfile -t DF  < <(python -c "import yaml;d=yaml.safe_load(open('$PIPE'));dd=d['DerivativeDefinitions'];print('\n'.join(n for n in d['DerivativeList'] if dd[n].get('for_dataframe')))")
RUN_ARGS=(); for x in "${RUN[@]}"; do RUN_ARGS+=(--derivative "$x"); done
DF_ARGS=();  for x in "${DF[@]}";  do DF_ARGS+=(--include-derivative "$x"); done
echo "run derivatives: ${#RUN[@]}   dataframe features: ${#DF[@]}"

echo "===== node: $(hostname)  job: ${SLURM_JOB_ID} ====="
python -c "import neurodags, antropy, neurokit2, fooof, mne, xarray, netCDF4; print('imports OK', neurodags.__version__)"

echo "===== RUN both variants, 1 file/dataset, on the shared BIDS ====="
neurodags --log-level INFO --log-file "$JSONL" \
  run "$PIPE" -d "$DS" --max-files-per-dataset 1 "${RUN_ARGS[@]}" --n-jobs "${SLURM_CPUS_PER_TASK}"

echo "===== dataframe (wide): both …Fixed and …Adaptive feature columns ====="
neurodags --log-level WARNING dataframe "$PIPE" -d "$DS" --max-files-per-dataset 1 \
  "${DF_ARGS[@]}" --format wide --output "outputs/feats_zapcompare_wide.csv" --n-jobs "${SLURM_CPUS_PER_TASK}"

echo "===== DONE ${SLURM_JOB_ID} ====="

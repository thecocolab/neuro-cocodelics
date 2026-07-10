#!/bin/bash
# FULL RUN — per-dataset job array (one task per dataset). Computes + caches all
# classical + DFA derivatives (NO experimental) to /scratch. A dependent job
# (run_full_dataframe.sbatch) builds the combined dataframe afterwards.
#   AID=$(sbatch --parsable --exclude=fc30557 cluster/run_full_array.sbatch)
#   sbatch --dependency=afterok:$AID --exclude=fc30557 cluster/run_full_dataframe.sbatch
#
#SBATCH --account=def-kjerbi
#SBATCH --job-name=nrd_full
#SBATCH --array=0-4              # 0=lsd(226) 1=ketamine(36) 2=perampanel(40) 3=psilo(30) 4=tiagabine(30)
#SBATCH --time=05:00:00          # LSD is the long pole (~226 files @ ~140 s/file / 4 workers ~2.2 h)
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G                # smoke peaked at MaxRSS 63.5 GiB @ 64G (4 workers) — bump for margin
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail

module load StdEnv/2023 python/3.11
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source "/scratch/${USER}/envs/neurodags/bin/activate"
export XDG_CACHE_HOME="/scratch/${USER}/.cache"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "/scratch/${USER}/cocodelics_neurodags"
mkdir -p logs outputs tmp_yaml

DATASETS=(lsd ketamine perampanel psilocybin tiagabine)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
PIPE=pipeline_cocodelics.yml
DS_YAML="tmp_yaml/datasets_${DS}.yml"
JSONL="logs/run_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jsonl"

# per-dataset datasets yaml: skip every dataset except $DS
python - "$DS" "$DS_YAML" <<'PYEOF'
import sys, yaml
ds, out = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open("datasets_cocodelics.yml"))
for k, v in d.items():
    if isinstance(v, dict) and "name" in v:
        v["skip"] = (k != ds)
yaml.safe_dump(d, open(out, "w"), sort_keys=False)
PYEOF

# production set = DerivativeList MINUS experimental tier (same as run_production.sbatch)
read -r -d '' PYSEL <<'PYEOF' || true
import yaml, sys
d = yaml.safe_load(open("pipeline_cocodelics.yml"))
dl = d["DerivativeList"]
EXCLUDE = {
 "Ep_Atoms","Ep_InfoDyn","Ep_IID","Ep_IIT","Ep_FisherInformation",
 "Ep_PowerSpectrumMultitaper","Ep_Harmonicity",
 "infodynMeanEpochs","infodynSDEpochs","iidMeanEpochs","iidSDEpochs",
 "iitMeanEpochs","iitSDEpochs","fisherInformationMeanEpochs","fisherInformationSDEpochs",
 "harmonicityMeanEpochs","harmonicitySDEpochs",
}
print("\n".join(x for x in dl if x not in EXCLUDE))
PYEOF
mapfile -t RUN < <(python -c "$PYSEL")
RUN_ARGS=(); for x in "${RUN[@]}"; do RUN_ARGS+=(--derivative "$x"); done

echo "===== node $(hostname) array ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID} dataset=${DS} run=${#RUN[@]} ====="
python -c "import neurodags, antropy, mne, xarray, netCDF4; print('imports OK', neurodags.__version__)"

echo "===== RUN ${DS} (all files) ====="
neurodags --log-level INFO --log-file "$JSONL" \
  run "$PIPE" -d "$DS_YAML" "${RUN_ARGS[@]}" --n-jobs "${SLURM_CPUS_PER_TASK}" --skip-errors

echo "===== DONE ${DS} (task ${SLURM_ARRAY_TASK_ID}) ====="

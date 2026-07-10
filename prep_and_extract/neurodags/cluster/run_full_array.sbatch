#!/bin/bash
# FULL RUN — one array task PER FILE (neurodags-idiomatic fan-out). A global file index
# (--only-index=$SLURM_ARRAY_TASK_ID over the full datasets yaml; 362 files, 0-361) means
# each task processes ONE recording through all classical+DFA derivatives.
#
# Why per-file (not per-dataset): the validation probes showed (a) a 600s file needs ~30G
# alone but 4 concurrent workers peaked at 102.7G (96G would OOM), and (b) per-dataset LSD
# (226 files) would be ~10h and blow the wall limit. One file/task => one worker (~30G) +
# short (~3-6 min) + resumable. Already-cached files (tiagabine/ketamine + smoke) return fast.
#
#   AID=$(sbatch --parsable --array=0-361%40 --exclude=fc30557 cluster/run_full_array.sbatch)
#   sbatch --dependency=afterok:$AID --exclude=fc30557 cluster/run_full_dataframe.sbatch
#
#SBATCH --account=def-kjerbi
#SBATCH --job-name=nrd_full
#SBATCH --array=0-361%40          # 362 files; cap 40 concurrent (good cluster citizen)
#SBATCH --time=01:30:00           # one 600s file is ~3-6 min; generous margin
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G                 # single-file peak ~30G (4-worker probe was 102.7G @128G)
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
mkdir -p logs outputs
PIPE=pipeline_cocodelics.yml
DS=datasets_cocodelics.yml
JSONL="logs/run_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jsonl"

# production set = DerivativeList MINUS the experimental phi/IIT/Fisher/Harmonicity tier
read -r -d '' PYSEL <<'PYEOF' || true
import yaml
dl = yaml.safe_load(open("pipeline_cocodelics.yml"))["DerivativeList"]
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

echo "===== node $(hostname) task ${SLURM_ARRAY_TASK_ID} (file index ${SLURM_ARRAY_TASK_ID}) run=${#RUN[@]} ====="
neurodags --log-level INFO --log-file "$JSONL" \
  run "$PIPE" -d "$DS" --only-index "${SLURM_ARRAY_TASK_ID}" "${RUN_ARGS[@]}" --n-jobs 1 --skip-errors

echo "===== DONE task ${SLURM_ARRAY_TASK_ID} ====="

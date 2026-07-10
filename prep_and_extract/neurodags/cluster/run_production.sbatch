#!/bin/bash
# Production feature extraction: classical battery + DFA long-window (NO experimental tier).
# Writes derivatives to /scratch (see datasets_cocodelics.yml). Parameterized by MAXFILES:
#   scale smoke : sbatch --exclude=fc30557 --export=ALL,MAXFILES=5 cluster/run_production.sbatch
#   full run    : sbatch --exclude=fc30557 cluster/run_production.sbatch        (MAXFILES unset = all)
#
#SBATCH --account=def-kjerbi
#SBATCH --job-name=nrd_production
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4        # each joblib worker loads a full CTF MEG file (~4-6 GB)
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
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
JSONL="logs/run_${SLURM_JOB_ID}.jsonl"
MAXFILES="${MAXFILES:-0}"          # 0 => no cap (all files)

# Production set = DerivativeList MINUS the experimental phi/IIT/Fisher/Harmonicity tier.
# Derived from the yaml so new classical/DFA features are auto-included; only edit EXCLUDE
# if new EXPERIMENTAL derivatives are added.
read -r -d '' PYSEL <<'PYEOF' || true
import yaml, sys
d = yaml.safe_load(open("pipeline_cocodelics.yml"))
defs = d["DerivativeDefinitions"]; dl = d["DerivativeList"]
EXCLUDE = {
 "Ep_Atoms","Ep_InfoDyn","Ep_IID","Ep_IIT","Ep_FisherInformation",
 "Ep_PowerSpectrumMultitaper","Ep_Harmonicity",
 "infodynMeanEpochs","infodynSDEpochs","iidMeanEpochs","iidSDEpochs",
 "iitMeanEpochs","iitSDEpochs","fisherInformationMeanEpochs","fisherInformationSDEpochs",
 "harmonicityMeanEpochs","harmonicitySDEpochs",
}
run = [x for x in dl if x not in EXCLUDE]
df  = [x for x in run if defs[x].get("for_dataframe")]
which = sys.argv[1]
print("\n".join(run if which == "run" else df))
PYEOF
mapfile -t RUN < <(python -c "$PYSEL" run)
mapfile -t DF  < <(python -c "$PYSEL" df)
RUN_ARGS=(); for x in "${RUN[@]}"; do RUN_ARGS+=(--derivative "$x"); done
DF_ARGS=();  for x in "${DF[@]}";  do DF_ARGS+=(--include-derivative "$x"); done

CAP_ARGS=()
[ "$MAXFILES" != "0" ] && CAP_ARGS=(--max-files-per-dataset "$MAXFILES")
SUFFIX="full"; [ "$MAXFILES" != "0" ] && SUFFIX="cap${MAXFILES}"

echo "===== node $(hostname) job ${SLURM_JOB_ID}  MAXFILES=${MAXFILES}  run=${#RUN[@]} df=${#DF[@]} ====="
python -c "import neurodags, antropy, mne, xarray, netCDF4; print('imports OK', neurodags.__version__)"

echo "===== RUN classical + DFA ====="
neurodags --log-level INFO --log-file "$JSONL" \
  run "$PIPE" -d "$DS" "${CAP_ARGS[@]}" "${RUN_ARGS[@]}" --n-jobs "${SLURM_CPUS_PER_TASK}"

echo "===== dataframe (wide) ====="
neurodags --log-level WARNING dataframe "$PIPE" -d "$DS" "${CAP_ARGS[@]}" \
  "${DF_ARGS[@]}" --format wide --output "outputs/feats_${SUFFIX}_wide.csv" --n-jobs "${SLURM_CPUS_PER_TASK}"

echo "===== bridge -> coco-pipe aggregate columns ====="
python to_cocopipe_columns.py "outputs/feats_${SUFFIX}_wide.csv" "outputs/aggregate_${SUFFIX}_raw.csv" || true

echo "===== DONE ${SLURM_JOB_ID} ====="

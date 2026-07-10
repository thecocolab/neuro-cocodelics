#!/bin/bash
# FULL RUN — combined dataframe. Runs AFTER run_full_array.sbatch (all datasets cached).
# Reads the cached for_dataframe derivatives across all 5 datasets -> one wide CSV +
# the coco-pipe-format aggregate. Submit with a dependency on the array:
#   sbatch --dependency=afterok:$AID --exclude=fc30557 cluster/run_full_dataframe.sbatch
#
#SBATCH --account=def-kjerbi
#SBATCH --job-name=nrd_full_df
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
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
mkdir -p outputs

PIPE=pipeline_cocodelics.yml
DS=datasets_cocodelics.yml            # full (all 5 unskipped) — reads from cache

# for_dataframe derivatives MINUS the experimental tier
read -r -d '' PYSEL <<'PYEOF' || true
import yaml
d = yaml.safe_load(open("pipeline_cocodelics.yml"))
defs = d["DerivativeDefinitions"]; dl = d["DerivativeList"]
EXCLUDE = {"infodynMeanEpochs","infodynSDEpochs","iidMeanEpochs","iidSDEpochs",
 "iitMeanEpochs","iitSDEpochs","fisherInformationMeanEpochs","fisherInformationSDEpochs",
 "harmonicityMeanEpochs","harmonicitySDEpochs"}
print("\n".join(x for x in dl if x not in EXCLUDE and defs[x].get("for_dataframe")))
PYEOF
mapfile -t DF < <(python -c "$PYSEL")
DF_ARGS=(); for x in "${DF[@]}"; do DF_ARGS+=(--include-derivative "$x"); done
echo "dataframe features: ${#DF[@]}"

echo "===== combined dataframe (wide) across all 5 datasets ====="
neurodags --log-level WARNING dataframe "$PIPE" -d "$DS" \
  "${DF_ARGS[@]}" --format wide --output "outputs/feats_full_wide.csv" --n-jobs "${SLURM_CPUS_PER_TASK}"

echo "===== bridge -> coco-pipe aggregate columns ====="
python to_cocopipe_columns.py "outputs/feats_full_wide.csv" "outputs/aggregate_full_raw.csv" || true

echo "===== DONE ====="

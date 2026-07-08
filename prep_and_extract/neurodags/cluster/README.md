# Cluster drivers — neurodags cocodelics feature extraction

Driver scripts kept **for traceability**: this is exactly what produced the
`MEG_*/derivatives_neurodags/` outputs on `fir`.

| File | Role |
|------|------|
| `build_env.sh` | Create the uv venv at `/scratch/$USER/envs/neurodags` (wheelhouse + PyPI, wheels-only). Run once, on login. |
| `datasets_smoke.yml` | 1-subject-per-dataset override for the smoke test. |
| `run_smoke.sbatch` | The Slurm job: env → `count-inputs` → `status` → `run` → `dataframe` → coco-pipe rename → `status`. |
| `submit_smoke.sh` | `sbatch run_smoke.sbatch`. |

## Order of operations

```bash
# 0. (local) stage the pipeline dir onto the cluster:
rsync -av prep_and_extract/neurodags/  fir:/scratch/$USER/cocodelics_neurodags/

# 1. build the env (login node — needs internet; wheels-only, no compile)
ssh fir 'bash -l /scratch/$USER/cocodelics_neurodags/cluster/build_env.sh'

# 2. submit the smoke test (1 subject x 5 datasets, ~20 files)
ssh fir 'cd /scratch/$USER/cocodelics_neurodags && bash cluster/submit_smoke.sh'

# 3. watch
ssh fir 'squeue --me'
ssh fir 'sacct -j <id> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode'
ssh fir 'seff <id>'
```

## Job resources (smoke)

`--account=def-kjerbi --time=01:30:00 --cpus-per-task=8 --mem=32G`. joblib
parallelizes over the ~20 files (`--n-jobs 8`); BLAS pinned to 1 thread to avoid
oversubscription. Outputs: `derivatives_neurodags/` (per-subject `.nc`),
`outputs/feats_wide_smoke.csv`, `outputs/aggregate_smoke_raw.csv`, and a per-run
JSONL trace `logs/run_<jobid>.jsonl`.

## Scaling to all subjects

Drop `-d cluster/datasets_smoke.yml` to use the full `datasets_cocodelics.yml`, or
generate proper per-derivative arrays:
`neurodags slurm-script pipeline_cocodelics.yml --pattern chained` (then add the
`#SBATCH --account=def-kjerbi` + `module load` + `source .../activate` header).

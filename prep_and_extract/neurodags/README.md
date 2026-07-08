# Feature extraction with neurodags (port of the coco-pipe r2c pipeline)

This directory reimplements cocodelics **feature extraction** on top of
[neurodags](https://github.com/yjmantilla/neurodags) — a declarative DAG framework
for M/EEG derivatives — replacing the coco-pipe `legacy/r2c_project` feature stage.

Scope: the **classical complexity/entropy battery** (the feature set the downstream
ML actually consumes). This maps 1:1 onto neurodags **built-in nodes** — no custom
nodes are required.

The `v1` (Fisher/Harmonicity) and `v2` (phi/IIT) "scientific" features are ALSO
provided, but as **EXPERIMENTAL / NEEDS-VALIDATION** custom nodes in
`experimental_features.py` (loaded alongside `custom_nodes.py` via the
`new_definitions` list). They are faithful ports of the r2c `@meeg_refactor`
implementations but have **not** been validated numerically against the original r2c
outputs — keep them clearly separate from the validated classical battery. They run
from a **separate `neurodags-experimental` venv** (`numpy<2`, because the `biotuner`
dependency pins it) with extra deps `phyid` + `neurokit2` + `biotuner` — built by
`cluster/build_env.sh` alongside the main env.

## Files

| File | Purpose |
|------|---------|
| `pipeline_cocodelics.yml` | The DAG: preprocessing + per-epoch features + epoch aggregation. |
| `datasets_cocodelics.yml` | The 5 MEG datasets, `fir` cluster paths + `local` dev placeholders. |
| `to_cocopipe_columns.py` | Rename neurodags wide CSV → coco-pipe `feature-*.spaces-*` aggregate for the existing ML. |

## What it computes

`prepare()` parity with `../legacy/r2c_project/redefinitions_cocosprint.py`:
**notch [50,100,150] Hz → bandpass 0.1–150 Hz → epoch 30 s / 20 s-overlap → resample 600 Hz**
(the `PrepDur30Ov20` derivative). Then, per 30 s epoch, per sensor:

`lzivComplexity`, `higuchiFd` (mean + var), `katzFd` (mean + SD), `petrosianFd`,
`svdEntropy`, `numZerocross`, `permEntropy`, `spectralEntropy`, `detrendedFluctuation`,
and Hjorth `mobility` + `complexity`. Each is aggregated over epochs
(`MeanEpochs`/`SDEpochs`/`VarEpochs`) into one scalar per sensor — the dataframe columns.

This reproduces the 12 features used by `machine-learning/` (`viz/aggregate_ml_results.py`
`feat_imp_templates`), plus `permEntropy`.

## Run it

neurodags is a separate package/repo. Use its environment (or `pip install neurodags`):

```bash
NRD=~/code/neurodags/.venv/bin/neurodags   # or just `neurodags` if installed on PATH

# 1. sanity-check the config + DAG
$NRD validate pipeline_cocodelics.yml
$NRD dag pipeline_cocodelics.yml --html dag.html      # optional visual

# 2. dry-run / status (what's cached vs missing) — needs the data mount
$NRD status  pipeline_cocodelics.yml
$NRD dry-run pipeline_cocodelics.yml --output plan.csv

# 3. run all derivatives (dependency-sorted). Compute-heavy → cluster only (see below).
$NRD run pipeline_cocodelics.yml --n-jobs -1

# 4. assemble the wide feature table
$NRD dataframe pipeline_cocodelics.yml --format wide --output feats_wide.csv --n-jobs -1

# 5. bridge to the existing ML
python to_cocopipe_columns.py feats_wide.csv aggregate@raw.csv
#    -> then data/split_csv.py consumes aggregate@raw.csv exactly as before.
```

`mount_point` in the pipeline is `fir`. Override the datasets for a local mirror with
`-d /abs/path/to/local_datasets.yml`.

## On the cluster (`fir`, Alliance)

Data lives at `~/scratch/datasets/cocodelics/MEG_*/` (group `def-kjerbi`). This pipeline
writes **new** derivatives to `MEG_*/derivatives_neurodags/`, leaving the coco-pipe
`derivatives/features@prepDur30Ov20` intact as a parity reference.

**Never run `neurodags run/dataframe` on a login node** — it imports MNE and extracts
features (real compute). Use `salloc`/`sbatch`. neurodags can generate the array scripts:

```bash
neurodags slurm-script pipeline_cocodelics.yml --pattern chained --output submit_pipeline.sh
```

Then edit the generated `run_one_derivative.sh` worker header to add:
```bash
#SBATCH --account=def-kjerbi
module load StdEnv/2023 python/3.11
source ~/path/to/neurodags-venv/bin/activate
```
Smoke-test one dataset with `--max-files-per-dataset 1` before the full array; check
`neurodags status` and `seff`. See `~/code/cluster-utilities/AGENT.md` for the rules.

## Column naming (neurodags ↔ coco-pipe)

neurodags flattens xarray derivatives to `<Derivative>.nc@spaces-<sensor>` (2-D derivatives
like Hjorth add `hjorthComponents-<mobility|complexity>_`). `to_cocopipe_columns.py` maps
these to coco-pipe's `feature-<name>.spaces-<sensor>` and parses `subject/session/task`
from the BIDS path, so `data/split_csv.py`, `machine-learning/run_ml.py`, and `viz/*`
run unchanged.

## Status & known deltas (verified 2026-07-07)

- **Verified end-to-end on synthetic MEG** (8 CTF-named channels, 90 s): all 24 derivatives
  compute, 0 errors, wide dataframe + coco-pipe rename produce the expected columns.
- **Not yet run on real cluster data** — do a 1-subject smoke test per dataset first.
- **Parity knobs to check against the old coco-pipe derivatives** (`derivatives/features@prepDur30Ov20`):
  - `spectral_entropy` / `lziv_complexity` **normalization** — antropy defaults return
    un-normalized values here; confirm whether coco-pipe normalized and add
    `normalize: true` to those node `args` if so.
  - Channel selection: this pipeline computes on **all** channels in the `.fif`. coco-pipe
    likely operated on the ~271 CTF magnetometers only. Add a `keep_channels` step (or a
    MEG-pick) if you need to restrict; otherwise the ML slices to its sensor list anyway.
  - `higuchi_fd` `kmax`, `perm_entropy`/`svd_entropy` order/delay — antropy defaults;
    match to coco-pipe params if strict numerical parity is required.
- Only deviation from a naive port: `binarize_with_median` can't read an MNE `.fif`, so the
  LZiv chain routes through an explicit `Ep_XarrayEpochs` (`meeg_to_xarray`) derivative.

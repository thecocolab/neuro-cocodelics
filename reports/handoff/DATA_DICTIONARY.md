# cocodelics feature table — data dictionary

`cocodelics_features_notch.csv` — MEG complexity/fractal features for drug-vs-placebo
decoding, 5 drug datasets. This is the table the accompanying report's ML results were
computed on. See `cocodelics_report.html` for the full pipeline + results.

## Shape
- **362 rows** = one MEG recording each (subject × session; for LSD also × task).
- **3,827 columns** = 5 identifier columns + **3,822 feature columns** (14 features × 273 sensors).

## Identifier columns (first 5)
| column | meaning |
|---|---|
| `dataset` | drug dataset: `lsd`, `ketamine`, `perampanel`, `psilocybin`, `tiagabine` |
| `subject` | subject code (unique **within** a dataset; each subject has a placebo + a drug session) |
| `session` | `placebo` or the drug name (`lsd`/`ketamine`/…) |
| `task` | `resting` (ketamine/perampanel/psilocybin/tiagabine) or the LSD task (`Open1/2`, `Closed1/2`, `Music`, `Video`) |
| `target` | **0 = placebo, 1 = drug** (derived from `session`) — the classification label |

## Feature columns
Named **`feature-<measure>.spaces-<sensor>`**, e.g. `feature-lzivComplexityMeanEpochs.spaces-MLC11`.

- **14 measures** (per-epoch value → mean/variance over epochs, per sensor):
  Lempel-Ziv complexity; Higuchi fractal dim (mean, var); Katz fractal dim (mean, SD);
  Petrosian fractal dim; SVD entropy; number of zero-crossings; permutation entropy;
  spectral entropy; Hjorth mobility; Hjorth complexity; detrended fluctuation (DFA, 30 s
  epochs); **alpha-envelope DFA** (`alphaEnvelopeDfa`, computed on a single 240 s continuous
  8–12 Hz Hilbert-envelope window — the long-range-temporal-correlation exponent).
- **273 sensors** — CTF magnetometer names (`M[LRZ][A-Z]…`). This is the **union** across
  datasets; an individual dataset recorded 271–273 of them.

## NaN — how to handle
Overall NaN ≈ **0.59 %**, entirely **sensor-driven** (not a feature-computation problem):
- **structural** (~90 %): a sensor absent from a whole dataset → its columns are all-NaN
  for that dataset (the 273 is the union across sites).
- **dead channels** (~10 %): a present sensor that was flat in some recordings — dominated
  by one systematically-bad sensor, **`MRT36`** (dead in ~all perampanel + psilocybin recordings).

Recommended: **per dataset, drop feature columns containing any NaN** (what our ML does), or
impute. Never row-drop (you'd lose recordings).

## Loading with coco-pipe

The feature columns use coco-pipe's `feature-<measure>.spaces-<sensor>` convention, so the
CSV loads directly with `coco_pipe.io.load_data` (note: **comma-separated**, target column
is `target`):

```python
from coco_pipe.io import load_data
dc = load_data("cocodelics_features_notch.csv", mode="tabular",
               target_col="target", sep=",",
               meta_columns=["dataset", "subject", "session", "task"])
# dc.X -> (362, 3822) features ; dc.y -> 0/1 ; dc.meta / dc.obs_table -> the id columns
```

For the decoding we ran (one problem per dataset, subject-grouped), the simplest route is to
subset per dataset and feed the decoding `Experiment` directly:

```python
import pandas as pd, numpy as np
from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import CVConfig, LogisticRegressionConfig, RandomForestClassifierConfig

df = pd.read_csv("cocodelics_features_notch.csv")
feat = [c for c in df.columns if c.startswith("feature-")]
sub = df[df.dataset == "ketamine"]
cols = sub[feat].dropna(axis=1).columns.tolist()          # drop any-NaN cols for this dataset
cfg = ExperimentConfig(
    task="classification",
    models={"LogReg": LogisticRegressionConfig(max_iter=2000),
            "RandomForest": RandomForestClassifierConfig(n_estimators=200, random_state=42)},
    metrics=["accuracy", "roc_auc", "f1"],
    cv=CVConfig(strategy="group_kfold", n_splits=5, auto_reduce_n_splits=True),
    use_scaler=True, n_jobs=-1)
res = Experiment(cfg).run(sub[cols].to_numpy(float),
                          (sub.session != "placebo").astype(int).to_numpy(),
                          groups=sub["subject"].to_numpy())
print(res.summary())
```

(Full drivers: `machine-learning/explore_decoding.py` + `perm_test.py` in the repo.)

## How to model it (what we did)
- Binary **drug (1) vs placebo (0)**, one problem per dataset (LSD optionally per task, or a
  task-average `lsd-avg`).
- **Group by `subject`** for cross-validation (GroupKFold) — each subject appears in only one
  fold, so scores reflect generalisation to new people. Each subject contributes exactly one
  placebo + one drug recording → a paired design (use within-subject label permutation for a
  null).
- Standardise features. L2 logistic regression and random forest were robust here; gradient
  boosting / SVC were unstable in this p≫n regime (3,822 features, 30–40 recordings/dataset).

## Provenance
Preprocessing: pick CTF MEG sensors → **notch 50/100/150 Hz** → band-pass 0.1–150 → epoch
30 s/20 s-overlap → resample 600 Hz. (A ZapLine-denoised variant of this table also exists;
notch is the production version — see the report.) Pipeline:
`thecocolab/neuro-cocodelics`, branch `feature-extraction-neurodags`, built on
[neurodags](https://github.com/yjmantilla/neurodags).

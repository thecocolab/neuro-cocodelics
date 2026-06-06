# Bidsification Comparison: `bidsification/` vs `r2c_project/redefinitions_cocosprint.py`

---

## Architecture

| | `bidsification/` | `r2c_project/redefinitions_cocosprint.py` |
|-|-----------------|------------------------------------------|
| Form | Standalone executable scripts, top-level code | Functions injected into pipeline via `redefine_bidsify` |
| Entry point | Run directly | Called by pipeline as `bidsify(source_path, bids_path, DATASET_CFG, pipeline_cfg)` |
| Config | Hardcoded paths in each script | Paths read from `DATASET_CFG` / `pipeline_cfg` |
| Coverage | LSD, perampanel, psilocybin, ketamine, tiagabine | LSD, perampanel, psilocybin only — **ketamine/tiagabine absent** |

---

## LSD — Preparation phase

**`bidsification/MEG_LSD/2bidsifying_preparation.py`** vs **`thanks_jordan_venkatesh()`**

| Aspect | Standalone | r2c |
|--------|-----------|-----|
| SOURCE_PATH | Hardcoded: `/home/yorguin/scratch/real_scratch/data/MEG_LSDV2/meg_data` | From `DATASET_CFG['bidsify']['paths']['source_path'][MOUNT]` |
| Pattern | Hardcoded: `.../MEG_LSDV2/meg_data/%subjectNumber%_LSD_%session%_%task%.ds` | From `DATASET_CFG['bidsify']['pattern'][MOUNT]` |
| ID_file | `os.path.join(os.path.dirname(__file__), 'IDs.xlsx')` | From `DATASET_CFG['bidsify']['ID_file'][MOUNT]` |
| Output | Saves `meg_bids.csv` to script dir | Returns DataFrame in memory to caller |
| Jordan correction logic | **Identical** | **Identical** |

---

## LSD — Conversion phase

**`bidsification/MEG_LSD/3bidsifying.py`** vs **`lsd_bids_conversion()`**

| Aspect | Standalone | r2c |
|--------|-----------|-----|
| Input | Reads `meg_bids.csv` from disk | Receives DataFrame in memory |
| OUTPUT_PATH | Hardcoded: `/home/yorguin/scratch/data/MEG_LSD/` | From `bids_path` argument |
| Existence check | None — always calls `write_raw_bids()` | Checks `os.path.isfile(bidsTree.fpath)`, skips if exists |
| `BIDSPath` args | `BIDSPath(subject, session, task, root=BIDS_ROOT)` | `BIDSPath(subject, session, task, root=BIDS_ROOT, datatype='meg', suffix='meg', extension='.fif')` |
| Read method | `mne.io.read_raw(filepath, preload=True)` | `mne.io.read_raw(filepath, preload=True)` — identical |

---

## FieldTrip datasets (perampanel, psilocybin, ketamine, tiagabine)

**Standalone `bidsification/MEG_*/1bidsifying.py`** vs **`fieldtrip_to_bids()`**

| Aspect | Standalone | r2c |
|--------|-----------|-----|
| Datasets covered | All four (one script each) | Perampanel + psilocybin only; **ketamine and tiagabine absent** |
| Source paths | Hardcoded: `def-kjerbi/data/MEG_*/` | From `source_path` argument |
| Output paths | Hardcoded: `scratch/data/MEG_*/meg_data_BIDS` | From `bids_path` argument |
| Inspection phase | Top-level code (runs on import) | Inside function, with `.pkl` cache check |
| Existence check | None | Checks `os.path.isfile(bidsTree.fpath)`, skips if exists |
| `BIDSPath` args | `BIDSPath(subject, session, task, root=BIDS_ROOT)` | `BIDSPath(subject, session, task, root=BIDS_ROOT, datatype='meg', suffix='meg', extension='.fif')` |
| `breakpoint()` | None | **Line 835** — hangs non-interactive runs |

### Perampanel — filename pattern

| | Standalone | r2c |
|-|-----------|-----|
| Pattern | `PMP_%session%_%subject%_%number%.mat` (hardcoded `PMP_` prefix) | `%ignore%_%session%_%subject%_%number%.mat` (generic prefix) |

Both yield same parsed fields (`session`, `subject`, `number`). r2c version more robust to prefix changes.

### Psilocybin — key functional difference

| | Standalone (`bidsification/MEG_psilocybin/1bidsifying.py`) | r2c (`fieldtrip_to_bids()`) |
|-|-----------------------------------------------------------|------------------------------|
| Data used | Full raw recording as-is | **Crops to InfusionStop → RestStop window** |
| Cropping logic | None | Finds `InfusionStop` and `RestStop` in `metadata['event']`, calls `raw.crop(tmin=infusion_sec, tmax=rest_sec)` |
| Raises if events missing | No | Yes — `ValueError("InfusionStop or RestStop event not found")` |
| Psilocybin pattern | `%ignore%/%session%_%subject%_%number%.mat` | `%ignore%/%session%_%subject%_%number%.mat` — identical |

The standalone script converts the full file. The r2c version extracts only the post-infusion rest window — a **substantive scientific difference** in what data gets stored in BIDS.

### Ketamine and tiagabine (standalone only)

Both have standalone scripts with identical structure to perampanel. Neither is handled in `fieldtrip_to_bids()`. In `datasets_cocosprint.yml` both have no `bidsify` section, indicating they were processed standalone and are treated as already-BIDS inputs by the pipeline.

| Dataset | Standalone pattern | session_bids logic |
|---------|-------------------|-------------------|
| ketamine | `KET_%session%_%subject%_%number%.mat` | `'placebo'` if `PLA` else `'ketamine'` |
| tiagabine | `TGB_%session%_%subject%_%number%.mat` | `'placebo'` if `PLA` else `'tiagabine'` |

---

## Summary of divergence

| Change | Direction | Impact |
|--------|-----------|--------|
| Config-driven paths | standalone → r2c | Good: portable across mounts |
| Existence check before write | standalone → r2c | Good: idempotent reruns |
| `BIDSPath` with `datatype/suffix/extension` | standalone → r2c | Changes existence check path resolution; more precise |
| Psilocybin infusion crop | absent in standalone → added in r2c | **Substantive**: r2c produces shorter event-locked data vs full recording |
| Ketamine/tiagabine support | standalone only | Gap: pipeline cannot bidsify these two datasets if run fresh |
| `breakpoint()` in `fieldtrip_to_bids()` line 835 | new regression in r2c | Bug: hangs all non-interactive FieldTrip bidsification runs |
| Jordan correction logic | identical in both | Both correct |
| Psilocybin filename pattern | identical in both | — |

# Bidsification Comparison: `bidsification/` vs `r2c_project/redefinitions_cocosprint.py`

Last updated: 2026-06-06 — improvements from r2c migrated to standalone; BIDS output paths updated.

---

## Terminology

**"FieldTrip scripts"** = the 4 standalone scripts handling FieldTrip `.mat` source data:
`MEG_perampanel/1bidsifying.py`, `MEG_psilocybin/1bidsifying.py`, `MEG_ketamine/1bidsifying.py`, `MEG_tiagabine/1bidsifying.py`.
Named after their r2c counterpart `fieldtrip_to_bids()`. Contrast with LSD which uses CTF `.ds` files.

---

## Architecture

| | `bidsification/` | `r2c_project/redefinitions_cocosprint.py` |
|-|-----------------|------------------------------------------|
| Form | Standalone executable scripts, top-level code | Functions injected into pipeline via `redefine_bidsify` |
| Entry point | Run directly | Called by pipeline as `bidsify(source_path, bids_path, DATASET_CFG, pipeline_cfg)` |
| Config | Hardcoded paths in each script | Paths read from `DATASET_CFG` / `pipeline_cfg` |
| Coverage | LSD, perampanel, psilocybin, ketamine, tiagabine | LSD, perampanel, psilocybin only — ketamine/tiagabine absent |

---

## Output paths

### BIDS roots (where `sub-*/` live)

| Dataset | Standalone | r2c |
|---------|-----------|-----|
| LSD | `/home/yorguin/scratch/datasets/cocodelics/MEG_LSD` | from `bids_path` arg (pipeline config) |
| perampanel | `/home/yorguin/scratch/datasets/cocodelics/MEG_perampanel` | from `bids_path` arg |
| psilocybin | `/home/yorguin/scratch/datasets/cocodelics/MEG_psilocybin` | from `bids_path` arg |
| ketamine | `/home/yorguin/scratch/datasets/cocodelics/MEG_ketamine` | not handled in r2c |
| tiagabine | `/home/yorguin/scratch/datasets/cocodelics/MEG_tiagabine` | not handled in r2c |

### Metadata inspection files (FieldTrip scripts only — CSV/pkl from `.mat` inspection)

These are NOT BIDS output. The FieldTrip scripts keep a separate `OUTPUT_PATH` for metadata:

| Dataset | Metadata OUTPUT_PATH |
|---------|---------------------|
| perampanel | `/home/yorguin/scratch/data/MEG_perampanel/` |
| psilocybin | `/home/yorguin/scratch/data/MEG_psilocybin/` |
| ketamine | `/home/yorguin/scratch/data/MEG_ketamine/` |
| tiagabine | `/home/yorguin/scratch/data/MEG_tiagabine/` |

These paths have **not** been updated to the new root — metadata stays in old scratch location.

---

## Current parity status

| Feature | Standalone | r2c |
|---------|:---------:|:---:|
| Existence check before write | ✓ | ✓ |
| `BIDSPath` with `datatype/suffix/extension` | ✓ | ✓ |
| Psilocybin InfusionStop→RestStop crop | ✓ | ✓ |
| Jordan correction logic (LSD) | ✓ | ✓ |
| Config-driven paths | — (intentional) | ✓ |
| Ketamine/tiagabine support | ✓ | — (gap remains) |
| `breakpoint()` regression | — | ✓ line 835 (unfixed in r2c) |

---

## LSD — Preparation phase

**`bidsification/MEG_LSD/2bidsifying_preparation.py`** vs **`thanks_jordan_venkatesh()`**

| Aspect | Standalone | r2c |
|--------|-----------|-----|
| SOURCE_PATH | Hardcoded: `/home/yorguin/scratch/real_scratch/data/MEG_LSDV2/meg_data` | From `DATASET_CFG['bidsify']['paths']['source_path'][MOUNT]` |
| Pattern | Hardcoded: `.../MEG_LSDV2/meg_data/%subjectNumber%_LSD_%session%_%task%.ds` | From `DATASET_CFG['bidsify']['pattern'][MOUNT]` |
| ID_file | `os.path.join(os.path.dirname(__file__), 'IDs.xlsx')` | From `DATASET_CFG['bidsify']['ID_file'][MOUNT]` |
| Output | Saves `meg_bids.csv` to script dir | Returns DataFrame in memory |
| Jordan correction logic | **Identical** | **Identical** |

No BIDS writing here — no path changes needed.

---

## LSD — Conversion phase

**`bidsification/MEG_LSD/3bidsifying.py`** vs **`lsd_bids_conversion()`**

| Aspect | Standalone | r2c |
|--------|-----------|-----|
| Input | Reads `meg_bids.csv` from disk | Receives DataFrame in memory |
| BIDS_ROOT | `/home/yorguin/scratch/datasets/cocodelics/MEG_LSD` | From `bids_path` argument |
| Existence check | ✓ skips if file exists | ✓ skips if file exists |
| `BIDSPath` args | `BIDSPath(..., datatype='meg', suffix='meg', extension='.fif')` | same |
| Read method | `mne.io.read_raw(filepath, preload=True)` | same |

**Functionally identical.**

---

## FieldTrip datasets

### Perampanel, ketamine, tiagabine

| Aspect | Standalone | r2c |
|--------|-----------|-----|
| BIDS_ROOT | `/home/yorguin/scratch/datasets/cocodelics/MEG_<dataset>` | from `bids_path` arg (perampanel only) |
| Metadata OUTPUT_PATH | `/home/yorguin/scratch/data/MEG_<dataset>/` | from `source_path` arg |
| Existence check | ✓ | ✓ |
| `BIDSPath` args | `BIDSPath(..., datatype='meg', suffix='meg', extension='.fif')` | same |
| Perampanel filename pattern | `PMP_%session%_%subject%_%number%.mat` | `%ignore%_%session%_%subject%_%number%.mat` |
| Ketamine/tiagabine | ✓ handled | **absent** |

Perampanel pattern difference is cosmetic — both parse the same fields.

### Psilocybin

| Aspect | Standalone | r2c |
|--------|-----------|-----|
| BIDS_ROOT | `/home/yorguin/scratch/datasets/cocodelics/MEG_psilocybin` | from `bids_path` arg |
| Metadata OUTPUT_PATH | `/home/yorguin/scratch/data/MEG_psilocybin/` | from `source_path` arg |
| Existence check | ✓ | ✓ |
| `BIDSPath` args | `BIDSPath(..., datatype='meg', suffix='meg', extension='.fif')` | same |
| Infusion crop | ✓ InfusionStop→RestStop | ✓ InfusionStop→RestStop |
| Filename pattern | `%ignore%/%session%_%subject%_%number%.mat` | same |

**Functionally identical.**

---

## Remaining divergences

| Issue | Location | Impact |
|-------|----------|--------|
| `breakpoint()` at line 835 | r2c `fieldtrip_to_bids()` only | Hangs non-interactive FieldTrip runs in pipeline |
| Ketamine/tiagabine missing | r2c `fieldtrip_to_bids()` | Pipeline cannot bidsify these two datasets if run fresh |
| Metadata inspection paths not updated | FieldTrip standalone scripts | Metadata CSV/pkl still write to old `scratch/data/MEG_*/` |
| Config-driven vs hardcoded paths | by design | No functional impact |
| In-memory vs CSV handoff (LSD) | by design | No functional impact |

# Bidsification Comparison: `bidsification/` vs `r2c_project/redefinitions_cocosprint.py`

Last updated: 2026-06-06 — improvements from r2c migrated back to standalone scripts.

---

## Architecture

| | `bidsification/` | `r2c_project/redefinitions_cocosprint.py` |
|-|-----------------|------------------------------------------|
| Form | Standalone executable scripts, top-level code | Functions injected into pipeline via `redefine_bidsify` |
| Entry point | Run directly | Called by pipeline as `bidsify(source_path, bids_path, DATASET_CFG, pipeline_cfg)` |
| Config | Hardcoded paths in each script | Paths read from `DATASET_CFG` / `pipeline_cfg` |
| Coverage | LSD, perampanel, psilocybin, ketamine, tiagabine | LSD, perampanel, psilocybin only — ketamine/tiagabine absent |

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
| `breakpoint()` regression | — | ✓ line 835 (unfixed) |

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

No changes needed here — no BIDS writing in this script.

---

## LSD — Conversion phase

**`bidsification/MEG_LSD/3bidsifying.py`** vs **`lsd_bids_conversion()`**

| Aspect | Standalone | r2c |
|--------|-----------|-----|
| Input | Reads `meg_bids.csv` from disk | Receives DataFrame in memory |
| OUTPUT_PATH | Hardcoded: `/home/yorguin/scratch/data/MEG_LSD/` | From `bids_path` argument |
| Existence check | ✓ skips if file exists | ✓ skips if file exists |
| `BIDSPath` args | `BIDSPath(..., datatype='meg', suffix='meg', extension='.fif')` | `BIDSPath(..., datatype='meg', suffix='meg', extension='.fif')` |
| Read method | `mne.io.read_raw(filepath, preload=True)` | `mne.io.read_raw(filepath, preload=True)` |

**Now identical in behavior.**

---

## FieldTrip datasets

### Perampanel, ketamine, tiagabine

| Aspect | Standalone | r2c |
|--------|-----------|-----|
| Existence check | ✓ | ✓ (perampanel only) |
| `BIDSPath` args | `BIDSPath(..., datatype='meg', suffix='meg', extension='.fif')` | same |
| Perampanel pattern | `PMP_%session%_%subject%_%number%.mat` | `%ignore%_%session%_%subject%_%number%.mat` |
| Ketamine/tiagabine | ✓ handled | **absent** |

Perampanel pattern difference is cosmetic — both parse the same fields. Standalone uses hardcoded `PMP_` prefix; r2c uses generic `%ignore%`. Either works.

### Psilocybin

| Aspect | Standalone | r2c |
|--------|-----------|-----|
| Existence check | ✓ | ✓ |
| `BIDSPath` args | `BIDSPath(..., datatype='meg', suffix='meg', extension='.fif')` | same |
| Infusion crop | ✓ InfusionStop→RestStop | ✓ InfusionStop→RestStop |
| Raises if events missing | ✓ `ValueError` | ✓ `ValueError` |
| Pattern | `%ignore%/%session%_%subject%_%number%.mat` | `%ignore%/%session%_%subject%_%number%.mat` |

**Now identical in behavior.**

---

## Remaining divergences

| Issue | Location | Impact |
|-------|----------|--------|
| `breakpoint()` at line 835 | r2c `fieldtrip_to_bids()` only | Hangs non-interactive FieldTrip runs in pipeline |
| Ketamine/tiagabine missing | r2c `fieldtrip_to_bids()` | Pipeline cannot bidsify these two datasets if run fresh |
| Config-driven vs hardcoded paths | by design | No functional impact; standalone is intentionally hardcoded |
| In-memory vs CSV handoff (LSD) | by design | No functional impact |

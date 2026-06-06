# r2c_project Report

**8 files total.** Project: `cocosprint`. Target: ComputeCanada (Cedar).

---

## Files

| File | Purpose |
|------|---------|
| `datasets_cocosprint.yml` | Dataset configs: paths, BIDS roots, participants |
| `pipeline_cocosprint.yml` | **v1** — Fisher Information + Harmonicity |
| `pipeline_cocosprint2.yml` | **v2** — Phi/IIT measures (Atoms, IID, IIT, InfoDyn) |
| `redefinitions_cocosprint.py` | Custom `bidsify()` + `prepare()` injected into pipeline |
| `requirements.txt` | Full deps |
| `requirements_cc_noindex.txt` | Minimal deps for CC no-index env |
| `requirements_extra.txt` | MNE stack + entropy libs + IID decomp |
| `requirements_epilepsy.txt` | mat73, ssqueezepy |

---

## Datasets (`datasets_cocosprint.yml`)

5 MEG datasets, all at `~/scratch/data/`, all 60Hz line noise, output `.fif`:

| Dataset | Source format | BIDSify method | Notes |
|---------|--------------|----------------|-------|
| `lsd` | CTF `.ds` | `no_bidsify` (custom) | 20 subjects, LSD vs placebo, complex ID corrections |
| `perampanel` | FieldTrip `.mat` | `no_bidsify` | Epilepsy drug, resting |
| `psilocybin` | FieldTrip `.mat` | `bidsify` | Crops to InfusionStop→RestStop window |
| `ketamine` | `.fif` (pre-BIDS) | — | Already BIDS |
| `tiagabine` | `.fif` (pre-BIDS) | — | GABA drug |

---

## `redefinitions_cocosprint.py` — Key logic

- **`bidsify()`**: dispatcher — calls `thanks_jordan_venkatesh()` + `lsd_bids_conversion()` for LSD; `fieldtrip_to_bids()` for perampanel/psilocybin.
- **`thanks_jordan_venkatesh()`**: parses LSD `.ds` files, applies Jordan Venkatesh's hand-written correction notes (misnamed subject IDs like `231109-1`→`230911-1`, typo tasks `Opoen1`→`Open1`, `Audio`→`Music`, drops spurious `Music2`/`Video2`/`Closed12`), assigns placebo/LSD sessions by date lookup, validates all 20 subjects have 6 tasks × 2 sessions.
- **`lsd_bids_conversion()`**: iterates corrected DataFrame → `write_raw_bids()`.
- **`fieldtrip_to_bids()`**: loads FieldTrip `.mat` → MNE RawArray → BIDS. Psilocybin crops to infusion window. **Has `breakpoint()` at line 835** — hangs in non-interactive runs.
- **`prepare()`**: overrides default cocopipe prepare. Notch 50/100/150 Hz, bandpass 0.1–150 Hz, `make_fixed_length_epochs`, resample to 600 Hz.

---

## Pipeline stages (shared structure both v1 and v2)

```
0_inspect → 1_dataset2bids → 3_preprocess → 4_features → aggregate → scalingAndFolding → ml → eda
```

`scalingAndFolding`, `ml`, `eda` sections **identical** in both versions.

---

## v1 vs v2 — Detailed Diff

### 1. Preprocessing pipelines defined

| | v1 | v2 |
|-|----|----|
| Defined preps | `prepDur30Ov15` only | `prepDur30Ov20`, `prepDur20Ov10`, `prepDur10Ov5`, `prepSingleEpoch` |
| Active (`prep_list`) | `prepDur30Ov15` | `prepDur30Ov20` + `prepSingleEpoch` |
| Active in `prep_inspection` | `prepDur30Ov15` | `prepSingleEpoch` only (`prepDur30Ov20` commented out) |

**Epoch overlap change**: v1 uses 15s overlap on 30s epochs; v2 uses **20s overlap** on 30s epochs. All v2 prep configs also explicitly set `downsample: null`, `normalization: False`, `filter_args: null` — v1 left these implicit.

**`prepSingleEpoch`** (new in v2): `epoch_config = 'SingleEpoch'` → in `prepare()`, produces one epoch spanning the full recording (`duration=raw.times[-1], overlap=0`).

### 2. Feature pipeline configurations defined

| | v1 | v2 |
|-|----|----|
| Defined feature pipelines | `features@prepDur30Ov15`, `features@prepSingleEpoch` | `features@prepDur30Ov20`, `features@prepDur20Ov10`, `features@prepDur10Ov5`, `features@prepSingleEpoch` |
| Active (`feature_pipeline_list`) | `features@prepDur30Ov15` | `features@prepDur30Ov20` (others commented) |

### 3. Active feature list — core scientific difference

**v1 active features** (in `features@prepDur30Ov15`):
```
FisherInformationMeanEpochs
FisherInformationSDEpochs
HarmonicityMeanEpochsFixed
HarmonicitySDEpochsFixed
```

**v2 active features** (in `features@prepDur30Ov20`):
```
Atoms
InfoDynMeanEpochs
InfoDynSDEpochs
IIDMeanEpochs
IIDSDEpochs
IITMeanEpochs
IITSDEpochs
```

**Scientific shift**: v1 = ordinal Fisher Information + spectral Harmonicity. v2 = phi-based measures from [integrated-info-decomp](https://github.com/Imperial-MIND-lab/integrated-info-decomp) (`requirements_extra.txt`).

### 4. New feature definitions in v2 (absent from v1)

All three phi measures share the same compute graph rooted at `Atoms`:

```
Atoms  (single_atoms: tau=5, kind=gaussian, redundancy=MMI)
  ├── InfoDyn  (atoms_results key=InformationDynamics, agg=mean-sum)
  │     ├── InfoDynMeanEpochs  (np.mean over epochs)
  │     └── InfoDynSDEpochs   (np.std over epochs)
  ├── IID  (atoms_results key=IntegratedInformationDecomposition, agg=mean-sum)
  │     ├── IIDMeanEpochs
  │     └── IIDSDEpochs
  └── IIT  (atoms_results key=IntegratedInformationTheory, agg=mean-sum)
        ├── IITMeanEpochs
        └── IITSDEpochs
```

### 5. Feature definitions removed in v2

v1 defined these (configured but not all ran); v2 drops them entirely:

- `Harmonicity` chain (`PowerSpectrum` → `feature_harmonicity` → `process_harmonicity_output` → `agg_numpy` with `np.nanmean`/`np.nanstd`)
- `FisherInformation` chain (`fisher_information_feature`: delay=1, dimension=3)
- `RateEntropy` (`rate_entropy_feature`, kmax=10)
- `Distance2Criticality` (`chaos_feature`, sigma=0.5, downsample=True)

### 6. `FooofFromAverage` chain changed

**v1:**
```yaml
chain:
  - feature: PowerSpectrumAverage
  - function: spectrum_keep_magnetometers   # filters to magnetometers only
    args: {dummy: None}
  - function: fooof_from_average
    args:
      internal_kwargs:
        FOOOF: {aperiodic_mode: knee}
        fit: {freq_range: [1,90]}
      freq_res: 0.2                         # explicit freq resolution
```

**v2:**
```yaml
chain:
  - feature: PowerSpectrumAverage
  - function: fooof_from_average            # no magnetometer filtering
    args:
      internal_kwargs:
        FOOOF: {aperiodic_mode: knee}
        fit: {freq_range: [1,90]}
                                            # no freq_res param
```

v2 removes the magnetometer-only filter step and drops explicit `freq_res`.

### 7. New `features@prepDur20Ov10` and `features@prepDur10Ov5` pipelines (v2 only, inactive)

Both defined but commented out of `feature_pipeline_list`. Would run the full classical complexity battery:

```
PowerSpectrum, PowerSpectrumAverage, RelativeBandPowerFromAverageSpectrum,
FooofFromAverage, RelativeBandPowerFromAverageFooof, detrendedFluctuation,
lzivComplexityMeanEpochs, sampleEntropyMeanEpochs, spectralEntropyMeanEpochs,
appEntropyMeanEpochs, hjorthParams, hjorthMobilityMeanEpochs,
hjorthComplexityMeanEpochs, numZerocrossMeanEpochs, permEntropyMeanEpochs,
svdEntropyMeanEpochs, higuchiFdMeanEpochs, higuchiFdVarEpochs,
katzFdMeanEpochs, katzFdSDEpochs, petrosianFdMeanEpochs, entropyMultiscaleMeanEpochs
```

### 8. Aggregate configs

| | v1 | v2 |
|-|----|----|
| `feature_aggregate_list` | `MeanEpochs@prepDur30Ov15` | `MeanEpochs@prepDur30Ov20` + `Epochs@prepSingleEpoch` |
| `MeanEpochs@*` features | HarmonicityMean/SD, FisherInformationMean/SD | IIDMean/SD, InfoDynMean/SD, IITMean/SD |
| `Epochs@prepSingleEpoch` (v2 only) | — | `fooofExponentWelch`, `fooofOffsetWelch` |
| `MeanEpochs@prepDur20Ov10` / `@prepDur10Ov5` (v2 only) | — | defined, not in active list |

v2 adds a second aggregate track: single-epoch Welch FOOOF (exponent + offset) as scalar summaries alongside the phi features.

### 9. `feature_return` registry

v1 registers: `HarmonicityMeanEpochsFixed`, `HarmonicitySDEpochsFixed`, `FisherInformationMeanEpochs`, `FisherInformationSDEpochs`.

v2 replaces with: `IIDMeanEpochs`, `IIDSDEpochs`, `InfoDynMeanEpochs`, `InfoDynSDEpochs`, `IITMeanEpochs`, `IITSDEpochs`. All other registered features (classical complexity) identical between versions.

---

## Notable issues

- `redefinitions_cocosprint.py:835` — `breakpoint()` left in `fieldtrip_to_bids()`, hangs non-interactive runs
- `features@prepSingleEpoch` in v2 is configured with Welch features but **not in `feature_pipeline_list`** — Welch features won't compute, yet `Epochs@prepSingleEpoch` aggregate expects `fooofExponentWelch`/`fooofOffsetWelch` to exist
- `requirements_epilepsy.txt` exists but no epilepsy dataset in `datasets_cocosprint.yml`

# Recording durations per dataset — and the DFA window choice

*Generated 2026-07-09 from a header-only survey of the freshly-bidsified BIDS on `fir`
(`/home/yorguin/projects/rrg-kjerbi/shared/neuro-cocodelics/bids/MEG_<D>/**/*_meg.fif`).
Reproduce with [`duration_survey.py`](./duration_survey.py) in the neurodags cluster env.*

## Why this exists

DFA (detrended fluctuation analysis) is **window-length-sensitive**: it estimates a
scaling exponent from fluctuations across box sizes, so the usable range of time
scales is bounded by the window length. The classical battery epochs at **30 s**,
which is short for DFA's slow (long-memory) dynamics. To add a dedicated longer DFA
window that is **comparable across datasets**, we first need the recording-length
distribution per dataset — a fixed window can be no longer than the shortest recording.

## Durations

| dataset | n | min (s) | median (s) | max (s) | notes |
|---|---:|---:|---:|---:|---|
| ketamine | 36 | 600.0 | 600.0 | 600.0 | uniform 10 min |
| perampanel | 40 | 600.0 | 600.0 | 600.0 | uniform 10 min |
| tiagabine | 30 | 300.0 | 300.0 | 300.0 | uniform 5 min |
| LSD | 226 | 420.0 | 420.0 | 600.0 | ~7 min; tasks mostly 420 s (see below) |
| **psilocybin** | 30 | **244.1** | 326.1 | 427.2 | **shortest**; cropped InfusionStop→RestStop, variable |

**Overall shortest recording = 244.1 s** (a psilocybin file) → hard ceiling for any
window shared across all datasets.

LSD by task (n, min/median/max s):

| task | n | min | median | max |
|---|---:|---:|---:|---:|
| Closed1 | 38 | 420.0 | 420.0 | 600.0 |
| Closed2 | 38 | 420.0 | 420.0 | 420.0 |
| Music | 36 | 420.0 | 420.0 | 552.1 |
| Open1 | 38 | 420.0 | 420.0 | 420.0 |
| Open2 | 38 | 420.0 | 420.0 | 420.0 |
| Video | 38 | 420.0 | 420.0 | 420.0 |

## Non-overlapping windows per recording (worst case = shortest recording in each dataset)

| window | ketamine | perampanel | tiagabine | LSD | psilocybin | fits all? |
|---:|---:|---:|---:|---:|---:|---|
| 30 s | 20 | 20 | 10 | 14 | 8 | yes |
| 60 s | 10 | 10 | 5 | 7 | 4 | yes |
| **120 s** | **5** | **5** | **2** | **3** | **2** | **yes** |
| 180 s | 3 | 3 | 1 | 2 | 1 | yes (1 min) |
| 240 s | 2 | 2 | 1 | 1 | 1 | barely (psilo 244 s) |

## Decision — **120 s window, 60 s overlap**

Rationale:
- **~4× the DFA scale range** vs the current 30 s battery (probes box sizes up to ~30 s).
- **Fits every dataset** with ≥2 non-overlapping windows; the 60 s overlap lifts the
  short recordings to ~3 windows (psilocybin 244 s → 3, tiagabine 300 s → 4) so we can
  average DFA across windows within a recording and estimate its variability.
- Comfortable margin over the 244 s floor (unlike 180/240 s, which give only 1 window
  on the short datasets and barely fit psilocybin).
- `sfreq` stays **600 Hz** so DFA box sizes in seconds are comparable across datasets.

Rejected alternatives: 60 s (little gain over 30 s), 180/240 s (no within-recording
averaging on short datasets, tight fit), whole-recording (length varies 244–600 s →
DFA scale range not comparable across datasets).

## Implementation

`pipeline_cocodelics.yml` gained a **separate** long-window prep that re-epochs the
same denoised+bandpassed raw and feeds **only** DFA (the rest of the battery stays on
`PrepDur30Ov20`):

- `PrepDur120Ov60` — PickedRaw → ZapLine (adaptive, `n_harmonics=2`) → bandpass 0.1–150
  → epoch **120 s / 60 s overlap** → resample 600 Hz.
- `Ep_DetrendedFluctuationDur120` → `detrendedFluctuationMeanEpochsDur120` +
  `detrendedFluctuationSDEpochsDur120` (mean + within-recording SD across windows).

The 30 s `detrendedFluctuationMeanEpochs` is kept for continuity/comparison.

> **Perf note (full scale):** this recomputes ZapLine a second time per subject. If
> wall-clock matters, factor the denoised+filtered continuous raw into its own cached
> derivative and epoch both preps from it (CPU vs one large cached `.fif`/subject —
> see `TODO.md` §4).

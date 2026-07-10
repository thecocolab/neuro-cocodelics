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

## Decision — **single 240 s continuous window, alpha-envelope DFA**

The window has to fit the **shortest recording in every dataset**. Two floors bind it:
psilocybin has one outlier at **244 s** (next 285 s, then all 28 others ≥300 s) and
**tiagabine is uniformly 300 s**. So ≤240 s keeps all 362 recordings with margin to trim
filter/Hilbert edges; ~280 s would drop the one short psilocybin file and squeeze
tiagabine; >300 s would drop all tiagabine. 240 s is the max that keeps everyone.

Chosen: **one 240 s continuous window** (not overlapping epochs). A literature pass
(Hardstone et al. 2012; typical M/EEG DFA uses minutes of continuous data) showed:
- DFA robustness grows with continuous length; the canonical fit range is **max box =
  signal_length / 10** (→ 24 s at 240 s), min box a few seconds. A single long window
  maximises the scale range and is comparable across datasets (identical length).
- The within-recording averaging the old 120 s-overlap plan gave is worth less than one
  long, continuous, comparable fit.
- Canonically DFA is run on the **amplitude envelope of a narrow band** (alpha 8–12 Hz),
  not broadband — so the dedicated DFA is alpha-envelope.

Rejected: 120 s/60 s-overlap epochs (shorter scale range, less standard); pushing past
240 s (drops recordings / eats edge margin for negligible scale gain).

## Implementation

`pipeline_cocodelics.yml`:
- **`DenoisedRaw`** — ZapLine (adaptive, `n_harmonics=2`) on the continuous raw, cached
  and **shared** by every downstream prep (runs once/subject — the scale smoke showed
  ZapLine was ~77% of prep compute and had been recomputed per prep).
- **`Ep_AlphaEnvelopeDFA`** (`alpha_envelope_dfa` custom node) → `alphaEnvelopeDfa`:
  from `DenoisedRaw`, band-pass 8–12 Hz → Hilbert amplitude envelope → downsample to
  100 Hz → crop first **240 s** (2 s edge pad) → DFA with fit range **[1 s, 24 s]**
  (= len/10). One exponent per channel.

The 30 s broadband `detrendedFluctuationMeanEpochs` is kept for r2c parity.

> **NEEDS VALIDATION:** alpha band, fit range, envelope downsample, and edge pad are
> principled defaults, not yet tuned against a reference (e.g. NBT).

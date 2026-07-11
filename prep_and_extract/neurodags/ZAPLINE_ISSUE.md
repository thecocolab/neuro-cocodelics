# ZapLine under-removal on CTF MEG — help wanted

**TL;DR.** Using `mne-denoise` ZapLine(-plus) to remove a strong 50 Hz mains line from
CTF 275-channel MEG, we only get a **~40 % reduction** of the 50 Hz peak (peak/baseline
~13–14× → ~8×), leaving a clearly visible residual line, even though ZapLine reports it
removed hundreds of components. Sweeping the adaptive parameters changes **nothing**, and
non-adaptive (fixed `n_remove`) is **worse**. Is ~40 % expected for this data, or is our
usage suboptimal? How do we get more complete line removal while keeping ZapLine's
signal-preserving advantage over a notch?

---

## Data
- **CTF 275-channel MEG**, ~271–273 MEG data sensors after picking (names `M[LRZ][A-Z]…`),
  **sfreq 1200 Hz**, magnetometer-scale (~1e-13 T).
- Resting / task recordings, 5 drug datasets (LSD, ketamine, perampanel, psilocybin,
  tiagabine); 4 UK/EU sites → **50 Hz mains** with a strong **100 Hz** harmonic, nothing at 60.
- The line is prominent: 50 Hz peak/baseline ≈ **13–21×** in the mean MEG PSD (raw).

## What we run
`mne_denoise.zapline.ZapLine`, applied to the **continuous** MEG-picked raw (before
band-pass / epoching):

```python
from mne_denoise.zapline import ZapLine
data = raw.get_data()                       # (n_ch, n_times), Tesla-scale ~1e-13
scale = float(np.std(data)) or 1.0          # see "scaling" note below
est = ZapLine(sfreq=1200.0, line_freq=50.0, n_harmonics=2, adaptive=True)
cleaned = est.fit_transform(data / scale) * scale
```

- **Scaling note:** at Tesla scale the covariance (~1e-26) underflows ZapLine's reg (1e-9)
  → *"Covariance matrix has no significant variance"* and it removes nothing. We normalise
  by `std` in and multiply back out (ZapLine is linear, so this is exact). With scaling it
  runs and removes components — but only achieves ~40 %.

## Evidence

**Per-recording (matched 2 s Welch, mean over MEG channels), 50 Hz peak/baseline:**

| dataset | raw | post-ZapLine | reduction | components removed (Σ over adaptive chunks) |
|---|---:|---:|---:|---:|
| LSD (sub-S3LR) | 13.8 | 8.0 | 42 % | 264 |
| ketamine | 13.2 | 8.2 | 38 % | 333 |

**Parameter sweep (perampanel & ketamine, 1 file each).** `r50`/`r100` = 50/100 Hz
peak/baseline of the cleaned continuous data; `keep` = median cleaned/raw PSD off the
line bands (signal preservation).

```
perampanel   raw r50=5.5  r100=3.1
 config                              n_rem   r50   r100   keep
 adapt default (sigma3,min1,prop.2)   335   3.67   2.29  0.992
 adapt sigma2                         335   3.67   2.29  0.992
 adapt sigma1.5                       335   3.67   2.29  0.992
 adapt sigma2,min3,prop.3             335   3.67   2.29  0.992
 adapt sigma1.5,min5,prop.5           335   3.67   2.29  0.992   <- adaptive_params: NO effect
 std auto thr3                         12   4.48   2.39  0.980
 std n_remove=3                         3   4.54   2.55  0.998
 std n_remove=5                         5   4.45   2.39  0.997
 std n_remove=8                         8   4.47   2.40  0.987   <- fixed mode: WORSE than adaptive

ketamine     raw r50=8.1  r100=2.3
 adapt (all variants)                 379   4.82   1.48  0.982
 std n_remove=3..8                    3-10   6.8+   1.5+  ~0.99
```

Observations:
1. **Adaptive beats fixed** on the fundamental, but only reaches ~40 % reduction.
2. **`adaptive_params` (`n_remove_params`: sigma / min_remove / max_prop) have no measurable
   effect** — `n_removed_` and the reduction are identical across sigma 3 / 2 / 1.5, etc.
   Adaptive seems to self-regulate to the same result regardless of these hints.
3. Signal is well preserved (`keep` ≈ 0.99), so it's not over-removing.
4. `n_removed_` is large (264–379) but that is the **sum across adaptive time-chunks**
   (~1–3 per chunk × ~150 chunks), not spatial components of 271.

## Questions for a ZapLine expert
- Is ~40 % line reduction **expected** for ZapLine-plus on a line this strong, or does it
  point at a usage error (the `std` scaling? `line_freq`/`n_harmonics`? `nfft`, `nkeep`,
  `nremove` per chunk)?
- Why do `adaptive_params` (sigma/min_remove/max_prop) have no effect on `n_removed_`?
- What settings would push the fundamental down toward baseline **without** gouging the band
  (the reason we chose ZapLine over a notch in the first place)?

## Reproducer (this folder)
- `diagnose_zapline.py` — loads a PickedRaw, reports raw→cleaned 50 Hz peak/baseline +
  `n_removed_`.
- `sweep_zapline.py` — the adaptive/fixed parameter sweep that produced the table above.
- `subset/` — a few **short (120 s) MEG-only** cropped recordings (one per site/dataset) as
  `*_PickedRaw120s.fif`: the exact continuous input we feed ZapLine. Load with
  `mne.io.read_raw_fif(...)` and run the scripts.

Env: `mne-denoise` (ZapLine), `mne`, `numpy`, `scipy`. Our neurodags node wrapping this is
`custom_nodes.py::zapline_denoise`.

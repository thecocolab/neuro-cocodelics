"""Diagnose ZapLine under-removal: sweep the removal-strength knobs on real MEG.

In ADAPTIVE mode, top-level n_remove/threshold are ignored; the knobs live in
adaptive_params['n_remove_params'] = {sigma, min_remove, max_prop}. In STANDARD
mode (adaptive=False), n_remove (int|'auto') + threshold control it. We sweep the
right knobs per mode and report, per config:
  n_removed_   : components ZapLine actually projected out (0 => did nothing)
  r50 / r100   : 50/100 Hz peak/baseline ratio on the CLEANED continuous data
                 (1.0 = flat/clean; higher = residual line)
  keep         : median cleaned/original PSD over non-line bands (~1.0 = signal
                 preserved; <<1 = gouging neural signal)
Run on the worst-case (perampanel) + a typical (ketamine) PickedRaw (continuous,
MEG-picked, pre-resample; already cached).
"""
import glob, re
import numpy as np
from scipy.signal import welch
import mne
from mne_denoise.zapline import ZapLine

mne.set_log_level("error")
DERIV = "/home/yorguin/projects/rrg-kjerbi/shared/neuro-cocodelics/derivatives_neurodags"
FILES = {
    "perampanel": sorted(glob.glob(f"{DERIV}/MEG_perampanel/**/*PickedRaw.fif", recursive=True)),
    "ketamine":   sorted(glob.glob(f"{DERIV}/MEG_ketamine/**/*PickedRaw.fif", recursive=True)),
}

# (label, kwargs to ZapLine beyond sfreq/line_freq/n_harmonics)
CONFIGS = [
    ("adapt default (sigma3,min1,prop.2)", dict(adaptive=True)),
    ("adapt sigma2",                        dict(adaptive=True, adaptive_params={"n_remove_params": {"sigma": 2.0}})),
    ("adapt sigma1.5",                      dict(adaptive=True, adaptive_params={"n_remove_params": {"sigma": 1.5}})),
    ("adapt sigma2,min3,prop.3",            dict(adaptive=True, adaptive_params={"n_remove_params": {"sigma": 2.0, "min_remove": 3, "max_prop": 0.3}})),
    ("adapt sigma1.5,min5,prop.5",          dict(adaptive=True, adaptive_params={"n_remove_params": {"sigma": 1.5, "min_remove": 5, "max_prop": 0.5}})),
    ("std auto thr3",                        dict(adaptive=False, n_remove="auto", threshold=3.0)),
    ("std n_remove=3",                       dict(adaptive=False, n_remove=3)),
    ("std n_remove=5",                       dict(adaptive=False, n_remove=5)),
    ("std n_remove=8",                       dict(adaptive=False, n_remove=8)),
]


def metrics(orig, clean, sf):
    nps = int(sf * 2)  # ~0.5 Hz resolution
    f, po = welch(orig, fs=sf, nperseg=nps, axis=-1); po = po.mean(0)
    _, pc = welch(clean, fs=sf, nperseg=nps, axis=-1); pc = pc.mean(0)

    def ratio(p, f0):
        peak = p[(f >= f0 - 1) & (f <= f0 + 1)].max()
        base = np.median(p[((f >= f0 - 5) & (f <= f0 - 2)) | ((f >= f0 + 2) & (f <= f0 + 5))])
        return float(peak / base)
    # preservation: bands away from 50/100 (and edges)
    keepmask = ((f >= 8) & (f <= 45)) | ((f >= 55) & (f <= 95)) | ((f >= 105) & (f <= 118))
    keep = float(np.median(pc[keepmask] / po[keepmask]))
    return ratio(pc, 50), ratio(pc, 100), keep


for ds, fs in FILES.items():
    if not fs:
        print(f"\n### {ds}: NO PickedRaw"); continue
    raw = mne.io.read_raw_fif(fs[0], preload=True, verbose="error")
    picks = [c for c in raw.ch_names if re.match(r"^M[LRZ][A-Z]", c)]
    raw.pick(picks)
    data = raw.get_data()
    sf = float(raw.info["sfreq"])
    scale = float(np.std(data)) or 1.0
    d = data / scale
    r50o, r100o, _ = metrics(data, data, sf)
    print(f"\n### {ds}  ({len(picks)} ch, sf {sf:g})   RAW r50={r50o:.1f} r100={r100o:.1f}")
    print(f"{'config':36} {'n_rem':>6} {'r50':>7} {'r100':>7} {'keep':>6}")
    for label, kw in CONFIGS:
        try:
            est = ZapLine(sfreq=sf, line_freq=50.0, n_harmonics=2, **kw)
            clean = est.fit_transform(d) * scale
            nrem = est.n_removed_
            if kw.get("adaptive") and getattr(est, "adaptive_results_", None):
                nrem = est.adaptive_results_.get("n_removed", nrem)
            r50, r100, keep = metrics(data, clean, sf)
            print(f"{label:36} {str(nrem):>6} {r50:7.2f} {r100:7.2f} {keep:6.3f}")
        except Exception as e:
            print(f"{label:36} ERR: {type(e).__name__}: {str(e)[:60]}")

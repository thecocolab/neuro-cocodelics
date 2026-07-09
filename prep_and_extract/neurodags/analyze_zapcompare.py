"""Quantify ZapLine Fixed-vs-Adaptive line removal + feature divergence."""
import glob, os, re
import numpy as np
import mne
import pandas as pd

mne.set_log_level("error")
DERIV = "/home/yorguin/projects/rrg-kjerbi/shared/neuro-cocodelics/derivatives_neurodags"
CSV = "/scratch/yorguin/cocodelics_neurodags/outputs/feats_zapcompare_wide.csv"
DATASETS = ["ketamine", "perampanel", "psilocybin", "tiagabine", "LSD"]


def line_ratio(fif, f0):
    ep = mne.read_epochs(fif, preload=True, verbose="error")
    picks = [c for c in ep.ch_names if re.match(r"^M[LRZ][A-Z]", c)]
    psd = ep.compute_psd(method="welch", fmin=1, fmax=120, picks=picks, verbose="error")
    p, f = psd.get_data(return_freqs=True)
    p = p.mean(axis=(0, 1))
    peak = p[(f >= f0 - 1) & (f <= f0 + 1)].max()
    base = np.median(p[((f >= f0 - 5) & (f <= f0 - 2)) | ((f >= f0 + 2) & (f <= f0 + 5))])
    return float(peak / base)


print("=== line-noise residual: peak/baseline ratio (1.0 = fully removed; higher = residual) ===")
hdr = f"{'dataset':12} {'50Fixed':>9} {'50Adapt':>9} {'100Fixed':>9} {'100Adapt':>9}   winner"
print(hdr)
rows = []
for d in DATASETS:
    v = {}
    for f0 in (50, 100):
        for V in ("Fixed", "Adaptive"):
            fs = sorted(glob.glob(f"{DERIV}/MEG_{d}/**/*PrepDur30Ov20{V}.fif", recursive=True))
            v[(f0, V)] = line_ratio(fs[0], f0) if fs else None

    def g(f0, V):
        return v[(f0, V)]

    def fmt(x):
        return f"{x:9.2f}" if isinstance(x, float) else f"{str(x):>9}"
    # winner = lower combined 50+100 residual
    fx = (g(50, "Fixed") or 9e9) + (g(100, "Fixed") or 9e9)
    ad = (g(50, "Adaptive") or 9e9) + (g(100, "Adaptive") or 9e9)
    win = "Fixed" if fx < ad else "Adaptive"
    print(f"{d:12} {fmt(g(50,'Fixed'))} {fmt(g(50,'Adaptive'))} {fmt(g(100,'Fixed'))} {fmt(g(100,'Adaptive'))}   {win}")
    rows.append((d, fx, ad, win))

print("\n=== feature-level Fixed vs Adaptive ===")
if os.path.exists(CSV):
    df = pd.read_csv(CSV)
    num = df.select_dtypes("number")
    nfix = sum("Fixed" in c for c in num.columns)
    nad = sum("Adaptive" in c for c in num.columns)
    print(f"CSV shape={df.shape} numeric={num.shape[1]} NaN_frac={num.isna().mean().mean():.4f} "
          f"FixedCols={nfix} AdaptiveCols={nad}")
    # pair Fixed vs Adaptive columns by stripping the variant tag, compute median rel diff
    def strip(c):
        return c.replace("Fixed", "§").replace("Adaptive", "§")
    pairs = {}
    for c in num.columns:
        if "Fixed" in c or "Adaptive" in c:
            pairs.setdefault(strip(c), {})["Fixed" if "Fixed" in c else "Adaptive"] = c
    diffs = []
    for base, cols in pairs.items():
        if "Fixed" in cols and "Adaptive" in cols:
            a, b = num[cols["Fixed"]], num[cols["Adaptive"]]
            rel = (a - b).abs() / (b.abs() + 1e-12)
            diffs.append(np.nanmedian(rel))
    diffs = np.array(diffs)
    print(f"paired feature cols={len(diffs)}  median |Fixed-Adaptive|/|Adaptive|: "
          f"median={np.nanmedian(diffs):.4f}  max={np.nanmax(diffs):.4f}")
    print("(near 0 -> denoising choice barely changes features; larger -> it matters)")
else:
    print("MISSING CSV:", CSV)

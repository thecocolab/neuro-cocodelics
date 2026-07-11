"""Report #2: per-sensor drug-vs-placebo decoding topomaps.

For each dataset and each MEG sensor, decode drug (1) vs placebo (0) using ONLY that
sensor's complexity features (subject-grouped OOF ROC-AUC via logistic regression). The
resulting per-sensor AUC is plotted on the CTF sensor layout -> a topographic map of
*where* the drug signal lives. Complements the whole-brain decoding in report #1.

Run:  ~/code/coco-pipe/.venv/bin/python make_topomap_report.py
"""
from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

mne.set_log_level("error")
ROOT = Path(__file__).resolve().parent.parent
AGG = ROOT / "prep_and_extract/neurodags/outputs/aggregate_notch_raw.csv"
INFO_FIF = ROOT / "cocodelics/info_for_plot_topomap_function.fif"
OUT = Path(__file__).resolve().parent / "cocodelics_topomaps.html"

DATASETS = [("lsd-avg", "LSD (avg of 6 tasks)"), ("tiagabine", "tiagabine"),
            ("ketamine", "ketamine"), ("perampanel", "perampanel"), ("psilocybin", "psilocybin")]


def per_sensor_auc(sub, feat, sensors_by):
    """OOF ROC-AUC decoding drug-vs-placebo from each sensor's features alone."""
    y = (sub["session"] != "placebo").astype(int).to_numpy()
    groups = sub["subject"].to_numpy()
    cv = GroupKFold(n_splits=int(min(5, len(np.unique(groups)))))
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    out = {}
    for s, cols in sensors_by.items():
        good = sub[cols].dropna(axis=1).columns.tolist()
        if not good:
            continue
        X = sub[good].to_numpy(float)
        try:
            proba = cross_val_predict(pipe, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
            out[s] = roc_auc_score(y, proba)
        except Exception:
            continue
    return out


def topomap_png(auc: dict, info, title) -> str:
    chans = [c for c in info.ch_names if c in auc]
    data = np.array([auc[c] for c in chans])
    sub_info = mne.pick_info(info, mne.pick_channels(info.ch_names, include=chans, ordered=True))
    lo, hi = np.percentile(data, 5), np.percentile(data, 95)   # per-dataset scale -> show variation
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    im, _ = mne.viz.plot_topomap(data, sub_info, axes=ax, show=False, cmap="Reds",
                                 vlim=(lo, hi), contours=4, sensors=True)
    cbar = fig.colorbar(im, ax=ax, shrink=0.66, aspect=12, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("per-sensor AUC", fontsize=7)
    ax.set_title(title, fontsize=10)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build():
    df = pd.read_csv(AGG)
    feat = [c for c in df.columns if c.startswith("feature-") and ".spaces-" in c]
    parsed = {c: re.match(r"feature-(.+)\.spaces-(.+)", c).groups() for c in feat}
    info = mne.io.read_raw_fif(INFO_FIF, preload=False).info
    info.rename_channels(lambda x: x.replace("-3305", ""))
    sensors = sorted({s for _, s in parsed.values()})
    sensors_by = {s: [c for c in feat if parsed[c][1] == s] for s in sensors}

    cards = []
    for ds, label in DATASETS:
        sub = df[df.dataset == ds] if ds != "lsd-avg" else \
            df[df.dataset == "lsd"].groupby(["subject", "session"], as_index=False)[feat].mean()
        auc = per_sensor_auc(sub, feat, sensors_by)
        vals = np.array(list(auc.values()))
        png = topomap_png(auc, info, label)
        cards.append((label, png, float(np.median(vals)), np.mean(vals >= 0.6), len(auc)))
        print(f"[{ds}] sensors={len(auc)} median_auc={np.median(vals):.3f} "
              f"frac>=0.6={np.mean(vals>=0.6):.2f}", flush=True)

    tiles = "".join(
        f'<figure class=card><img alt="{label} topomap" '
        f'src="data:image/png;base64,{png}"/>'
        f'<figcaption>{label}<br><span class=sub>median sensor AUC {med:.2f} · '
        f'{fr:.0%} of {n} sensors ≥0.60</span></figcaption></figure>'
        for label, png, med, fr, n in cards)

    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>cocodelics — per-sensor decoding topomaps</title>
<style>
  :root{{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--border:rgba(11,11,11,.10)}}
  @media (prefers-color-scheme:dark){{:root{{--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--border:rgba(255,255,255,.12)}}}}
  html{{color-scheme:light dark}}
  body{{margin:0;background:var(--page);color:var(--ink);line-height:1.55;
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
  .wrap{{max-width:900px;margin:0 auto;padding:38px 22px 64px}}
  h1{{font-size:23px;margin:0 0 6px}} h2{{font-size:17px;margin:30px 0 8px;border-top:1px solid var(--border);padding-top:16px}}
  p{{font-size:14.5px;color:var(--ink2)}}
  .grid{{display:flex;flex-wrap:wrap;gap:14px;justify-content:center;margin:18px 0}}
  .card{{margin:0;background:var(--surface);border:1px solid var(--border);border-radius:12px;
    padding:10px;text-align:center;width:230px}}
  .card img{{width:100%;height:auto}}
  figcaption{{font-size:13px;font-weight:600;margin-top:4px}}
  .sub{{font-weight:400;color:var(--muted);font-size:11px}}
  .note{{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--muted);
    border-radius:8px;padding:11px 14px;font-size:13px;color:var(--ink2);margin:14px 0}}
  code{{background:var(--border);padding:1px 5px;border-radius:4px;font-size:12.5px}}
</style></head><body><div class=wrap>
<h1>Per-sensor decoding topomaps — where the drug signal lives</h1>
<p>Companion to the main report. For each dataset and each MEG sensor, drug-vs-placebo is decoded
using <b>only that sensor's complexity features</b> (14 measures), with subject-grouped
cross-validation. The map shows the resulting per-sensor ROC-AUC on the CTF layout — warmer =
that sensor alone separates drug from placebo better (0.5 = chance).</p>

<h2>Datasets</h2>
<div class=grid>{tiles}</div>

<div class=note>
<b>Reading these.</b> Single-sensor AUC is necessarily lower than the whole-brain decoding in the
main report (one sensor × 14 features vs all 3,822) — the value here is the <b>spatial pattern</b>,
not the absolute height. Warm regions indicate where the drug-vs-placebo complexity differences are
strongest. Colour scale 0.35–0.65 centred on chance (0.5); the 2 sensors without layout positions
(incl. the dead <code>MRT36</code>) are omitted. Per-sensor AUCs are single-model (logistic
regression), not permutation-tested individually — treat as descriptive attribution supporting the
significant whole-brain effects.
</div>
</div></body></html>
"""


if __name__ == "__main__":
    OUT.write_text(build())
    print("wrote", OUT)

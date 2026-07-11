"""Report #3: feature-level attribution.

(A) Per-feature decoding — for each of the 14 complexity measures, decode drug-vs-placebo
    using that measure across all sensors (subject-grouped OOF ROC-AUC). A feature x dataset
    heatmap answers "which measure carries each drug's signal".
(B) Per-feature effect topomaps — for an exemplar dataset (LSD, task-avg), the paired
    (drug - placebo) t-value per sensor for each measure, on the CTF layout: the spatial
    signature of each complexity measure.

Run:  ~/code/coco-pipe/.venv/bin/python make_feature_report.py
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
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

mne.set_log_level("error")
ROOT = Path(__file__).resolve().parent.parent
AGG = ROOT / "prep_and_extract/neurodags/outputs/aggregate_notch_raw.csv"
INFO_FIF = ROOT / "cocodelics/info_for_plot_topomap_function.fif"
OUT = Path(__file__).resolve().parent / "cocodelics_features_report.html"

DATASETS = [("lsd-avg", "LSD (avg)"), ("tiagabine", "tiagabine"), ("ketamine", "ketamine"),
            ("perampanel", "perampanel"), ("psilocybin", "psilocybin")]
NICE = {"lzivComplexityMeanEpochs": "Lempel-Ziv", "higuchiFdMeanEpochs": "Higuchi FD",
        "higuchiFdVarEpochs": "Higuchi FD (var)", "katzFdMeanEpochs": "Katz FD",
        "katzFdSDEpochs": "Katz FD (SD)", "petrosianFdMeanEpochs": "Petrosian FD",
        "svdEntropyMeanEpochs": "SVD entropy", "numZerocrossMeanEpochs": "zero-crossings",
        "permEntropyMeanEpochs": "perm. entropy", "spectralEntropyMeanEpochs": "spectral entropy",
        "hjorthMobilityMeanEpochs": "Hjorth mobility", "hjorthComplexityMeanEpochs": "Hjorth complexity",
        "detrendedFluctuationMeanEpochs": "DFA (30 s)", "alphaEnvelopeDfa": "alpha-env DFA"}


def subset(df, ds, feat):
    if ds == "lsd-avg":
        return df[df.dataset == "lsd"].groupby(["subject", "session"], as_index=False)[feat].mean()
    return df[df.dataset == ds]


def main():
    df = pd.read_csv(AGG)
    feat = [c for c in df.columns if c.startswith("feature-") and ".spaces-" in c]
    parsed = {c: re.match(r"feature-(.+)\.spaces-(.+)", c).groups() for c in feat}
    features = [f for f in NICE if any(parsed[c][0] == f for c in feat)]   # 14, ordered
    cols_by_feat = {f: [c for c in feat if parsed[c][0] == f] for f in features}
    info = mne.io.read_raw_fif(INFO_FIF, preload=False).info
    info.rename_channels(lambda x: x.replace("-3305", ""))

    # ---- (A) per-feature decoding AUC (feature x dataset) ----
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    aucs = {}
    for ds, _ in DATASETS:
        sub = subset(df, ds, feat)
        y = (sub["session"] != "placebo").astype(int).to_numpy()
        g = sub["subject"].to_numpy()
        cv = GroupKFold(n_splits=int(min(5, len(np.unique(g)))))
        for f in features:
            good = sub[cols_by_feat[f]].dropna(axis=1).columns.tolist()
            if not good:
                aucs[(f, ds)] = np.nan; continue
            proba = cross_val_predict(pipe, sub[good].to_numpy(float), y, groups=g, cv=cv,
                                      method="predict_proba")[:, 1]
            aucs[(f, ds)] = roc_auc_score(y, proba)
        print(f"[A {ds}] done", flush=True)

    # heatmap (colored HTML table); color scale 0.5-0.9
    def cell(v):
        if np.isnan(v):
            return '<td class="c na">–</td>'
        t = max(0.0, min(1.0, (v - 0.5) / 0.4))    # 0.5->0, 0.9->1
        return f'<td class="c" style="--i:{t:.2f}">{v:.2f}</td>'
    hdr = "".join(f"<th>{lab}</th>" for _, lab in DATASETS)
    hrows = []
    # order features by mean AUC across datasets (most discriminative on top)
    fmean = {f: np.nanmean([aucs[(f, ds)] for ds, _ in DATASETS]) for f in features}
    for f in sorted(features, key=lambda x: -fmean[x]):
        cells = "".join(cell(aucs[(f, ds)]) for ds, _ in DATASETS)
        hrows.append(f"<tr><th class=rl>{NICE[f]}</th>{cells}</tr>")
    heat = (f"<table class=heat><thead><tr><th></th>{hdr}</tr></thead>"
            f"<tbody>{''.join(hrows)}</tbody></table>")

    # ---- (B) per-feature paired-t topomaps for LSD (avg) ----
    lsd = subset(df, "lsd-avg", feat)
    piv = {s: lsd[lsd.session == s].set_index("subject") for s in ["placebo", "lsd"]}
    common = piv["placebo"].index.intersection(piv["lsd"].index)
    chans = info.ch_names
    panels = []
    tmax_all = 0.0
    tmaps = {}
    for f in features:
        by_sensor = {parsed[c][1]: c for c in cols_by_feat[f]}
        present = [s for s in chans if s in by_sensor]
        pl = piv["placebo"].loc[common, [by_sensor[s] for s in present]].to_numpy(float)
        dr = piv["lsd"].loc[common, [by_sensor[s] for s in present]].to_numpy(float)
        t, _ = ttest_rel(dr, pl, axis=0, nan_policy="omit")
        tmaps[f] = (present, np.asarray(t))
        tmax_all = max(tmax_all, np.nanmax(np.abs(t)))
    vlim = float(np.ceil(tmax_all))
    for f in features:
        present, t = tmaps[f]
        sub_info = mne.pick_info(info, mne.pick_channels(chans, include=present, ordered=True))
        fig, ax = plt.subplots(figsize=(2.2, 2.2))
        mne.viz.plot_topomap(t, sub_info, axes=ax, show=False, cmap="RdBu_r",
                             vlim=(-vlim, vlim), contours=3, sensors=False)
        ax.set_title(NICE[f], fontsize=9)
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", transparent=True)
        plt.close(fig)
        panels.append(f'<figure class=tp><img src="data:image/png;base64,'
                      f'{base64.b64encode(buf.getvalue()).decode()}"/></figure>')
        print(f"[B {f}] done", flush=True)

    OUT.write_text(f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>cocodelics — feature-level attribution</title>
<style>
 :root{{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
  --border:rgba(11,11,11,.10);--seq:#256abf}}
 @media (prefers-color-scheme:dark){{:root{{--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
  --border:rgba(255,255,255,.12);--seq:#3987e5}}}}
 html{{color-scheme:light dark}}
 body{{margin:0;background:var(--page);color:var(--ink);line-height:1.55;
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
 .wrap{{max-width:900px;margin:0 auto;padding:38px 22px 64px}}
 h1{{font-size:23px;margin:0 0 6px}} h2{{font-size:17px;margin:32px 0 8px;border-top:1px solid var(--border);padding-top:16px}}
 p{{font-size:14.5px;color:var(--ink2)}}
 table.heat{{border-collapse:collapse;font-size:13px;margin:12px 0}}
 .heat th{{color:var(--ink2);font-weight:600;font-size:12px;padding:5px 9px}}
 .heat th.rl{{text-align:right;font-weight:600;color:var(--ink)}}
 td.c{{width:74px;height:26px;text-align:center;font-variant-numeric:tabular-nums;
  border:2px solid var(--surface);border-radius:5px;
  background:color-mix(in srgb, var(--seq) calc(var(--i)*85%), var(--surface));
  color:var(--ink)}}
 td.c.na{{background:var(--surface);color:var(--muted)}}
 .grid{{display:flex;flex-wrap:wrap;gap:4px;justify-content:center;margin:12px 0}}
 .tp{{margin:0;width:150px}} .tp img{{width:100%}}
 .cbar{{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--ink2);justify-content:center;margin-top:6px}}
 .cbar i{{display:inline-block;width:120px;height:11px;border-radius:3px;
  background:linear-gradient(90deg,#2166ac,#f7f7f7,#b2182b)}}
 .note{{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--muted);
  border-radius:8px;padding:11px 14px;font-size:13px;color:var(--ink2);margin:14px 0}}
</style></head><body><div class=wrap>
<h1>Feature-level attribution — which complexity measures &amp; where</h1>
<p>Third companion report. (A) which of the 14 complexity measures decode each drug; (B) the
spatial signature of each measure for LSD.</p>

<h2>A. Per-feature decoding — ROC-AUC by measure × drug</h2>
<p>Each cell: drug-vs-placebo decoded from a single measure across all sensors (subject-grouped
OOF ROC-AUC, logistic regression). Rows ordered by mean AUC across drugs.</p>
{heat}
<div class=note>Warmer = that measure alone separates drug from placebo better (0.5 = chance).
Reading down a column shows which measures a given drug loads on; across a row, how broadly a
measure generalises across drugs. Descriptive (single-model, not permutation-tested per cell).</div>

<h2>B. Spatial signature — paired (drug − placebo) t-value per sensor, LSD</h2>
<p>For LSD (task-average, 19 paired subjects), the per-sensor paired t-statistic of drug vs placebo
for each measure. Red = higher on drug, blue = higher on placebo. Shared scale ±{vlim:.0f}.</p>
<div class=grid>{''.join(panels)}</div>
<div class=cbar><span>placebo &gt;</span><i></i><span>&gt; drug</span> &nbsp; (t, ±{vlim:.0f})</div>
<div class=note>Paired t across subjects — uncorrected for multiple comparisons (271 sensors × 14
measures); read as the spatial/measure pattern, not per-sensor significance. LSD shown as the
strongest exemplar; other datasets available on request.</div>
</div></body></html>
""")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

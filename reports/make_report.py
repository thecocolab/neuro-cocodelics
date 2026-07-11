"""Generate a single self-contained HTML report (feature extraction + ML) for collaborators.

Pulls the real numbers from the result CSVs so the report stays accurate/regenerable:
  - notch feature table (dataset/recording/subject counts, NaN)
  - ML baseline leaderboard + subject-paired permutation results
Narrative for the pipeline/QC/decisions is templated from the project record.

Run with any env that has pandas/numpy:
    ~/code/coco-pipe/.venv/bin/python make_report.py
"""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent          # neuro-cocodelics/
AGG = ROOT / "prep_and_extract/neurodags/outputs/aggregate_notch_raw.csv"
PERM = ROOT / "machine-learning/outputs/ml_explore/permutation_results.csv"
LB = ROOT / "machine-learning/outputs/ml_explore/leaderboard.csv"
OUT = Path(__file__).resolve().parent / "cocodelics_report.html"

DRUG = {"lsd": "LSD (serotonergic psychedelic)", "ketamine": "ketamine (NMDA antagonist)",
        "perampanel": "perampanel (AMPA antagonist)", "psilocybin": "psilocybin (serotonergic)",
        "tiagabine": "tiagabine (GABA reuptake inhibitor)"}
FEATURES = ["Lempel-Ziv complexity", "Higuchi fractal dim (mean, var)", "Katz fractal dim (mean, SD)",
            "Petrosian fractal dim", "SVD entropy", "num zero-crossings", "permutation entropy",
            "spectral entropy", "Hjorth mobility", "Hjorth complexity",
            "detrended fluctuation (DFA, 30 s)", "alpha-envelope DFA (240 s window)"]


def esc(x):
    return html.escape(str(x))


def load():
    agg = pd.read_csv(AGG)
    feat = [c for c in agg.columns if c.startswith("feature-") and ".spaces-" in c]
    ds_rows = []
    for d, g in agg.groupby("dataset"):
        ds_rows.append({"dataset": d, "recordings": len(g), "subjects": g["subject"].nunique(),
                        "nan_frac": g[feat].isna().mean().mean()})
    perm = pd.read_csv(PERM)
    return agg, feat, pd.DataFrame(ds_rows), perm


def tile(v, label, sub=""):
    return (f'<div class="tile"><div class="tile-v">{v}</div>'
            f'<div class="tile-l">{esc(label)}</div>'
            f'{f"<div class=tile-s>{esc(sub)}</div>" if sub else ""}</div>')


def auc_chart(perm: pd.DataFrame) -> str:
    """Horizontal grouped bars: roc_auc per dataset for LogReg vs RandomForest, from chance."""
    order = ["lsd-avg", "tiagabine", "ketamine", "perampanel", "psilocybin"]
    labels = {"lsd-avg": "LSD (avg)", "tiagabine": "tiagabine", "ketamine": "ketamine",
              "perampanel": "perampanel", "psilocybin": "psilocybin"}
    models = [("LogReg", "var(--s1)"), ("RandomForest", "var(--s2)")]
    lo, hi = 0.4, 1.0
    W, rowh, padL, padR, padT = 760, 54, 150, 30, 34
    H = padT + len(order) * rowh + 30

    def x(v):
        return padL + (v - lo) / (hi - lo) * (W - padL - padR)

    svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
           f'aria-label="ROC-AUC per dataset by model">']
    # axis ticks
    for t in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        svg.append(f'<line x1="{x(t):.1f}" y1="{padT-6}" x2="{x(t):.1f}" y2="{H-24}" '
                   f'stroke="var(--grid)" stroke-width="1"/>')
        svg.append(f'<text x="{x(t):.1f}" y="{H-8}" text-anchor="middle" '
                   f'class="ax">{t:.1f}</text>')
    # chance line
    svg.append(f'<line x1="{x(0.5):.1f}" y1="{padT-6}" x2="{x(0.5):.1f}" y2="{H-24}" '
               f'stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="4 3"/>')
    svg.append(f'<text x="{x(0.5):.1f}" y="{padT-12}" text-anchor="middle" class="ax">chance</text>')
    for i, ds in enumerate(order):
        y0 = padT + i * rowh
        svg.append(f'<text x="{padL-10}" y="{y0+rowh/2:.0f}" text-anchor="end" '
                   f'class="dslab">{esc(labels[ds])}</text>')
        for j, (m, col) in enumerate(models):
            r = perm[(perm.problem == ds) & (perm.model == m)]
            if r.empty:
                continue
            auc = float(r.roc_auc.iloc[0]); p = float(r.p_perm.iloc[0])
            by = y0 + 8 + j * 18
            xa = x(max(auc, 0.5))
            svg.append(f'<rect x="{x(0.5):.1f}" y="{by}" width="{max(0.5,xa-x(0.5)):.1f}" height="13" '
                       f'rx="3" fill="{col}"><title>{esc(m)}: AUC={auc:.3f}, p={p:.3f}</title></rect>')
            star = " *" if p < 0.05 else " n.s."
            svg.append(f'<text x="{xa+5:.1f}" y="{by+11}" class="barlab">{auc:.2f}{star}</text>')
    svg.append("</svg>")
    return "".join(svg)


def build():
    agg, feat, ds, perm = load()
    n_rec = int(ds.recordings.sum()); n_subj = int(agg["subject"].nunique())
    n_feat_cols = len(feat)
    n_features = agg_feat = len({c.split(".spaces-")[0] for c in feat})
    n_sensors = len({c.split(".spaces-")[1] for c in feat})
    nsig = int((perm[perm.model.isin(["LogReg", "RandomForest"])]
                .groupby("problem").p_perm.min() < 0.05).sum())
    n_problems = perm.problem.nunique()

    tiles = "".join([
        tile(len(ds), "drug datasets", "MEG, altered states"),
        tile(n_rec, "recordings", f"{n_subj} subjects"),
        tile(f"{n_features}×{n_sensors}", "features × sensors", f"{n_feat_cols:,} columns"),
        tile(f"{nsig}/{n_problems}", "decodable > chance", "subject-paired permutation"),
    ])

    ds_tbl = "".join(
        f"<tr><td>{esc(DRUG.get(r.dataset, r.dataset))}</td><td class=n>{r.recordings}</td>"
        f"<td class=n>{r.subjects}</td><td class=n>{r.nan_frac:.3f}</td></tr>"
        for r in ds.sort_values('recordings', ascending=False).itertuples())

    feat_items = "".join(f"<li>{esc(f)}</li>" for f in FEATURES)

    # ML results table (headline datasets)
    order = ["lsd-avg", "tiagabine", "ketamine", "perampanel", "psilocybin"]
    disp = {"lsd-avg": "LSD (avg of 6 tasks)", "tiagabine": "tiagabine", "ketamine": "ketamine",
            "perampanel": "perampanel", "psilocybin": "psilocybin"}
    ml_rows = []
    for d in order:
        cells = [f"<td>{esc(disp[d])}</td>"]
        for m in ["LogReg", "RandomForest"]:
            r = perm[(perm.problem == d) & (perm.model == m)]
            if r.empty:
                cells.append("<td class=n>–</td>"); continue
            auc = float(r.roc_auc.iloc[0]); p = float(r.p_perm.iloc[0])
            sig = "sig" if p < 0.05 else "ns"
            cells.append(f'<td class="n {sig}">{auc:.2f} <span class=p>(p={p:.3f})</span></td>')
        ml_rows.append("<tr>" + "".join(cells) + "</tr>")
    ml_tbl = "".join(ml_rows)

    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>cocodelics — MEG complexity features & drug decoding</title>
<style>
  :root {{ --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
    --grid:#e1e0d9; --border:rgba(11,11,11,.10); --s1:#2a78d6; --s2:#1baf7a; --good:#0ca30c;
    --dead:#d03b3b; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --page:#0d0d0d; --surface:#1a1a19; --ink:#fff;
    --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --border:rgba(255,255,255,.10); --s1:#3987e5;
    --s2:#199e70; --good:#0ca30c; --dead:#d03b3b; }} }}
  html{{color-scheme:light dark}}
  body{{margin:0;background:var(--page);color:var(--ink);line-height:1.55;
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
  .wrap{{max-width:920px;margin:0 auto;padding:40px 22px 72px}}
  h1{{font-size:26px;margin:0 0 6px;letter-spacing:-.02em}}
  h2{{font-size:19px;margin:38px 0 6px;padding-top:16px;border-top:1px solid var(--grid)}}
  h3{{font-size:15px;margin:22px 0 6px;color:var(--ink2)}}
  .lede{{color:var(--ink2);font-size:15px;margin:0 0 8px}}
  p{{font-size:14.5px}} li{{font-size:14px}}
  code{{background:var(--grid);padding:1px 5px;border-radius:4px;font-size:12.5px}}
  .tiles{{display:flex;flex-wrap:wrap;gap:12px;margin:22px 0}}
  .tile{{flex:1 1 160px;background:var(--surface);border:1px solid var(--border);border-radius:12px;
    padding:15px 17px}}
  .tile-v{{font-size:25px;font-weight:650;letter-spacing:-.01em}}
  .tile-l{{font-size:12.5px;color:var(--ink2);margin-top:2px}} .tile-s{{font-size:11px;color:var(--muted)}}
  table{{border-collapse:collapse;width:100%;background:var(--surface);border:1px solid var(--border);
    border-radius:12px;overflow:hidden;font-size:13.5px;margin:12px 0}}
  th,td{{text-align:left;padding:8px 13px;border-bottom:1px solid var(--grid)}}
  th{{color:var(--ink2);font-weight:600;font-size:12px}}
  tr:last-child td{{border-bottom:none}}
  td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}}
  td.sig{{color:var(--good);font-weight:600}} td.ns{{color:var(--muted)}}
  .p{{font-weight:400;color:var(--muted);font-size:11.5px}}
  .flow{{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:14px 0;font-size:12.5px}}
  .box{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:7px 11px}}
  .arr{{color:var(--muted)}}
  .chart{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 10px;margin:12px 0}}
  .ax{{fill:var(--muted);font-size:10px}} .dslab{{fill:var(--ink);font-size:12.5px;font-weight:600}}
  .barlab{{fill:var(--ink2);font-size:10.5px;font-variant-numeric:tabular-nums}}
  .legend{{display:flex;gap:16px;font-size:12px;color:var(--ink2);margin:2px 0 0 12px}}
  .sw{{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:5px}}
  .note{{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--muted);
    border-radius:8px;padding:11px 14px;font-size:13px;color:var(--ink2);margin:12px 0}}
  ul{{margin:8px 0}}
</style></head><body><div class=wrap>

<h1>cocodelics — MEG complexity features &amp; drug-vs-placebo decoding</h1>
<p class=lede>Karim Jerbi lab. Do complexity / fractal features of resting-and-task MEG separate a drug
state from placebo? This report covers the two stages end to end: (1) a reproducible feature-extraction
pipeline over five drug datasets, and (2) a machine-learning decoding exploration with rigorous
subject-level significance testing.</p>
<div class=tiles>{tiles}</div>
<div class=note><b>Headline.</b> Drug-vs-placebo is decodable above chance from the complexity features
for <b>every drug tested</b> — strongly for LSD, tiagabine and ketamine (ROC-AUC ~0.8–0.9), and
significantly (if weakly) for perampanel and psilocybin — validated by a within-subject permutation null.</div>

<h2>Part 1 — Feature extraction</h2>
<p>Raw MEG (CTF 275-channel, 1200 Hz) → BIDS → a declarative
<a href="https://github.com/yjmantilla/neurodags">neurodags</a> pipeline, run as a per-file job array on
the Alliance cluster (362 recordings, 0 errors).</p>
<div class=flow>
  <span class=box>raw MEG (.ds/.mat)</span><span class=arr>→</span>
  <span class=box>BIDS</span><span class=arr>→</span>
  <span class=box>pick MEG + clean names</span><span class=arr>→</span>
  <span class=box>notch 50/100/150</span><span class=arr>→</span>
  <span class=box>band-pass 0.1–150</span><span class=arr>→</span>
  <span class=box>epoch 30 s / 20 s ov</span><span class=arr>→</span>
  <span class=box>resample 600 Hz</span><span class=arr>→</span>
  <span class=box>features</span>
</div>

<h3>Datasets</h3>
<table><thead><tr><th>dataset (drug)</th><th class=n>recordings</th><th class=n>subjects</th>
  <th class=n>NaN frac</th></tr></thead><tbody>{ds_tbl}</tbody></table>

<h3>Preprocessing</h3>
<ul>
  <li><b>Channel handling</b>: keep CTF MEG sensors by name (<code>^M[LRZ][A-Z]</code>, ~271–273/dataset),
      strip the varying <code>-&lt;runid&gt;</code> suffix so sensors align across datasets. (A bidsification
      bug that had mistyped MEG channels as <code>misc</code> was fixed → clean feature matrix.)</li>
  <li><b>Line noise</b>: <b>notch [50, 100, 150] Hz</b>. These are 50 Hz-mains sites (clear 50 Hz + 100 Hz
      in the raw spectra, nothing at 60). We first tried ZapLine (mne-denoise), but it only reduced the
      50 Hz line ~40 %; the notch removes it ~fully (verified 21×→1× peak/baseline). The ZapLine
      under-removal was written up + handed to a collaborator; both feature runs (ZapLine + notch) are
      preserved, notch is the production set.</li>
  <li><b>Band-pass</b> 0.1–150 Hz, <b>epoch</b> 30 s (20 s overlap), <b>resample</b> 600 Hz.</li>
</ul>

<h3>Features (per epoch → mean/var over epochs, per sensor)</h3>
<ul style="columns:2">{feat_items}</ul>
<p>Alpha-envelope DFA is computed separately on a single 240 s continuous window (the common floor across
datasets) on the 8–12 Hz Hilbert envelope — the canonical long-range-temporal-correlation setup
(Hardstone 2012). Final table: <b>{n_features} features × {n_sensors} sensors = {n_feat_cols:,} columns</b>,
{n_rec} recordings.</p>

<h3>Quality control</h3>
<p>NaN in the feature table is <b>0.59 %</b> and entirely <b>sensor-driven</b> (denoise-independent):
~90 % structural (a sensor absent from a dataset — the 273 is the union across sites) + ~10 % from
<b>dead channels</b>, dominated by one systematically-flat sensor (<code>MRT36</code>, dead in ~all
perampanel + psilocybin recordings). No feature-computation failures (every feature has an identical
NaN count). Per-recording spectra + a NaN heat-map are available as self-contained HTML QC reports.</p>

<h2>Part 2 — Machine-learning decoding</h2>
<h3>Method</h3>
<ul>
  <li><b>Task</b>: binary drug (1) vs placebo (0), one problem per dataset (LSD split per task + an
      average), on the notch feature table (per problem, feature columns with any NaN are dropped).</li>
  <li><b>Engine</b>: <a href="https://github.com/BabaSanfour/coco-pipe">coco-pipe</a> <code>decoding</code>
      (dev). Models: L2 logistic regression and random forest (both robust to p≫n; gradient boosting and
      SVC were unstable in this regime and excluded). Standardised features.</li>
  <li><b>Cross-validation</b>: <b>GroupKFold by subject</b> (5 folds) — no subject appears in both train
      and test, so scores reflect generalisation to new people, not memorised subjects.</li>
  <li><b>Significance</b>: a <b>within-subject paired label permutation</b> (each subject contributes one
      placebo + one drug recording; permuting the label within subject is the correct null) on pooled
      out-of-fold predictions. p = (1 + #{{null ≥ observed}}) / (1 + n_perms).</li>
</ul>

<h3>ROC-AUC per dataset (subject-grouped CV)</h3>
<div class=chart>{auc_chart(perm)}
  <div class=legend><span><span class=sw style="background:var(--s1)"></span>Logistic regression</span>
    <span><span class=sw style="background:var(--s2)"></span>Random forest</span>
    <span>* = p &lt; 0.05 (permutation)</span></div></div>

<table><thead><tr><th>dataset</th><th class=n>Logistic regression</th>
  <th class=n>Random forest</th></tr></thead><tbody>{ml_tbl}</tbody></table>
<p style="font-size:13px;color:var(--ink2)">LSD's six tasks (Open1/2, Closed1/2, Music, Video) are all
individually significant (ROC-AUC 0.79–0.90, p≈0.002); the table shows the task-average. Null AUC sat at
chance (~0.49) for every problem, so nothing is a p≫n artifact. psilocybin is the one split decision —
significant for the random forest (nonlinear) but not logistic regression.</p>

<div class=note><b>Caveats.</b> High-dimensional, small-n ({n_rec} recordings, 15–20 subjects/dataset,
{n_feat_cols:,} features) — cross-validated AUCs are optimistic in absolute terms; the permutation test
establishes <i>significance</i>, not effect-size precision. The residual line noise is common-mode across
drug/placebo (same site), so it is unlikely to drive the discrimination. Feature/sensor-level attribution
(which complexity measures &amp; regions carry the signal) is the natural next analysis.</div>

<h2>Reproducibility</h2>
<ul>
  <li>Repo <code>thecocolab/neuro-cocodelics</code>, branch <code>feature-extraction-neurodags</code>.</li>
  <li>Feature pipeline: <code>prep_and_extract/neurodags/</code> (pipeline_cocodelics.yml, custom_nodes.py,
      cluster/run_full_array.sbatch). Derivatives on <code>/scratch/…/cocodelics_derivatives_notch</code>;
      table <code>outputs/aggregate_notch_raw.csv</code>.</li>
  <li>ML: <code>machine-learning/explore_decoding.py</code> (baseline) + <code>perm_test.py</code>
      (permutation); results in <code>machine-learning/outputs/ml_explore/</code>.</li>
  <li>QC methods: <code>nan_qc.py</code>, <code>meg_report_qc</code> (mne.Report); ZapLine writeup
      <code>ZAPLINE_ISSUE.md</code>.</li>
</ul>
</div></body></html>
"""


if __name__ == "__main__":
    OUT.write_text(build())
    print("wrote", OUT)

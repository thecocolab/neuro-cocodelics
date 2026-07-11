"""Quality-control analysis of NaNs in a coco-pipe-format feature table.

Given an aggregate CSV with columns ``feature-<feat>.spaces-<sensor>`` (+ id columns
``dataset``/``subject``/``session``/``filepath``), decompose every NaN cell into:

  * STRUCTURAL  -- the sensor is absent from that whole dataset (the sensors are the
    *union* across sites; a dataset recorded with fewer channels leaves those union
    columns all-NaN for all its recordings). Expected, not a data-quality problem.
  * SCATTERED   -- the sensor IS present in the dataset but is NaN in a specific
    recording. Split into:
      - dead channel : ALL features NaN for that (recording, sensor) -> flat/dead sensor.
      - partial      : only SOME features NaN for a present (recording, sensor) ->
                       a feature-specific computation failure (worth investigating).

Emits a self-contained, theme-aware **HTML** report (summary tiles, per-dataset table,
a sensor x dataset NaN heatmap distinguishing structural/dead, dead-channel table,
bad-sensor ranking) and a short Markdown twin. Reusable on any aggregate CSV.

Usage:
    python nan_qc.py [aggregate_csv] [--html report.html] [--md report.md]
    (defaults: outputs/aggregate_full_raw.csv -> NAN_QC_REPORT.{html,md})
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd

FEAT_RE = re.compile(r"feature-(.+)\.spaces-(.+)")


def analyze(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c.startswith("feature-") and ".spaces-" in c]
    if not feat_cols:
        raise SystemExit(f"No 'feature-<f>.spaces-<sensor>' columns in {csv_path}")
    parsed = {c: FEAT_RE.match(c).groups() for c in feat_cols}
    feats = sorted({f for f, _ in parsed.values()})
    sensors = sorted({s for _, s in parsed.values()})
    cols_by_sensor = {s: [c for c in feat_cols if parsed[c][1] == s] for s in sensors}
    n_feat = len(feats)

    total_cells = df.shape[0] * len(feat_cols)
    total_nan = int(df[feat_cols].isna().sum().sum())

    cols_by_feat = {f: [c for c in feat_cols if parsed[c][0] == f] for f in feats}

    datasets = list(df["dataset"].unique())
    recs_per = {d: int((df["dataset"] == d).sum()) for d in datasets}
    present = {}
    per_dataset = []
    for d, g in df.groupby("dataset"):
        pres = {s for s in sensors if g[cols_by_sensor[s]].notna().any().any()}
        present[d] = pres
        sub = g[feat_cols]
        per_dataset.append({
            "dataset": d, "recordings": len(g), "sensors_present": len(pres),
            "nan_frac": float(sub.isna().mean().mean()), "nan_cells": int(sub.isna().sum().sum()),
        })

    structural = 0
    dead = []       # (dataset, subject, session, filename, sensor)
    partial = []    # (dataset, subject, session, sensor, n_feat_nan)
    dead_count = {}     # (dataset, sensor) -> #recordings dead
    sensor_dead = {}    # sensor -> #recordings dead (any dataset)
    for _, row in df.iterrows():
        d = row["dataset"]
        for s in sensors:
            block = row[cols_by_sensor[s]]
            n_nan = int(block.isna().sum())
            if n_nan == 0:
                continue
            if s not in present[d]:
                structural += n_nan
            elif n_nan == n_feat:
                dead.append((d, row.get("subject"), row.get("session"),
                             str(row.get("filepath", "")).split("/")[-1], s))
                dead_count[(d, s)] = dead_count.get((d, s), 0) + 1
                sensor_dead[s] = sensor_dead.get(s, 0) + 1
            else:
                partial.append((d, row.get("subject"), row.get("session"), s, n_nan))
    scattered = len(dead) * n_feat + sum(p[4] for p in partial)

    # ---- feature dimension: is NaN feature-dependent, or purely sensor-driven? ----
    feat_nan = {f: int(df[cols_by_feat[f]].isna().sum().sum()) for f in feats}
    feat_frac = {f: feat_nan[f] / (df.shape[0] * len(cols_by_feat[f])) for f in feats}
    feat_ds = {}   # (feature, dataset) -> NaN cell count
    for d, g in df.groupby("dataset"):
        for f in feats:
            feat_ds[(f, d)] = int(g[cols_by_feat[f]].isna().sum().sum())
    # uniform => every feature has the SAME total NaN => NaN is sensor-driven, not feature-specific
    feat_uniform = len(set(feat_nan.values())) == 1

    # sensors interesting for the heatmap: absent-in-some-dataset OR dead-somewhere
    absent = {d: (set(sensors) - present[d]) for d in datasets}
    interesting = sorted({s for d in datasets for s in absent[d]} | set(sensor_dead))

    # per (interesting sensor, dataset) status for the heatmap
    grid = {}
    for s in interesting:
        for d in datasets:
            if s not in present[d]:
                grid[(s, d)] = ("absent", recs_per[d], recs_per[d])
            elif (d, s) in dead_count:
                grid[(s, d)] = ("dead", dead_count[(d, s)], recs_per[d])
            else:
                grid[(s, d)] = ("live", 0, recs_per[d])

    return dict(
        csv=csv_path, n_recordings=df.shape[0], n_features=n_feat, n_sensors=len(sensors),
        total_cells=total_cells, total_nan=total_nan, structural=structural, scattered=scattered,
        per_dataset=per_dataset, dead=dead, partial=partial, sensor_dead=sensor_dead,
        datasets=datasets, interesting=interesting, grid=grid, recs_per=recs_per, feats=feats,
        feat_nan=feat_nan, feat_frac=feat_frac, feat_ds=feat_ds, feat_uniform=feat_uniform,
    )


# ----------------------------------------------------------------------------- markdown
def to_markdown(r: dict) -> str:
    L = [f"# NaN QC report — neurodags feature table\n",
         f"- source: `{r['csv']}`",
         f"- shape: {r['n_recordings']} recordings x {r['n_features']} features x "
         f"{r['n_sensors']} sensors (union) = {r['total_cells']} cells",
         f"- **total NaN: {r['total_nan']} ({r['total_nan']/r['total_cells']:.2%})** "
         f"= structural {r['structural']} ({r['structural']/max(r['total_nan'],1):.0%}) "
         f"+ scattered {r['scattered']} ({r['scattered']/max(r['total_nan'],1):.0%})", ""]
    L += ["## Per-dataset\n", "| dataset | recordings | sensors present | NaN frac | NaN cells |",
          "|---|---:|---:|---:|---:|"]
    for d in sorted(r["per_dataset"], key=lambda x: -x["nan_cells"]):
        L.append(f"| {d['dataset']} | {d['recordings']} | {d['sensors_present']} | "
                 f"{d['nan_frac']:.4f} | {d['nan_cells']} |")
    L += ["", "## By feature (is NaN feature-dependent?)\n"]
    if r["feat_uniform"]:
        v = next(iter(r["feat_nan"].values()))
        L.append(f"**Feature-independent** — all {r['n_features']} features have an identical NaN "
                 f"count ({v}), so NaN is driven entirely by sensors (absent/dead), not by any "
                 f"feature's computation. No feature is disproportionately failing.\n")
    else:
        L.append("NaN counts DIFFER across features — a feature computes NaN more than others "
                 "(see table); investigate that feature.\n")
    L += ["| feature | NaN cells | NaN frac |", "|---|---:|---:|"]
    for f, n in sorted(r["feat_nan"].items(), key=lambda x: -x[1]):
        L.append(f"| {f} | {n} | {r['feat_frac'][f]:.4f} |")
    L += ["", "## Dead channels (present sensor, ALL features NaN)\n",
          f"{len(r['dead'])} (recording x sensor) instances.\n", "### Bad-sensor ranking",
          "| sensor | dead in N recordings |", "|---|---:|"]
    for s, n in sorted(r["sensor_dead"].items(), key=lambda x: -x[1]):
        L.append(f"| {s} | {n} |")
    L.append("")
    if r["partial"]:
        L += ["## Partial-channel NaN (investigate)\n", "| dataset | subject | session | sensor | #feat NaN |",
              "|---|---|---|---|---:|"]
        for d, sub, ses, s, n in r["partial"][:50]:
            L.append(f"| {d} | {sub} | {ses} | {s} | {n} |")
    else:
        L.append("## Partial-channel NaN\n\nNone — every scattered NaN is a whole dead channel.")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------- html
def _tile(label, value, sub=""):
    return (f'<div class="tile"><div class="tile-v">{value}</div>'
            f'<div class="tile-l">{html.escape(label)}</div>'
            f'{f"<div class=tile-s>{html.escape(sub)}</div>" if sub else ""}</div>')


def to_html(r: dict) -> str:
    esc = html.escape
    nan_pct = r["total_nan"] / r["total_cells"]
    struct_pct = r["structural"] / max(r["total_nan"], 1)
    scat_pct = r["scattered"] / max(r["total_nan"], 1)

    tiles = "".join([
        _tile("total NaN", f"{nan_pct:.2%}", f"{r['total_nan']:,} / {r['total_cells']:,} cells"),
        _tile("structural (expected)", f"{struct_pct:.0%}", "sensor absent from a dataset"),
        _tile("scattered (data quality)", f"{scat_pct:.0%}", f"{r['scattered']:,} cells"),
        _tile("dead-channel instances", f"{len(r['dead'])}", f"{len(r['sensor_dead'])} distinct sensors"),
        _tile("partial (feature bug?)", f"{len(r['partial'])}", "0 = pipeline clean"),
    ])

    # per-dataset table
    pd_rows = "".join(
        f"<tr><td>{esc(d['dataset'])}</td><td class=n>{d['recordings']}</td>"
        f"<td class=n>{d['sensors_present']}</td><td class=n>{d['nan_frac']:.4f}</td>"
        f"<td class=n>{d['nan_cells']:,}</td></tr>"
        for d in sorted(r["per_dataset"], key=lambda x: -x["nan_cells"]))

    # heatmap: interesting sensors (rows) x dataset (cols)
    datasets = r["datasets"]
    head = "".join(f"<th class=hcol>{esc(d)}</th>" for d in datasets)
    hrows = []
    for s in r["interesting"]:
        cells = []
        for d in datasets:
            status, n, tot = r["grid"][(s, d)]
            if status == "absent":
                cells.append(f'<td class="cell absent" title="{esc(s)} — absent (not recorded) in '
                             f'{esc(d)}; structural, expected">N/A</td>')
            elif status == "dead":
                frac = n / max(tot, 1)
                a = 0.60 + 0.40 * frac  # red intensity by fraction dead; floor 0.60 keeps white text legible
                cells.append(f'<td class="cell dead" style="--a:{a:.2f}" '
                             f'title="{esc(s)} DEAD in {n}/{tot} {esc(d)} recordings (all features NaN)">'
                             f'{n}/{tot}</td>')
            else:
                cells.append(f'<td class="cell live" title="{esc(s)} present &amp; live in {esc(d)} '
                             f'({tot} recordings)"></td>')
        hrows.append(f"<tr><th class=hrow>{esc(s)}</th>{''.join(cells)}</tr>")
    heatmap = (f"<table class=heat><thead><tr><th></th>{head}</tr></thead>"
               f"<tbody>{''.join(hrows)}</tbody></table>")

    # feature x dataset heatmap (sequential blue by NaN count; normalized for contrast)
    maxfd = max(r["feat_ds"].values()) or 1
    frows = []
    for f in sorted(r["feats"]):
        cells = []
        for d in datasets:
            n = r["feat_ds"][(f, d)]
            recs = r["recs_per"][d]
            frac = n / (recs * r["n_sensors"]) if recs else 0.0
            if n == 0:
                cells.append(f'<td class="cell live" title="{esc(f)}: 0 NaN in {esc(d)}"></td>')
            else:
                inten = n / maxfd
                cells.append(f'<td class="cell seq" style="--i:{inten:.3f}" '
                             f'title="{esc(f)}: {n} NaN cells in {esc(d)} ({frac:.2%})">{n}</td>')
        frows.append(f"<tr><th class=hrow>{esc(f)}</th>{''.join(cells)}</tr>")
    feat_heat = (f"<table class=heat><thead><tr><th></th>{head}</tr></thead>"
                 f"<tbody>{''.join(frows)}</tbody></table>")
    if r["feat_uniform"]:
        v = next(iter(r["feat_nan"].values()))
        feat_note = (f'<p class="ok">Feature-independent — all {r["n_features"]} features have an '
                     f'<b>identical</b> NaN count ({v} cells each), so within every dataset the '
                     f'columns below are uniform. NaN is driven entirely by sensors (absent or dead), '
                     f'not by any feature\'s computation — no feature is disproportionately failing.</p>')
    else:
        feat_note = ('<p class="ok" style="border-color:var(--dead)">NaN counts DIFFER across features '
                     '— some feature computes NaN more than others (see the non-uniform rows below); '
                     'investigate it.</p>')

    # bad-sensor ranking
    rank = "".join(f"<tr><td>{esc(s)}</td><td class=n>{n}</td></tr>"
                   for s, n in sorted(r["sensor_dead"].items(), key=lambda x: -x[1]))

    # partial (should be empty)
    if r["partial"]:
        prows = "".join(f"<tr><td>{esc(str(d))}</td><td>{esc(str(sub))}</td><td>{esc(str(ses))}</td>"
                        f"<td>{esc(str(s))}</td><td class=n>{n}</td></tr>"
                        for d, sub, ses, s, n in r["partial"][:100])
        partial_block = (f"<h2>⚠ Partial-channel NaN — feature-specific failures ({len(r['partial'])})</h2>"
                         f"<table class=tbl><thead><tr><th>dataset</th><th>subject</th><th>session</th>"
                         f"<th>sensor</th><th class=n>#feat NaN</th></tr></thead><tbody>{prows}</tbody></table>")
    else:
        partial_block = ('<h2>Partial-channel NaN</h2><p class="ok">None — every scattered NaN is a whole '
                         'dead channel (all 14 features NaN together). No feature-computation failures: '
                         'the pipeline produced a value for every live channel.</p>')

    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>NaN QC — cocodelics features</title>
<style>
  :root {{
    --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
    --grid:#e1e0d9; --border:rgba(11,11,11,.10); --absent:#d8d7d0; --dead:#d03b3b; --seq:#256abf;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
      --grid:#2c2c2a; --border:rgba(255,255,255,.10); --absent:#3a3a37; --dead:#d03b3b; --seq:#3987e5; }}
  }}
  html{{color-scheme:light dark}}
  body{{margin:0;background:var(--page);color:var(--ink);
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5}}
  .wrap{{max-width:960px;margin:0 auto;padding:32px 20px 64px}}
  h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:34px 0 10px}}
  .sub{{color:var(--ink2);font-size:13px;margin:0 0 24px}}
  code{{background:var(--grid);padding:1px 5px;border-radius:4px;font-size:12px}}
  .tiles{{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}}
  .tile{{flex:1 1 150px;background:var(--surface);border:1px solid var(--border);
    border-radius:10px;padding:14px 16px}}
  .tile-v{{font-size:24px;font-weight:650;letter-spacing:-.01em}}
  .tile-l{{font-size:12px;color:var(--ink2);margin-top:2px}} .tile-s{{font-size:11px;color:var(--muted);margin-top:3px}}
  table{{border-collapse:collapse;font-size:13px}}
  .tbl{{width:100%;background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden}}
  .tbl th,.tbl td{{text-align:left;padding:7px 12px;border-bottom:1px solid var(--grid)}}
  .tbl th{{color:var(--ink2);font-weight:600;font-size:12px}}
  .tbl tr:last-child td{{border-bottom:none}}
  td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}}
  .heatwrap{{overflow-x:auto}}
  .heat{{margin-top:4px}}
  .heat th.hcol{{font-size:11px;color:var(--ink2);font-weight:600;padding:4px 8px;text-align:center;
    writing-mode:horizontal-tb}}
  .heat th.hrow{{font-size:12px;color:var(--ink2);font-weight:600;padding:3px 12px 3px 4px;text-align:right;
    font-variant-numeric:tabular-nums}}
  .cell{{width:74px;height:26px;text-align:center;font-size:11px;font-variant-numeric:tabular-nums;
    border:2px solid var(--surface);border-radius:5px;color:var(--ink)}}
  .cell.live{{background:var(--grid);opacity:.35}}
  .cell.absent{{background:var(--absent);color:var(--muted)}}
  .cell.dead{{background:color-mix(in srgb, var(--dead) calc(var(--a)*100%), var(--surface));
    color:#fff;font-weight:600}}
  .cell.seq{{background:color-mix(in srgb, var(--seq) calc(var(--i)*60%), var(--surface));
    color:var(--ink)}}
  .legend{{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin:10px 2px}}
  .sw{{display:inline-block;width:13px;height:13px;border-radius:3px;vertical-align:-2px;margin-right:5px;
    border:1px solid var(--border)}}
  .ok{{color:var(--ink2);font-size:13px;background:var(--surface);border:1px solid var(--border);
    border-radius:10px;padding:12px 14px}}
</style></head>
<body><div class=wrap>
<h1>NaN QC — cocodelics feature table</h1>
<p class=sub>source <code>{esc(r['csv'])}</code> · {r['n_recordings']} recordings ×
  {r['n_features']} features × {r['n_sensors']} sensors (union) = {r['total_cells']:,} cells</p>
<div class=tiles>{tiles}</div>

<h2>Per-dataset</h2>
<table class=tbl><thead><tr><th>dataset</th><th class=n>recordings</th><th class=n>sensors present</th>
  <th class=n>NaN frac</th><th class=n>NaN cells</th></tr></thead><tbody>{pd_rows}</tbody></table>

<h2>Sensor × dataset NaN map <span style="font-weight:400;color:var(--muted);font-size:12px">
  (only sensors with any NaN)</span></h2>
<div class=legend>
  <span><span class=sw style="background:var(--absent)"></span>absent — not recorded (structural, expected)</span>
  <span><span class=sw style="background:var(--dead)"></span>dead — present but all-NaN (data quality); N/M = recordings</span>
  <span><span class=sw style="background:var(--grid);opacity:.35"></span>present &amp; live</span>
</div>
<div class=heatwrap>{heatmap}</div>

<h2>Feature × dataset NaN <span style="font-weight:400;color:var(--muted);font-size:12px">
  (is any feature disproportionately NaN?)</span></h2>
{feat_note}
<div class=heatwrap>{feat_heat}</div>

<h2>Bad-sensor ranking</h2>
<table class=tbl><thead><tr><th>sensor</th><th class=n>dead in N recordings</th></tr></thead>
  <tbody>{rank}</tbody></table>

{partial_block}
</div></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="outputs/aggregate_full_raw.csv")
    ap.add_argument("--html", default="NAN_QC_REPORT.html")
    ap.add_argument("--md", default="NAN_QC_REPORT.md")
    args = ap.parse_args()
    r = analyze(args.csv)
    Path(args.html).write_text(to_html(r))
    Path(args.md).write_text(to_markdown(r))
    print(f"total NaN {r['total_nan']} ({r['total_nan']/r['total_cells']:.2%}): "
          f"structural {r['structural']} + scattered {r['scattered']} "
          f"({len(r['dead'])} dead-channel instances, {len(r['partial'])} partial). "
          f"-> {args.html}, {args.md}")


if __name__ == "__main__":
    main()

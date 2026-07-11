"""Quality-control analysis of NaNs in a coco-pipe-format feature table.

Given an aggregate CSV with columns ``feature-<feat>.spaces-<sensor>`` (+ id columns
``dataset``/``subject``/``session``/``filepath``), decompose every NaN cell into:

  * STRUCTURAL  -- the sensor is absent from that whole dataset (the 273 sensors are the
    *union* across sites; a dataset recorded with fewer channels leaves those union
    columns all-NaN for all its recordings). Expected, not a data-quality problem.
  * SCATTERED   -- the sensor IS present in the dataset but is NaN in a specific
    recording. Split further into:
      - dead channel   : ALL features NaN for that (recording, sensor) -> flat/dead sensor.
      - partial        : only SOME features NaN for a present (recording, sensor) ->
                         a feature-specific computation failure (worth investigating).

Emits a Markdown report (per-dataset summary, structural/scattered decomposition, the
dead-channel table, and a bad-sensor ranking) and prints a one-line summary.

Usage:
    python nan_qc.py [aggregate_csv] [-o report.md]
    (defaults: outputs/aggregate_full_raw.csv -> NAN_QC_REPORT.md)
"""
from __future__ import annotations

import argparse
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

    # per-dataset presence + summary
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

    # decompose every recording x sensor
    structural = 0
    dead = []       # (dataset, subject, session, filepath, sensor)
    partial = []    # (dataset, subject, session, sensor, n_feat_nan)
    sensor_dead = {}
    for _, row in df.iterrows():
        d = row["dataset"]
        for s in sensors:
            block = row[cols_by_sensor[s]]
            n_nan = int(block.isna().sum())
            if n_nan == 0:
                continue
            if s not in present[d]:
                structural += n_nan                      # absent sensor for this dataset
            elif n_nan == n_feat:
                dead.append((d, row.get("subject"), row.get("session"),
                             str(row.get("filepath", "")).split("/")[-1], s))
                sensor_dead[s] = sensor_dead.get(s, 0) + 1
            else:
                partial.append((d, row.get("subject"), row.get("session"), s, n_nan))

    scattered = len(dead) * n_feat + sum(p[4] for p in partial)
    return dict(
        csv=csv_path, n_recordings=df.shape[0], n_features=n_feat, n_sensors=len(sensors),
        total_cells=total_cells, total_nan=total_nan, structural=structural,
        scattered=scattered, per_dataset=per_dataset, dead=dead, partial=partial,
        sensor_dead=sensor_dead, feats=feats,
    )


def to_markdown(r: dict) -> str:
    L = []
    L.append("# NaN QC report — neurodags feature table\n")
    L.append(f"- source: `{r['csv']}`")
    L.append(f"- shape: {r['n_recordings']} recordings x {r['n_features']} features x "
             f"{r['n_sensors']} sensors (union) = {r['total_cells']} cells")
    L.append(f"- **total NaN: {r['total_nan']} ({r['total_nan']/r['total_cells']:.2%})**")
    L.append(f"  - structural (sensor absent from dataset — expected): "
             f"{r['structural']} ({r['structural']/max(r['total_nan'],1):.0%})")
    L.append(f"  - scattered (sensor present but NaN — data quality): "
             f"{r['scattered']} ({r['scattered']/max(r['total_nan'],1):.0%})")
    L.append("")
    L.append("## Per-dataset\n")
    L.append("| dataset | recordings | sensors present | NaN frac | NaN cells |")
    L.append("|---|---:|---:|---:|---:|")
    for d in sorted(r["per_dataset"], key=lambda x: -x["nan_cells"]):
        L.append(f"| {d['dataset']} | {d['recordings']} | {d['sensors_present']} | "
                 f"{d['nan_frac']:.4f} | {d['nan_cells']} |")
    L.append("")
    L.append("## Dead channels (present sensor, ALL features NaN in a recording)\n")
    L.append(f"{len(r['dead'])} (recording x sensor) instances across "
             f"{len({(d[0],d[1],d[2]) for d in r['dead']})} recordings.\n")
    L.append("### Bad-sensor ranking")
    L.append("| sensor | dead in N recordings |")
    L.append("|---|---:|")
    for s, n in sorted(r["sensor_dead"].items(), key=lambda x: -x[1]):
        L.append(f"| {s} | {n} |")
    L.append("")
    if r["partial"]:
        L.append("## ⚠️ Partial-channel NaN (present sensor, SOME features NaN — investigate)\n")
        L.append("| dataset | subject | session | sensor | #features NaN |")
        L.append("|---|---|---|---|---:|")
        for d, sub, ses, s, n in r["partial"][:50]:
            L.append(f"| {d} | {sub} | {ses} | {s} | {n} |")
        if len(r["partial"]) > 50:
            L.append(f"| … | | | | ({len(r['partial'])-50} more) |")
    else:
        L.append("## Partial-channel NaN\n\nNone — every scattered NaN is a whole dead channel "
                 "(all features NaN together), so no feature-specific computation failures.")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="outputs/aggregate_full_raw.csv")
    ap.add_argument("-o", "--output", default="NAN_QC_REPORT.md")
    args = ap.parse_args()
    r = analyze(args.csv)
    Path(args.output).write_text(to_markdown(r))
    print(f"total NaN {r['total_nan']} ({r['total_nan']/r['total_cells']:.2%}): "
          f"structural {r['structural']} + scattered {r['scattered']} "
          f"({len(r['dead'])} dead-channel instances, {len(r['partial'])} partial). "
          f"-> {args.output}")


if __name__ == "__main__":
    main()

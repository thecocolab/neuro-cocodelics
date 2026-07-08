#!/usr/bin/env python3
"""Convert a neurodags wide feature CSV into the coco-pipe aggregate format.

The neurodags `dataframe --format wide` output uses columns like:

    lzivComplexityMeanEpochs.nc@spaces-MRC41
    hjorthMeanEpochs.nc@hjorthComponents-mobility_spaces-MRC41

The downstream ML (`machine-learning/run_ml.py`, `data/split_csv.py`, `viz/*`)
expects the coco-pipe convention:

    feature-lzivComplexityMeanEpochs.spaces-MRC41
    feature-hjorthMobilityMeanEpochs.spaces-MRC41

plus id/metadata columns (filepath, dataset, subject, session, task, id).

This script performs that rename + adds the metadata columns parsed from the
BIDS-style file_path, producing an `aggregate@raw.csv` that `split_csv.py` can
consume unchanged.

Usage:
    python to_cocopipe_columns.py feats_wide.csv aggregate@raw.csv
"""
from __future__ import annotations

import re
import sys

import pandas as pd

# neurodags column: "<deriv>.nc@<flattened dims>"; dims sorted alphabetically and
# joined by "_", each as "<dim>-<coordvalue>". Channel dim is "spaces".
_COL_RE = re.compile(r"^(?P<deriv>.+?)\.nc@(?P<dims>.+)$")
_HJORTH = {"mobility": "hjorthMobilityMeanEpochs", "complexity": "hjorthComplexityMeanEpochs"}


def rename_feature_column(col: str) -> str | None:
    """neurodags feature column -> coco-pipe `feature-<name>.spaces-<sensor>` (or None if not a feature)."""
    m = _COL_RE.match(col)
    if not m:
        return None
    deriv, dims = m.group("deriv"), m.group("dims")
    parts = dict(p.split("-", 1) for p in dims.split("_") if "-" in p)
    sensor = parts.get("spaces")
    if sensor is None:
        return None
    # hjorth carries a second dim (hjorthComponents) -> map to coco's split feature names
    if "hjorthComponents" in parts:
        deriv = _HJORTH.get(parts["hjorthComponents"], f"{deriv}_{parts['hjorthComponents']}")
    return f"feature-{deriv}.spaces-{sensor}"


def parse_entities(file_path: str) -> dict[str, str]:
    """Pull BIDS entities (subject/session/task) from a *_meg.fif path."""
    def grab(key: str) -> str:
        m = re.search(rf"{key}-([A-Za-z0-9]+)", str(file_path))
        return m.group(1) if m else ""
    return {"subject": grab("sub"), "session": grab("ses"), "task": grab("task")}


def convert(in_csv: str, out_csv: str) -> pd.DataFrame:
    df = pd.read_csv(in_csv)
    feature_cols = {c: rename_feature_column(c) for c in df.columns}
    feature_cols = {c: new for c, new in feature_cols.items() if new is not None}

    ent = df["file_path"].apply(parse_entities).apply(pd.Series)
    out = pd.DataFrame(
        {
            "filepath": df["file_path"],
            "dataset": df["dataset"],
            "subject": ent["subject"],
            "session": ent["session"],
            "task": ent["task"],
            "id": ent["subject"],  # split_csv drops this; kept for schema parity
        }
    )
    renamed = df[list(feature_cols)].rename(columns=feature_cols)
    out = pd.concat([out, renamed], axis=1)
    out.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {out.shape[0]} rows x {out.shape[1]} cols "
          f"({renamed.shape[1]} feature cols)")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python to_cocopipe_columns.py <neurodags_wide.csv> <aggregate@raw.csv>")
    convert(sys.argv[1], sys.argv[2])

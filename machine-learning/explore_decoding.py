"""Quick drug-vs-placebo decoding exploration with coco-pipe's `decoding` module.

Runs, per problem (each drug dataset; LSD split per task + an lsd-avg), a subject-grouped
cross-validated classification (placebo=0 vs drug=1) over the notch feature table, across
several classical models, and writes a leaderboard.

Feature table: aggregate_notch_raw.csv (coco-pipe cols `feature-<f>.spaces-<sensor>`).
Per problem we drop any feature column with a NaN (dead/structural sensors — matches
data/split_csv.py), scale, and evaluate with GroupKFold by subject.

Run with the coco-pipe venv:
    ~/code/coco-pipe/.venv/bin/python explore_decoding.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import (
    CVConfig,
    DummyClassifierConfig,
    HistGradientBoostingClassifierConfig,
    LogisticRegressionConfig,
    RandomForestClassifierConfig,
    SVCConfig,
)

AGG = Path("~/code/sprint/neuro-cocodelics/prep_and_extract/neurodags/"
           "outputs/aggregate_notch_raw.csv").expanduser()
OUT = Path("~/code/sprint/neuro-cocodelics/machine-learning/outputs/ml_explore").expanduser()
OUT.mkdir(parents=True, exist_ok=True)

METRICS = ["accuracy", "roc_auc", "f1"]


def models():
    return {
        "LogReg": LogisticRegressionConfig(max_iter=2000),
        "SVC-rbf": SVCConfig(kernel="rbf"),          # probability=True by default
        "RandomForest": RandomForestClassifierConfig(n_estimators=200, random_state=42),
        "HistGB": HistGradientBoostingClassifierConfig(random_state=42),
        "Dummy": DummyClassifierConfig(strategy="stratified", random_state=42),
    }


def problems(df: pd.DataFrame, feat: list[str]):
    """Yield (name, sub_df) — one binary drug-vs-placebo problem each."""
    for ds in ["ketamine", "perampanel", "psilocybin", "tiagabine"]:
        yield ds, df[df.dataset == ds]
    lsd = df[df.dataset == "lsd"]
    for task, g in lsd.groupby("task"):
        yield f"lsd-{task}", g
    # lsd-avg: mean features over tasks, per subject x session
    avg = lsd.groupby(["subject", "session"], as_index=False)[feat].mean()
    yield "lsd-avg", avg


def run():
    df = pd.read_csv(AGG)
    feat = [c for c in df.columns if c.startswith("feature-") and ".spaces-" in c]
    rows = []
    for name, sub in problems(df, feat):
        cols = sub[feat].dropna(axis=1).columns.tolist()       # drop any-NaN cols (dead/structural)
        X = sub[cols].to_numpy(dtype=float)
        y = (sub["session"] != "placebo").astype(int).to_numpy()
        groups = sub["subject"].to_numpy()
        n_subj = len(np.unique(groups))
        n_splits = int(min(5, n_subj))
        cfg = ExperimentConfig(
            task="classification",
            models=models(),
            metrics=METRICS,
            cv=CVConfig(strategy="group_kfold", n_splits=n_splits, auto_reduce_n_splits=True),
            use_scaler=True,
            random_state=42,
            n_jobs=-1,
            verbose=False,
        )
        t0 = time.time()
        try:
            res = Experiment(cfg).run(X, y, groups=groups, feature_names=cols)
            summ = res.summary()
        except Exception as e:
            print(f"[{name}] ERROR: {type(e).__name__}: {e}")
            continue
        dt = time.time() - t0
        print(f"[{name}] n={len(y)} subj={n_subj} feat={len(cols)} ({dt:.0f}s)")
        for model, r in summ.iterrows():
            rows.append({
                "problem": name, "n": len(y), "n_subjects": n_subj, "n_features": len(cols),
                "model": model,
                "accuracy": r.get("accuracy_mean"), "accuracy_std": r.get("accuracy_std"),
                "roc_auc": r.get("roc_auc_mean"), "roc_auc_std": r.get("roc_auc_std"),
                "f1": r.get("f1_mean"), "f1_std": r.get("f1_std"),
            })
        # per-problem full summary for the record
        summ.to_csv(OUT / f"summary_{name}.csv")

    lb = pd.DataFrame(rows)
    lb.to_csv(OUT / "leaderboard.csv", index=False)
    print("\n===== LEADERBOARD (roc_auc, non-Dummy, sorted) =====")
    show = lb[lb.model != "Dummy"].sort_values("roc_auc", ascending=False)
    with pd.option_context("display.width", 160, "display.max_rows", None):
        print(show[["problem", "model", "n", "n_subjects", "n_features",
                    "roc_auc", "accuracy", "f1"]].round(3).to_string(index=False))
    print(f"\nwrote {OUT}/leaderboard.csv")


if __name__ == "__main__":
    run()

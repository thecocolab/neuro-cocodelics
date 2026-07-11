"""Subject-paired permutation significance for the drug-vs-placebo decoding.

coco-pipe's built-in permutation errors on single-class folds (GroupKFold + shuffled
labels on small n). Instead we use out-of-fold (OOF) predictions and score roc_auc on
the FULL OOF set (both classes always present -> no per-fold degeneracy), with a
WITHIN-SUBJECT label permutation (each subject has 1 placebo + 1 drug, so this is a
paired sign-flip null — the correct unit of inference for this repeated-measures design).

p = (1 + #{null_auc >= obs_auc}) / (1 + n_perms).

Run:  ~/code/coco-pipe/.venv/bin/python perm_test.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AGG = Path("../prep_and_extract/neurodags/outputs/aggregate_notch_raw.csv")
OUT = Path("outputs/ml_explore"); OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "LogReg": (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), 500),
    "RandomForest": (lambda: make_pipeline(StandardScaler(),
                     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)), 300),
}


def problems(df, feat):
    for ds in ["ketamine", "perampanel", "psilocybin", "tiagabine"]:
        yield ds, df[df.dataset == ds]
    lsd = df[df.dataset == "lsd"]
    for task, g in lsd.groupby("task"):
        yield f"lsd-{task}", g
    yield "lsd-avg", lsd.groupby(["subject", "session"], as_index=False)[feat].mean()


def oof_auc_acc(make, X, y, groups, cv):
    proba = cross_val_predict(make(), X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, proba), accuracy_score(y, (proba >= 0.5).astype(int))


def within_subject_perm(y, groups, rng):
    yp = y.copy()
    for s in np.unique(groups):
        idx = np.where(groups == s)[0]
        yp[idx] = rng.permutation(y[idx])
    return yp


def main():
    df = pd.read_csv(AGG)
    feat = [c for c in df.columns if c.startswith("feature-") and ".spaces-" in c]
    rows = []
    for name, sub in problems(df, feat):
        cols = sub[feat].dropna(axis=1).columns.tolist()
        X = sub[cols].to_numpy(float)
        y = (sub["session"] != "placebo").astype(int).to_numpy()
        groups = sub["subject"].to_numpy()
        cv = GroupKFold(n_splits=int(min(5, len(np.unique(groups)))))
        for model, (make, n_perms) in MODELS.items():
            t0 = time.time()
            rng = np.random.default_rng(42)
            obs_auc, obs_acc = oof_auc_acc(make, X, y, groups, cv)
            null = np.empty(n_perms)
            for i in range(n_perms):
                yp = within_subject_perm(y, groups, rng)
                null[i], _ = oof_auc_acc(make, X, yp, groups, cv)
            p = (1 + int(np.sum(null >= obs_auc))) / (1 + n_perms)
            dt = time.time() - t0
            rows.append({"problem": name, "model": model, "n": len(y),
                         "n_subjects": int(len(np.unique(groups))),
                         "roc_auc": round(obs_auc, 3), "accuracy": round(obs_acc, 3),
                         "null_auc_mean": round(float(null.mean()), 3),
                         "null_auc_p95": round(float(np.quantile(null, 0.95)), 3),
                         "p_perm": round(p, 4), "n_perms": n_perms})
            print(f"[{name}/{model}] auc={obs_auc:.3f} p={p:.4f} "
                  f"(null~{null.mean():.2f}) {dt:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    res["sig_0.05"] = res["p_perm"] < 0.05
    res.to_csv(OUT / "permutation_results.csv", index=False)
    print("\n===== PERMUTATION SIGNIFICANCE (within-subject paired null) =====")
    with pd.option_context("display.width", 170, "display.max_rows", None):
        print(res.sort_values(["model", "p_perm"]).to_string(index=False))
    print(f"\nwrote {OUT}/permutation_results.csv")


if __name__ == "__main__":
    main()

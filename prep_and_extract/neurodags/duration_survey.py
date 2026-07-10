"""Survey recording durations per dataset (header-only) to size the DFA window.

DFA needs long windows to probe slow dynamics; 30 s epochs are short. To pick a
window that is COMPARABLE across datasets we need the recording-length
distribution per dataset (they differ: resting vs LSD tasks; psilocybin is cropped
to InfusionStop->RestStop). Reports per-dataset min/median/max duration (s) + how
many non-overlapping windows of a few candidate lengths each recording yields.
"""
import glob, os, re
from collections import defaultdict
import numpy as np
import mne

mne.set_log_level("error")
BIDS = "/home/yorguin/projects/rrg-kjerbi/shared/neuro-cocodelics/bids"
DATASETS = ["ketamine", "perampanel", "psilocybin", "tiagabine", "LSD"]
CANDS = [30, 60, 120, 180, 240]  # candidate DFA window lengths (s)


def task_of(path):
    m = re.search(r"task-([A-Za-z0-9]+)", path)
    return m.group(1) if m else "?"


print(f"{'dataset':12} {'n':>4} {'min_s':>8} {'med_s':>8} {'max_s':>8}   windows/rec @ " +
      " ".join(f"{c}s" for c in CANDS))
overall_min = np.inf
per_ds_durs = {}
for ds in DATASETS:
    fs = sorted(glob.glob(f"{BIDS}/MEG_{ds}/**/*_meg.fif", recursive=True))
    fs = [f for f in fs if "split-0" not in f or "split-01" in f]  # drop split continuations
    durs = []
    task_durs = defaultdict(list)
    for f in fs:
        try:
            raw = mne.io.read_raw_fif(f, preload=False, verbose="error")
            d = raw.n_times / raw.info["sfreq"]
            durs.append(d)
            task_durs[task_of(f)].append(d)
        except Exception as e:
            print(f"  !! {os.path.basename(f)}: {type(e).__name__}")
    if not durs:
        print(f"{ds:12}  NO FILES"); continue
    durs = np.array(durs)
    per_ds_durs[ds] = durs
    overall_min = min(overall_min, durs.min())
    wins = [f"{int(np.floor(durs.min()/c)):>{len(str(c))+1}}" for c in CANDS]  # worst-case (min rec)
    print(f"{ds:12} {len(durs):>4} {durs.min():8.1f} {np.median(durs):8.1f} {durs.max():8.1f}   "
          + " ".join(f"{int(np.floor(durs.min()/c))}@{c}s" for c in CANDS))
    if ds == "LSD":  # tasks differ a lot
        for t, td in sorted(task_durs.items()):
            td = np.array(td)
            print(f"    task-{t:8} n={len(td):>3} min={td.min():7.1f} med={np.median(td):7.1f} max={td.max():7.1f}")

print(f"\noverall shortest recording = {overall_min:.1f} s")
print("=> a common non-overlapping DFA window must be <= that, ideally with >=2-3 windows/rec.")
for c in CANDS:
    n_ok = sum(int(np.floor(d.min() / c) >= 1) for d in per_ds_durs.values())
    worst = min(int(np.floor(d.min() / c)) for d in per_ds_durs.values())
    print(f"  window {c:>4}s: every dataset yields >=1 window in {n_ok}/{len(per_ds_durs)} datasets; "
          f"worst-case windows/rec across datasets = {worst}")

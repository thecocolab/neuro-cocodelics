"""Per-derivative wall-time profile of a neurodags run from its jsonl log.

Usage: python profile_run.py <run_*.jsonl>
neurodags processes one derivative at a time across all files, so bucketing the
"Processed file successfully" events by derivative and diffing each derivative's
last-completion timestamp gives a clean per-derivative wall-time + s/file profile
(handy for spotting the expensive features, e.g. LZiv/ZapLine).
"""
import json, sys, collections
from datetime import datetime

def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

path = sys.argv[1]
# (timestamp, derivative) for each file completion; neurodags runs one derivative
# at a time across all files, so bucketing by derivative + ordering by time gives
# a clean per-derivative wall-time profile.
rows = []
for l in open(path):
    e = json.loads(l)
    if e.get("event") == "Processed file successfully":
        rows.append((ts(e["timestamp"]), e.get("derivative", "?")))
rows.sort()

# per-derivative first/last completion + count
agg = collections.OrderedDict()
for t, d in rows:
    if d not in agg:
        agg[d] = {"first": t, "last": t, "n": 0}
    agg[d]["last"] = t
    agg[d]["n"] += 1

# order derivatives by when they finished; wall_i = last_i - last_{i-1}
order = sorted(agg, key=lambda d: agg[d]["last"])
print(f"{'derivative':36} {'n':>4} {'wall_s':>8} {'s/file':>7}")
prev_last = None
prof = []
for d in order:
    last = agg[d]["last"]
    wall = (last - prev_last).total_seconds() if prev_last else (last - agg[d]["first"]).total_seconds()
    prof.append((wall, d, agg[d]["n"]))
    prev_last = last
for wall, d, n in sorted(prof, reverse=True):
    print(f"{d:36} {n:>4} {wall:8.1f} {wall/max(n,1):7.1f}")
print(f"\ntotal wall (first->last file completion): "
      f"{(rows[-1][0]-rows[0][0]).total_seconds():.0f}s over {len(rows)} file-derivative completions")

"""Summarize a neurodags run jsonl: errors (by derivative), completions, coverage."""
import json, sys, collections

path = sys.argv[1]
errs = collections.Counter()
ok = collections.Counter()
for line in open(path):
    try:
        e = json.loads(line)
    except Exception:
        continue
    ev = e.get("event")
    if ev == "Error running derivative":
        errs[e.get("derivative", "?")] += 1
    elif ev == "Processed file successfully":
        ok[e.get("derivative", "?")] += 1

print("errors:", dict(errs) if errs else "NONE")
print("derivatives_ok:", len(ok), "completions:", sum(ok.values()))
print("alphaEnvelopeDfa completions:", ok.get("alphaEnvelopeDfa", 0))
print("DenoisedRaw completions:", ok.get("DenoisedRaw", 0))

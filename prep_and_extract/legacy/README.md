# legacy/ — deprecated coco-pipe artifacts (kept for reference only)

**Do not use for new work.** These are the coco-pipe–era pipeline definitions, retained
only as a **feature-parity reference** while the project migrates off coco-pipe.

## Why this is legacy

The project's feature extraction is being reimplemented on **neurodags**
(`../neurodags/`), and bidsification is consolidated onto the **standalone per-dataset
scripts** (`../bidsification/`). That makes everything under `r2c_project/` obsolete as a
*live* path:

- `redefinitions_cocosprint.py` — coco-pipe-injected `bidsify()` + `prepare()`. Its
  `bidsify()` duplicated the standalone scripts' logic, only covered LSD/perampanel/psilocybin
  (the **ketamine/tiagabine gap**), and still has a live `breakpoint()` at line 835. Its
  `prepare()` is now replaced by the neurodags `basic_preprocessing` node.
- `pipeline_cocosprint{,2}.yml`, `datasets_cocosprint.yml`, `requirements*.txt` — coco-pipe
  pipeline configs (v1 Fisher/Harmonicity, v2 phi/IIT). Superseded by the neurodags pipeline.

## Still useful here (as reference)

- `REPORT.md` — the authoritative write-up of the v1/v2 feature battery. Used to build the
  neurodags port and to plan the still-missing "scientific" features (see `../TODO.md §1`).
- `BIDSIFICATION_COMPARISON.md` — narrative standalone-vs-pipeline bidsification comparison.

The single active bidsification strategy is `../bidsification/` (see
`../bidsification/BIDSIFICATION_VERSIONS.html`).

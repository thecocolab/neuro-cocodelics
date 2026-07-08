# prep_and_extract — backlog / TODO

## 1. Add the "scientific" features we explored in the old (coco-pipe) version

The neurodags reimplementation (`neurodags/`) currently ports **only the classical
complexity battery** (antropy/spectral built-in nodes). The old coco-pipe pipeline
(`r2c_project/pipeline_cocosprint{,2}.yml`, see `r2c_project/REPORT.md`) also computed
"scientific" features that are **not yet reimplemented**. Add them as neurodags custom
nodes (`new_definitions.py`):

- **v1 features**
  - `FisherInformation` (ordinal Fisher information; coco-pipe `fisher_information_feature`, delay=1, dim=3)
  - `Harmonicity` (spectral; `PowerSpectrum` → `feature_harmonicity` → `process_harmonicity_output`)
- **v2 features** (phi / integrated-information-decomposition, https://github.com/Imperial-MIND-lab/integrated-info-decomp)
  - `Atoms` (single_atoms: tau=5, kind=gaussian, redundancy=MMI) → then:
  - `InfoDyn` (InformationDynamics), `IID` (IntegratedInformationDecomposition), `IIT` (IntegratedInformationTheory), each Mean/SD over epochs
- **also-defined-but-inactive in coco-pipe** (decide if wanted): `RateEntropy` (kmax=10),
  `Distance2Criticality` (chaos_feature, sigma=0.5).

Each becomes a `@register_node` wrapping the same upstream lib coco-pipe used, then a
derivative + epoch aggregation in `pipeline_cocodelics.yml`, mirroring the classical ones.

## 2. Parity check of the classical battery vs old coco-pipe derivatives

Compare `derivatives_neurodags/` against the reference `derivatives/features@prepDur30Ov20/`:
- **channel selection + naming — CONFIRMED BROKEN by the smoke run (job 47578952), fix first.**
  The pipeline computes on ALL channels in the `.fif`, so the aggregate had 612 "sensors"
  incl 68 non-MEG (`BG/BP/BR` ref coils, `EEG057-059`, `UPPT` trigger, `SCLK` clock, `HLC`),
  and every one of 7956 feature cols had a NaN (78k cells) because CTF names carry a varying
  `-<runid>` suffix (`-3305`, `-177`, `-4408`) so the same sensor mis-aligns across datasets.
  Fix (before any parity work): in preprocessing (a) pick MEG data channels only
  (`pick_types(meg=True, ref_meg=False)` / the ~271 CTF mags), and (b) strip the `-<runid>`
  suffix so names are clean `MLC11` etc. `viz/plot_functions.py` already does
  `rename_channels(lambda x: x.replace("-3305",""))` — mirror that. Likely needs a small custom
  node (basic_preprocessing has no pick/rename) or fixing it at the bidsification step.
- **normalization**: antropy `spectral_entropy` / `lziv_complexity` default to un-normalized;
  confirm whether coco-pipe normalized and add `normalize: true` to those node args if so.
- **higuchi `kmax`, perm/svd order & delay**: antropy defaults — match to coco-pipe if needed.

## 2b. FIF split-file handling (latent bug — neurodags does NOT handle it)

MNE can only write ~2 GB per `.fif`; larger recordings split into
`..._split-01_meg.fif` (entry point, carries a `next_fname` pointer) + `..._split-02_meg.fif` …
Reading `-01` auto-loads the rest. A plain scanner sees `-02+` as standalone files.

- **neurodags exposure:** `iterators.get_files_from_pattern` is a bare `glob.glob(**/*_meg.fif)`
  with a single optional `exclude_filter` glob — **no split awareness**. So `_split-02+` WOULD be
  picked up as independent source files → duplicate/partial-data feature rows.
- **Current status:** NOT triggered — 0 split files in any cocodelics BIDS dir on scratch
  (all recordings stayed under 2 GB). Latent; any future >2 GB bidsified file trips it.
- **Old r2c handling (trace):** `data/split_csv.py` drops rows whose filepath contains
  `split-02|split-03` and strips `_split-01` — a downstream aggregate cleanup (wasteful: `-02`
  features were already computed; only covers up to `-03`). NOT in `code/psychostimulants`.
- **Fix (preferred):** teach the neurodags scanner to drop split *continuations* generically —
  after glob, filter filenames whose split index ≥ 2 (BIDS `_split-DD`, and mne's plain
  `name-1.fif`/`-2.fif`), keeping `-01`/non-split. It's yjmantilla's repo → do it there (with a
  test); benefits every project. **Short-term defensive:** set per-dataset `exclude_pattern`
  (absolute glob, e.g. `.../MEG_<D>/**/*_split-0[2-9]_meg.fif`) once/if splits appear — note the
  single-glob limitation misses `_split-10+` (>18 GB, implausible here).

## 3. Make the bidsification scripts reproducible from the shared /project source

Scripts: `bidsification/MEG_{ketamine,perampanel,psilocybin,tiagabine}/1bidsifying.py`
(FieldTrip `.mat`) + `bidsification/MEG_LSD/{1download,2bidsifying_preparation,3bidsifying}.py`
(CTF `.ds`, incl the Jordan-Venkatesh ID corrections). Source of truth for raw data is now
`/home/yorguin/projects/def-kjerbi/data/MEG_*` (see memory `cocodelics-data-sources`).

**Output location DECIDED:** BIDS → `/home/yorguin/projects/rrg-kjerbi/shared/neuro-cocodelics/MEG_<D>`
— the lab's real shared project dir (owner hamza97, group def-kjerbi, group-writable, already holds
bidsified `EEG_DMT/sub-*`, `MEG_LSD/`, `MEG_ketamine/`, `test/`). Matches the existing `EEG_DMT`
convention (`sub-*` directly under the dataset dir). rrg-kjerbi is durable/backed-up (32/299 TiB).
NOTE: `rrg-kjerbi/shared/` itself is not writable, but `shared/neuro-cocodelics/` is. (The
`rrg-kjerbi/datasets/` dir is sesma's personal folder, NOT a shared area — earlier misread.)

DONE (2026-07-08):
- All 5 scripts repointed: raw source → `/project/def-kjerbi/data/MEG_*/meg_data` (LSD too),
  BIDS output → `.../shared/neuro-cocodelics/MEG_<D>`, `os.makedirs(OUTPUT_PATH)` added to the
  4 FieldTrip scripts. Metadata (csv/pkl/txt) → the same `MEG_<D>/` dir. All syntax-checked.

STILL TODO:
- **Test-run each script on the cluster** (needs `salloc` + the env deps: `sovabids`, `mat73`,
  `scipy`, `mne-bids`). Not yet executed — path fixes are untested against real data.
- LSD path: verify `2bidsifying_preparation.py` still finds the 413 `.ds` and the Jordan-Venkatesh
  ID corrections still hold with the `/project/.../MEG_LSD` tree (task set, session dates).
- Remove the dead trailing docstring block in the FieldTrip scripts; add a `main()` guard;
  drop the debug print loops. (Cosmetic — they run as-is.)
- **Repoint the feature pipeline** (`neurodags/datasets_cocodelics.yml`) to the new rrg BIDS
  root once bidsification has actually written there (currently it reads the scratch `bids`).
- Add a short run-order README (LSD = 3 steps: download → prepare → bidsify; others = 1 step).

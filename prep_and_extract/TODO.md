# prep_and_extract — backlog / TODO

## 1. Add the "scientific" features we explored in the old (coco-pipe) version

The neurodags reimplementation (`neurodags/`) currently ports **only the classical
complexity battery** (antropy/spectral built-in nodes). The old coco-pipe pipeline
(`legacy/r2c_project/pipeline_cocosprint{,2}.yml`, see `legacy/r2c_project/REPORT.md`) also computed
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

Each becomes a `@register_node` **in neurodags** (custom_nodes, NOT coco-pipe), then a
derivative + epoch aggregation in `pipeline_cocodelics.yml`, mirroring the classical ones.

**Reference implementations located (2026-07-08):** the r2c feature code lives in
`github.com/alberto-jj/raw_to_classification`, branch **`meeg_refactor`**, file
`eeg_raw_to_classification/features2.py`:
- `single_atoms(epochs, tau=5, redundancy='MMI', kind='gaussian')` — calls `phyid.calculate.calc_PhiID`
- `atoms_results(atoms, key=..., aggregation_mode='mean-sum')` — extracts InfoDyn / IID / IIT
- `fisher_information_feature(epochs, delay=1, dimension=3)`
- `feature_harmonicity(input_dict, height, distance, bands)` + `process_harmonicity_output(...)`
- `rate_entropy_feature(epochs, kmax=10)`
phi core = `github.com/Imperial-MIND-lab/integrated-info-decomp` (the `phyid` package).
Plan: port these bodies into a neurodags custom-node module (adapt mne-Epochs→xarray,
depend on `phyid`), mark experimental / needs-validation. Equivalence check vs the r2c
outputs where available.

## 2. Parity check of the classical battery vs old coco-pipe derivatives

Compare `derivatives_neurodags/` against the reference `derivatives/features@prepDur30Ov20/`:
- **channel selection + naming — FIXED (custom_nodes.py `pick_meg_clean_names`, 2026-07-08).**
  Smoke job 47578952 had computed on ALL channels → 612 "sensors" incl 68 non-MEG (`BG/BP/BR`
  ref coils, `EEG*`, `UPPT` trigger, `SCLK` clock, `HLC`), and all 7956 feature cols had NaNs
  (78k cells) from the varying CTF `-<runid>` suffix mis-aligning sensors across datasets.
  Fix: new custom node picks CTF sensors by name (`^M[LRZ][A-Z]`, since types were mistyped)
  and strips the `-<runid>` suffix; wired as step 1 of `PrepDur30Ov20` (before basic_preprocessing).
  Validated locally end-to-end on junk+suffix synthetic data → only clean MEG sensors, 0 NaN.
  REMAINING: re-run on the cluster with `overwrite: True` (the existing `derivatives_neurodags/`
  from the smoke were computed on the pre-fix prep and are junk-contaminated).
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

**STATUS — RAN successfully on fir 2026-07-08 (all 5 datasets).** Output:
`/home/yorguin/projects/rrg-kjerbi/shared/neuro-cocodelics/bids/MEG_<D>` — ketamine 36/18,
perampanel 40/20, psilocybin 30/15, tiagabine 30/15, LSD 226/19 (6 tasks × 2 ses).
Two deviations from the original plan, forced by real perms:
- Output is under a `bids/` subdir (not `neuro-cocodelics/MEG_<D>`): hamza97 owns those per-dataset
  dirs (group r-x), but `shared/neuro-cocodelics/` itself is group-writable.
- LSD source = `scratch/datasets/cocodelics/MEG_LSDV2/meg_data` (NOT /project): the /project
  `MEG_LSD` raw `.ds` are `drwx--S--- jobyrne` (~90% unreadable to yorguin). **Follow-up:
  PI/jobyrne `chmod -R g+rX /project/def-kjerbi/data/MEG_LSD` to use /project as LSD source.**
Remaining: point the neurodags feature pipeline (`neurodags/datasets_cocodelics.yml`) at this
new shared BIDS; cosmetic script cleanup (dead docstring, `main()` guard). Cluster gotchas hit:
arch-specific wheelhouse numpy (→ install PyPI numpy first) + a bad node fc30557/cold-scratch-read
venv import flakiness (→ pin/retry).

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

## 4. Preprocessing roadmap (settle prep, then a dedicated DFA window)

Current prep (`PrepDur30Ov20`): pick MEG sensors + clean names → notch [50,100,150] → bandpass
0.1–150 → epoch 30 s / 20 s-overlap → resample 600 Hz. Improvements to land:

- **Line-noise: ZapLine instead of the fixed notch.** Use `mne-denoise` (ZapLine / ZapLine-plus,
  `line_freq=None` auto-detect) so line noise (50 vs 60 Hz — currently hardcoded 50/100/150,
  contradicts the `PowerLineFrequency=60` in the datasets yml) is removed adaptively per dataset
  and the signal is better preserved than a notch. Add as a custom node replacing `notch_filter`.
- **Raw-spectrum QC figure.** A node that computes the PSD on the RAW (after MEG-pick, before
  filtering), averages across MEG channels, and saves a `.png` figure artifact — lets us visually
  confirm the true line frequency + data quality per dataset/subject before trusting the prep.
- **THEN — dedicated longer window for DFA (do AFTER the prep above is settled).** DFA is
  window-sensitive (see the DFA discussion): 30 s is short. Plan: (1) once prep is final, measure
  the max feasible epoch/window length **per dataset** (recording durations differ — resting vs
  LSD tasks); (2) pick a **common window across datasets** (the min of the per-dataset maxima, or
  a principled value) so DFA is comparable; (3) add a separate prep (e.g. `prepSingleEpoch` or a
  `prepDurNNN`) feeding only the DFA node, leaving the 30 s prep for the rest of the battery.

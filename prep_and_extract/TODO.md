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
- **normalization**: antropy `spectral_entropy` / `lziv_complexity` default to un-normalized;
  confirm whether coco-pipe normalized and add `normalize: true` to those node args if so.
- **channel selection**: neurodags pipeline computes on ALL channels in the `.fif`; coco-pipe
  likely used the ~271 CTF magnetometers only. Add a `keep_channels`/MEG-pick step if strict.
- **higuchi `kmax`, perm/svd order & delay**: antropy defaults — match to coco-pipe if needed.

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

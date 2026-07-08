#!/bin/bash
# Build the uv venv(s) for the neurodags cocodelics pipeline, on scratch.
#
# Run on the LOGIN node of `fir` (needs internet for the PyPI-only packages).
# `--only-binary :all:` guarantees NO source compilation — every dependency is a
# prebuilt wheel (Alliance wheelhouse for the sci-stack, PyPI for neurodags/
# neurokit2/fooof/biotuner), so this stays a light download-only op, not compute.
# (The one exception is `phyid`, which has no PyPI release and is built from a git
#  source checkout — pure python, no C compilation.)
#
#   ssh fir 'bash -l ~/scratch/cocodelics_neurodags/cluster/build_env.sh'
#
# Builds TWO venvs:
#   $ENV      neurodags              -> numpy>=2. The VALIDATED classical battery.
#   $ENV_EXP  neurodags-experimental -> numpy<2.  The EXPERIMENTAL "scientific"
#                                        features (phi/IIT, Fisher, Harmonicity).
# They are kept separate on purpose: biotuner (needed by the Harmonicity node) pins
# `numpy<2`, which is incompatible with the numpy>=2 classical stack. Run the
# experimental derivatives with $ENV_EXP and the classical ones with $ENV.
set -euo pipefail

module load StdEnv/2023 python/3.11

ENV="/scratch/${USER}/envs/neurodags"
ENV_EXP="/scratch/${USER}/envs/neurodags-experimental"
NRD="/scratch/${USER}/neurodags"

WHEELHOUSE=(
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v4
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic
)
FL=()
for w in "${WHEELHOUSE[@]}"; do FL+=(--find-links "$w"); done

# Sci-stack shared by both envs (numpy is pinned per-env below; neurodags installed
# from the git checkout, editable, in each env).
BASE_PKGS=(
  neurokit2 fooof
  mne mne-bids antropy xarray h5netcdf
  scipy pandas scikit-learn matplotlib
  structlog pydantic joblib tqdm pyyaml
)

# neurodags: git checkout, shared by both envs + EDITABLE install, so `git pull` on the
# cluster updates it instantly (pure python, no reinstall). Public repo -> HTTPS.
if [ ! -d "$NRD/.git" ]; then
  git clone https://github.com/yjmantilla/neurodags.git "$NRD"
fi

# netCDF4 must come from PyPI, NOT the wheelhouse: the +computecanada netCDF4 is
# MPI-linked and imports mpi4py, which only exists on the cvmfs PYTHONPATH. The job
# runs with `unset PYTHONPATH` (so the venv is self-contained and numpy's C-extension
# loads on the compute node), so we need the self-contained PyPI netCDF4 (bundles libs).

# ============================================================================
# 1) MAIN env — numpy>=2 — VALIDATED classical complexity/entropy battery
# ============================================================================
echo "Creating MAIN venv at $ENV (python: $(which python))"
uv venv "$ENV" --python "$(which python)"
uv pip install --python "$ENV/bin/python" --only-binary :all: "${FL[@]}" \
  numpy "${BASE_PKGS[@]}"
uv pip install --python "$ENV/bin/python" --no-deps -e "$NRD"
uv pip install --python "$ENV/bin/python" --only-binary :all: --reinstall-package netcdf4 netCDF4
echo "MAIN ENV READY: $ENV"

# ============================================================================
# 2) EXPERIMENTAL env — numpy<2 — "scientific" features (NEEDS VALIDATION)
# ============================================================================
# Required only by experimental_features.py (single_atoms/atoms_results,
# fisher_information_feature, spectrum_multitaper, feature_harmonicity). SEPARATE from
# the main env because biotuner pins numpy<2.
#
#   phyid      : PhiID integrated-information decomposition (v2 phi/IIT). No PyPI
#                release -> installed from a git SOURCE checkout (pure python, no
#                compilation), hence NOT under `--only-binary`.
#   biotuner   : harmonicity metrics (Tenney height / harmonic similarity / subharmonic
#                tension). This is what forces numpy<2 for the whole experimental env.
#   neurokit2  : Fisher information (already in BASE_PKGS).
echo "Creating EXPERIMENTAL venv at $ENV_EXP (python: $(which python))"
uv venv "$ENV_EXP" --python "$(which python)"
# `numpy<2` up-front so resolution is consistent with biotuner's pin.
uv pip install --python "$ENV_EXP/bin/python" --only-binary :all: "${FL[@]}" \
  "numpy<2" "${BASE_PKGS[@]}" biotuner
uv pip install --python "$ENV_EXP/bin/python" --no-deps -e "$NRD"
uv pip install --python "$ENV_EXP/bin/python" --only-binary :all: --reinstall-package netcdf4 netCDF4
# phyid from git (source build; not on PyPI). Deps (numpy/scipy) already satisfied.
uv pip install --python "$ENV_EXP/bin/python" \
  "phyid @ git+https://github.com/Imperial-MIND-lab/integrated-info-decomp.git"
echo "EXPERIMENTAL ENV READY: $ENV_EXP"

echo "NOTE: run the experimental derivatives (Ep_Atoms/Ep_InfoDyn/Ep_IID/Ep_IIT/"
echo "      Ep_FisherInformation/Ep_PowerSpectrumMultitaper/Ep_Harmonicity + their"
echo "      *MeanEpochs/*SDEpochs) with \$ENV_EXP; the classical battery with \$ENV."
echo "NOTE: the job does 'unset PYTHONPATH' + 'export PYTHONNOUSERSITE=1' before use."
echo "NOTE: import sanity-check (mne/neurodags) runs inside the sbatch job, NOT on login."

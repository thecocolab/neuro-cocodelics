#!/bin/bash
# Build the uv venv for the neurodags cocodelics pipeline, on scratch.
#
# Run on the LOGIN node of `fir` (needs internet for the PyPI/git-only packages).
# `--only-binary :all:` guarantees NO source compilation for the wheel packages —
# every dependency is a prebuilt wheel (Alliance wheelhouse for the sci-stack, PyPI
# for neurokit2/fooof/sympy), so this stays a light download-only op, not compute.
# (The exceptions are `phyid` and editable `neurodags`, built from git source
#  checkouts — both pure python, no C compilation.)
#
#   ssh fir 'bash -l ~/scratch/cocodelics_neurodags/cluster/build_env.sh'
#
# ONE env covers BOTH the validated classical battery AND the experimental
# "scientific" features. The experimental features need no numpy<2 pin: the phi/IIT
# math is `phyid`, Fisher is `neurokit2`, and the Harmonicity metrics are VENDORED
# into experimental_features.py (numpy+sympy only) instead of depending on biotuner
# (which would have dragged in PyEMD/pyACA/mido/… and pinned numpy<2).
set -euo pipefail

module load StdEnv/2023 python/3.11

ENV="/scratch/${USER}/envs/neurodags"
NRD="/scratch/${USER}/neurodags"

WHEELHOUSE=(
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v4
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic
)
FL=()
for w in "${WHEELHOUSE[@]}"; do FL+=(--find-links "$w"); done

# Sci-stack. `sympy` is for the vendored Harmonicity metrics; `neurokit2` for Fisher.
PKGS=(
  neurokit2 fooof sympy
  mne mne-bids antropy xarray h5netcdf
  numpy scipy pandas scikit-learn matplotlib
  structlog pydantic joblib tqdm pyyaml
)

# neurodags: git checkout + EDITABLE install, so `git pull` on the cluster updates it
# instantly (pure python, no reinstall). Public repo -> HTTPS clone needs no auth.
if [ ! -d "$NRD/.git" ]; then
  git clone https://github.com/yjmantilla/neurodags.git "$NRD"
fi

echo "Creating venv at $ENV (python: $(which python))"
uv venv "$ENV" --python "$(which python)"
uv pip install --python "$ENV/bin/python" --only-binary :all: "${FL[@]}" "${PKGS[@]}"
uv pip install --python "$ENV/bin/python" --no-deps -e "$NRD"

# netCDF4 must come from PyPI, NOT the wheelhouse: the +computecanada netCDF4 is
# MPI-linked and imports mpi4py, which only exists on the cvmfs PYTHONPATH. The job
# runs with `unset PYTHONPATH` (so the venv is self-contained and numpy's C-extension
# loads on the compute node), so we need the self-contained PyPI netCDF4 (bundles libs).
uv pip install --python "$ENV/bin/python" --only-binary :all: --reinstall-package netcdf4 netCDF4

# phyid (v2 phi/IIT integrated-information decomposition): no PyPI release -> git source
# build (pure python, no compilation), so NOT under `--only-binary`. numpy-agnostic.
uv pip install --python "$ENV/bin/python" \
  "phyid @ git+https://github.com/Imperial-MIND-lab/integrated-info-decomp.git"

echo "ENV READY: $ENV"
echo "NOTE: the job does 'unset PYTHONPATH' + 'export PYTHONNOUSERSITE=1' before use."
echo "NOTE: import sanity-check (mne/neurodags) runs inside the sbatch job, NOT on login."

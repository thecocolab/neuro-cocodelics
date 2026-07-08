#!/bin/bash
# Build the uv venv for the neurodags cocodelics pipeline, on scratch.
#
# Run on the LOGIN node of `fir` (needs internet for the 3 PyPI-only packages).
# `--only-binary :all:` guarantees NO source compilation — every dependency is a
# prebuilt wheel (Alliance wheelhouse for the sci-stack, PyPI for neurodags/
# neurokit2/fooof), so this stays a light download-only op, not compute.
#
#   ssh fir 'bash -l ~/scratch/cocodelics_neurodags/cluster/build_env.sh'
#
set -euo pipefail

module load StdEnv/2023 python/3.11

ENV="/scratch/${USER}/envs/neurodags"

WHEELHOUSE=(
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v4
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic
)
FL=()
for w in "${WHEELHOUSE[@]}"; do FL+=(--find-links "$w"); done

echo "Creating venv at $ENV (python: $(which python))"
uv venv "$ENV" --python "$(which python)"

# Sci-stack from wheelhouse (via --find-links), the 3 pure-python packages from PyPI.
uv pip install --python "$ENV/bin/python" --only-binary :all: "${FL[@]}" \
  neurodags neurokit2 fooof \
  mne mne-bids antropy xarray h5netcdf \
  numpy scipy pandas scikit-learn matplotlib \
  structlog pydantic joblib tqdm pyyaml

# netCDF4 must come from PyPI, NOT the wheelhouse: the +computecanada netCDF4 is
# MPI-linked and imports mpi4py, which only exists on the cvmfs PYTHONPATH. The job
# runs with `unset PYTHONPATH` (so the venv is self-contained and numpy's C-extension
# loads on the compute node), so we need the self-contained PyPI netCDF4 (bundles libs).
uv pip install --python "$ENV/bin/python" --only-binary :all: --reinstall-package netcdf4 netCDF4

echo "ENV READY: $ENV"
echo "NOTE: the job does 'unset PYTHONPATH' + 'export PYTHONNOUSERSITE=1' before use."
echo "NOTE: import sanity-check (mne/neurodags) runs inside the sbatch job, NOT on login."

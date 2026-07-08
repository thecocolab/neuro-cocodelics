#!/bin/bash
# Build the uv venv for the standalone bidsification scripts, on scratch.
# Run on the LOGIN node of `fir` (needs internet for sovabids git + PyPI wheels).
# Light download-only op (wheels + one pure-python git source), no compilation.
#
#   ssh fir 'bash -l ~/scratch/cocodelics_bids/cluster/build_bids_env.sh'
#
# Deps come straight from the scripts' imports:
#   mne, mne-bids     : read raw CTF .ds / build BIDS (write_raw_bids)
#   sovabids          : parse_from_placeholder (filename -> BIDS entities); git-only
#   scipy             : scipy.io.loadmat (FieldTrip .mat)
#   pandas, numpy     : metadata tables
#   openpyxl          : LSD reads IDs.xlsx / 'LSD Analysis.xlsx' via pandas.read_excel
# numpy-agnostic (no biotuner), so numpy>=2 is fine.
set -euo pipefail

module load StdEnv/2023 python/3.11
ENV="/scratch/${USER}/envs/cocodelics-bids"

WHEELHOUSE=(
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v4
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic
  /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic
)
FL=(); for w in "${WHEELHOUSE[@]}"; do FL+=(--find-links "$w"); done

# On /scratch, uv's hardlink-from-cache can fall back to a partial copy and corrupt a
# package (seen: numpy missing _globals). Force full copies.
export UV_LINK_MODE=copy

uv venv "$ENV" --python "$(which python)"
# numpy FIRST, from PyPI (manylinux, arch-agnostic). The wheelhouse +computecanada numpy is
# built per-microarch (v4/v3) and fails to import on some compute nodes
# ("cannot import name 'set_module' from numpy._utils"); PyPI numpy runs on all nodes.
# Installing it up front means the wheelhouse sci-stack below resolves against it.
uv pip install --python "$ENV/bin/python" --only-binary :all: numpy
# Sci-stack from the wheelhouse (numpy already satisfied above, so not re-pulled).
uv pip install --python "$ENV/bin/python" --only-binary :all: "${FL[@]}" \
  scipy pandas mne mne-bids openpyxl pyyaml
# sovabids: no PyPI release -> git source (pure python). Deps already satisfied above.
uv pip install --python "$ENV/bin/python" --no-deps \
  "sovabids @ git+https://github.com/yjmantilla/sovabids.git"

echo "BIDS ENV READY: $ENV"
echo "NOTE: the sbatch does 'unset PYTHONPATH' + 'export PYTHONNOUSERSITE=1' (numpy C-ext)."

"""EXPERIMENTAL neurodags custom nodes — cocodelics "scientific" MEG features.

    !!! EXPERIMENTAL / NEEDS-VALIDATION !!!

Faithful, computationally-equivalent PORTS of the ``raw_to_classification`` (r2c)
feature implementations from
``github.com/alberto-jj/raw_to_classification`` @ branch ``meeg_refactor``
(file ``eeg_raw_to_classification/features2.py``). Only the *I/O* is adapted:

* input is coerced from a neurodags ``NodeResult`` / path / MNE object (or, for the
  ``atoms_results`` / ``feature_harmonicity`` post-processing steps, an ``xarray``
  ``.nc`` derivative), and
* output is packaged as an ``xarray.DataArray`` inside a ``NodeResult`` ``.nc``
  artifact (with a ``spaces`` channel dim so the standard neurodags dataframe
  flattening — ``<deriv>.nc@spaces-<sensor>`` — works).

The numerical algorithm bodies (atom ordering, ``aggregation_mode='mean-sum'``,
``tau``/``delay``/``dimension``, band definitions, metric dictionaries) are copied
VERBATIM from features2.py.

Validation status (2026-07-08): each node was cross-checked BIT-FOR-BIT against the
actual features2.py functions on synthetic MEG (max relative diff 0.0 for
single_atoms / atoms_results[InformationDynamics, IntegratedInformationDecomposition] /
fisher_information_feature / spectrum_multitaper /
feature_harmonicity+process_harmonicity_output). The ONE intentional deviation is the
IIT ``IntegratedInformation`` metric, where we fix a dead-code guard so the ``rtr``
subtraction fires (see caveat below) — that single column differs from r2c by exactly
the ``rtr`` time-mean; everything else in IIT is identical. These are still marked
**EXPERIMENTAL / NEEDS-VALIDATION** because they have NOT been validated end-to-end
against materialized r2c derivatives (the epoch-aggregation path here uses neurodags
``aggregate_across_dimension`` rather than r2c ``agg_numpy``, and the downstream ML
consumption is unverified). Treat every column they produce as experimental.

Two tiers of "scientific" features are provided:

* **v2 — phi / integrated information** (heavy; depends on ``phyid``):
    - ``single_atoms``      : PhiID atoms per epoch/channel  (r2c ``single_atoms``, tau=5,
                              redundancy='MMI', kind='gaussian').
    - ``atoms_results``     : collapse atoms into InformationDynamics /
                              IntegratedInformationDecomposition / IntegratedInformationTheory
                              metrics (r2c ``atoms_results``, aggregation_mode='mean-sum').
* **v1 — Fisher information & Harmonicity**:
    - ``fisher_information_feature`` : Fisher information per epoch/channel
                              (r2c ``fisher_information_feature``, delay=1, dimension=3;
                              via ``neurokit2``).
    - ``spectrum_multitaper``        : multitaper power spectrum feeding harmonicity
                              (r2c ``spectrum_multitaper``).
    - ``feature_harmonicity``        : Tenney height / harmonic similarity / subharmonic
                              tension of spectral-peak sets (r2c ``feature_harmonicity`` +
                              ``process_harmonicity_output``; via ``biotuner``).

Optional heavy deps (``phyid``, ``neurokit2``, ``biotuner``) are imported LAZILY inside
each node, so this module imports fine even when a dep is absent — only the affected
node raises at call time.

Loaded via ``new_definitions`` (list) in ``pipeline_cocodelics.yml`` alongside
``custom_nodes.py``. Does NOT require any change to neurodags or coco-pipe.

Equivalence caveats (see also the report / README):
* The r2c ``single_atoms`` drops channels whose name does not start with ``'M'``; this
  is now redundant because the upstream ``pick_meg_clean_names`` node already keeps
  MEG-only sensors (all named ``M...``). The filter is kept verbatim but is a no-op for
  the cocodelics data, so it does not change the numeric result.
* In r2c ``atoms_results`` the IntegratedInformationTheory ``rtr`` subtraction is guarded
  by ``if name == "Integrated information"`` while the actual metric key is
  ``"IntegratedInformation"`` — so that branch is DEAD CODE and never fires in r2c. This
  is a bug (the IntegratedInformation metric is meant to subtract the intrinsic ``rtr``
  term). We FIX the guard so the subtraction fires. Consequence: the IIT
  ``IntegratedInformation`` column here differs from r2c by exactly the ``rtr`` time-mean;
  every other atom/metric is bit-for-bit identical to r2c.
"""
from __future__ import annotations

import json
import os
import warnings

from fractions import Fraction
from itertools import combinations as _combinations, product as _product

import numpy as np
import xarray as xr

try:  # mne is a hard dependency of neurodags, but import defensively regardless.
    import mne
    from mne import BaseEpochs
except ImportError:  # pragma: no cover
    mne = None
    BaseEpochs = ()  # type: ignore[assignment]

from neurodags.definitions import Artifact, NodeResult
from neurodags.loaders import load_meeg
from neurodags.loggers import get_logger
from neurodags.nodes import register_node

log = get_logger(__name__)


# ============================================================================
# Constants — copied VERBATIM from features2.py (@meeg_refactor)
# ============================================================================

# features2.py:601 — fixed order of the 16 PhiID atoms.
ATOM_NAMES = [
    "rtr", "rtx", "rty", "rts",
    "xtr", "xtx", "xty", "xts",
    "ytr", "ytx", "yty", "yts",
    "str", "stx", "sty", "sts",
]

# features2.py:545
INFORMATION_DYNAMICS_METRICS = {
    "Storage": ["rtr", "xtx", "yty", "sts"],
    "Copy": ["xtx", "yty"],
    "Transfer": ["xty", "ytx"],
    "Erasure": ["rtx", "rty"],
    "DownwardCausation": ["sty", "stx", "str"],
    "UpwardCausation": ["xts", "yts", "rts"],
}

# features2.py:554
IIT_METRICS = {
    "InformationStorage": ["xtx", "yty", "rtr", "sts"],
    "TransferEntropy": ["xty", "xtr", "str", "sty"],
    "CausalDensity": ["xtr", "ytr", "sty", "str", "str", "xty", "ytx", "stx"],
    "IntegratedInformation": ["rts", "xts", "sts", "sty", "str", "yts", "ytx", "stx", "xty"],
}


# ============================================================================
# Harmonicity metrics — VENDORED from biotuner (AntoineBellemare/biotuner:
# metrics.py + biotuner_utils.py), ported verbatim. Only these 3 metrics
# (+2 helpers) are used by feature_harmonicity. Vendoring them avoids depending
# on biotuner, whose package __init__ drags in a heavy stack (PyEMD/pyACA/mido/
# fooof/…) and pins numpy<2 — none of which these pure number-theory functions
# need (numpy + sympy + stdlib only). Verified bit-for-bit equal to biotuner
# (incl. the coincident-peak divide-by-zero and single-peak NaN paths), 2026-07-08.
# ============================================================================

def _getPairs(peaks):
    """biotuner_utils.getPairs — all ordered pairs (verbatim, incl. pop(i-i))."""
    peaks_ = list(peaks).copy()
    out = []
    for i in range(len(peaks_) - 1):
        a = peaks_.pop(i - i)  # i-i == 0
        for j in peaks_:
            out.append([a, j])
    return out


def _dyad_similarity(ratio):
    """biotuner.metrics.dyad_similarity — Gill & Purves (2009)."""
    frac = Fraction(float(ratio)).limit_denominator(1000)
    x, y = frac.numerator, frac.denominator
    return ((x + y - 1) / (x * y)) * 100


def ratios2harmsim(ratios):
    """biotuner.metrics.ratios2harmsim — harmonic similarity per ratio."""
    fracs = [Fraction(r).limit_denominator(1000) for r in ratios]
    return np.array([_dyad_similarity(f.numerator / f.denominator) for f in fracs])


def integral_tenneyHeight(peaks, avg=True):
    """biotuner.metrics.integral_tenneyHeight — prime-factorised Tenney height."""
    from sympy import factorint  # lazy: only harmonicity needs sympy
    from numpy import log2

    pairs = _getPairs(peaks)
    tenney = []
    for p in pairs:
        try:
            frac = Fraction(p[0] / p[1]).limit_denominator(1000)
            x, y = frac.numerator, frac.denominator
            fx, fy = factorint(x), factorint(y)
            th = sum(fx[k] * log2(k) for k in fx) + sum(fy[k] * log2(k) for k in fy)
            tenney.append(th)
        except ZeroDivisionError:
            continue
    if avg:
        return np.average(tenney) if tenney else 0
    return tenney


def compute_subharmonic_tension(chord, n_harmonics, delta_lim, min_notes=2):
    """biotuner.metrics.compute_subharmonic_tension — Chan et al. (2019)."""
    if not chord or len(chord) < min_notes:
        return [], [], "NaN", []
    subharms = [np.array([1000 / (i / j) for j in range(1, n_harmonics + 1)]) for i in chord]
    combi = np.array(list(_product(*subharms)))
    delta_t, common_subs = [], []
    for group in range(len(combi)):
        for sc in _combinations(combi[group], min_notes):
            if all(np.abs(np.diff(sc)) < delta_lim):
                delta_t.append(np.min(np.abs(np.diff(sc))))
                common_subs.append(np.mean(sc))
    harm_temp, overall_temp, subharm_tension = [], [], []
    if len(delta_t) > 0:
        try:
            for i in range(len(delta_t)):
                delta_norm = delta_t[i] / common_subs[i]
                harm_temp.append(1 / delta_norm)
                overall_temp.append((1 / common_subs[i]) * (delta_t[i]))
            try:
                subharm_tension.append(((sum(overall_temp)) / len(delta_t)))
            except ZeroDivisionError:
                subharm_tension.append("NaN")
        except IndexError:
            subharm_tension = "NaN"
    else:
        subharm_tension = "NaN"
    return common_subs, delta_t, subharm_tension, harm_temp


# ============================================================================
# I/O coercion helpers (the only part NOT ported verbatim)
# ============================================================================

def _as_epochs(mne_object):
    """Coerce a NodeResult(.fif) / path / MNE object into an ``mne`` Epochs/Raw.

    Mirrors the input-coercion pattern in ``custom_nodes.pick_meg_clean_names`` and
    ``neurodags.nodes.descriptive.meeg_to_xarray``.
    """
    if isinstance(mne_object, NodeResult):
        if ".fif" not in mne_object.artifacts:
            raise ValueError("NodeResult does not contain a .fif artifact to process.")
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)
    return mne_object


def _as_dataarray(data):
    """Coerce a NodeResult(.nc) / .nc path / xarray object into a ``DataArray``."""
    if isinstance(data, NodeResult):
        if ".nc" not in data.artifacts:
            raise ValueError("NodeResult does not contain a .nc artifact to process.")
        data = data.artifacts[".nc"].item
    if isinstance(data, (str, os.PathLike)):
        data = xr.load_dataarray(data)
    if isinstance(data, xr.Dataset):
        if len(data.data_vars) != 1:
            raise ValueError("Dataset input must expose exactly one data variable.")
        data = next(iter(data.data_vars.values()))
    if not isinstance(data, xr.DataArray):
        raise ValueError(f"Expected an xarray DataArray, got {type(data).__name__}.")
    return data


def _nc_writer(da):
    return lambda path, arr=da: arr.to_netcdf(path, engine="netcdf4", format="NETCDF4")


def _wrap(da) -> NodeResult:
    return NodeResult(artifacts={".nc": Artifact(item=da, writer=_nc_writer(da))})


# ============================================================================
# v2 — phi / integrated information
# ============================================================================

@register_node
def single_atoms(epochs, tau: int = 5, redundancy: str = "MMI", kind: str = "gaussian") -> NodeResult:
    """EXPERIMENTAL — PhiID "atoms" per epoch / channel (r2c ``single_atoms``).

    NEEDS-VALIDATION. Ported verbatim from raw_to_classification@meeg_refactor
    (features2.py:573). Computes, for every epoch and every channel, the 16 PhiID
    information-decomposition atoms between that channel (source) and the mean of all
    other channels (target), via ``phyid.calculate.calc_PhiID``.

    Parameters
    ----------
    epochs : NodeResult(.fif) | path | mne.Epochs
        Epoched MEG (dims epochs x channels x times). Assumed already MEG-only and
        cleaned by the upstream ``pick_meg_clean_names`` node.
    tau : int
        PhiID time lag (r2c default 5). Output time length is ``n_time - tau``.
    redundancy : str
        Redundancy measure, ``'MMI'`` or ``'CCS'`` (r2c default 'MMI').
    kind : str
        ``'gaussian'`` (continuous) or ``'discrete'`` (r2c default 'gaussian').

    Returns
    -------
    NodeResult
        ``.nc`` DataArray with dims ``(epochs, spaces, atoms, times)``.

    Notes
    -----
    Heavy: PhiID is evaluated ``n_epochs * n_channels`` times. This artifact is large
    (16 atoms x nearly the full time axis) and is cached so the downstream
    ``atoms_results`` metrics reuse it. Depends on ``phyid`` (git+
    https://github.com/Imperial-MIND-lab/integrated-info-decomp.git).
    """
    from phyid.calculate import calc_PhiID  # lazy: heavy optional dep

    epochs = _as_epochs(epochs)

    # --- r2c body (features2.py:576) --------------------------------------
    if isinstance(epochs, BaseEpochs):
        # drop channels not starting with 'M' (r2c cocosprint/cocodelics custom code).
        # NOTE: redundant here — pick_meg_clean_names already kept MEG-only ('M...')
        # sensors, so this keeps every channel. Left verbatim; harmless / no-op.
        idx_to_keep = [i for i, ch in enumerate(epochs.ch_names) if ch.startswith("M")]
        chans_to_keep = [ch for ch in epochs.ch_names if ch.startswith("M")]
        matrix = epochs.get_data()
        matrix = matrix[:, idx_to_keep, :]  # shape (n_epochs, n_channels, n_time)
        channel_labels = chans_to_keep
    else:
        matrix = np.asarray(epochs, dtype=float)
        if matrix.ndim != 3:
            raise ValueError("Input must be a 3D numpy array (epochs x channels x timepoints).")
        channel_labels = None

    n_epochs, n_channels, n_time = matrix.shape

    if channel_labels is None:
        channel_labels = [f"ch{i}" for i in range(n_channels)]

    atom_names = list(ATOM_NAMES)
    n_atoms = len(atom_names)

    atoms_vals = np.zeros((n_epochs, n_channels, n_atoms, n_time - tau), dtype=np.float64)

    # Compute PhiID for each channel vs. the mean of all other channels.
    for e in range(n_epochs):
        data = matrix[e, :, :]  # shape (n_channels, n_time)
        for i in range(n_channels):
            src = data[i]
            if n_channels > 1:
                trg = np.mean(data[np.arange(n_channels) != i], axis=0)
            else:
                trg = src  # single-channel fallback (r2c "Antoine")
            try:
                atoms_res, _ = calc_PhiID(src, trg, tau, kind=kind, redundancy=redundancy)
                for key in atoms_res.keys():
                    vals = atoms_res[key]
                    atoms_vals[e, i, atom_names.index(key), :] = vals  # size n_time - tau
            except Exception as ex:
                log.warning(
                    "single_atoms: error, filling NaN",
                    epoch=e, channel=channel_labels[i], error=str(ex),
                )
                atoms_vals[e, i, :, :] = np.nan
    # ---------------------------------------------------------------------

    sfreq = float(epochs.info["sfreq"]) if isinstance(epochs, BaseEpochs) else 1.0
    time_axis = np.arange(n_time - tau) / sfreq  # seconds (r2c convention)

    da = xr.DataArray(
        atoms_vals,
        dims=("epochs", "spaces", "atoms", "times"),
        coords={
            "epochs": np.arange(n_epochs),
            "spaces": list(channel_labels),
            "atoms": atom_names,
            "times": time_axis,
        },
        name="Atoms",
    )
    da.attrs["metadata"] = json.dumps(
        {
            "type": "Atoms",
            "experimental": True,
            "ported_from": "raw_to_classification@meeg_refactor:single_atoms",
            "validated_against_r2c": False,
            "tau": int(tau),
            "redundancy": redundancy,
            "kind": kind,
        }
    )
    return _wrap(da)


@register_node
def atoms_results(
    atoms,
    key: str = "InformationDynamics",
    aggregation_mode: str = "mean-sum",
) -> NodeResult:
    """EXPERIMENTAL — collapse PhiID atoms into named metrics (r2c ``atoms_results``).

    NEEDS-VALIDATION. Ported verbatim from raw_to_classification@meeg_refactor
    (features2.py:687). Consumes the ``single_atoms`` output and produces, per epoch and
    channel, the metrics for one of three views:

    * ``key='InformationDynamics'``               -> Storage/Copy/Transfer/Erasure/
                                                      DownwardCausation/UpwardCausation
    * ``key='IntegratedInformationDecomposition'`` -> the 16 raw atoms (time-averaged)
    * ``key='IntegratedInformationTheory'``        -> InformationStorage/TransferEntropy/
                                                      CausalDensity/IntegratedInformation

    ``aggregation_mode='mean-sum'`` (r2c default): mean each selected atom over time
    first, then sum the means. ``'sum-mean'``: sum atoms per timepoint, then mean over
    time.

    Parameters
    ----------
    atoms : NodeResult(.nc) | .nc path | xarray.DataArray
        Output of :func:`single_atoms`, dims ``(epochs, spaces, atoms, times)``.
    key : str
        Which metric view to compute (see above).
    aggregation_mode : str
        ``'mean-sum'`` (default, matches r2c) or ``'sum-mean'``.

    Returns
    -------
    NodeResult
        ``.nc`` DataArray with dims ``(epochs, spaces, metrics)``.
    """
    da = _as_dataarray(atoms)

    # DataArray dims (epochs, spaces, atoms, times); atom order from the 'atoms' coord.
    atom_names = [str(a) for a in np.asarray(da.coords["atoms"].values)]
    values_full = np.asarray(da.transpose("epochs", "spaces", "atoms", "times").values)
    n_epochs, n_channels, _n_atoms, _n_time = values_full.shape
    space_names = [str(s) for s in np.asarray(da.coords["spaces"].values)]

    # mean of each atom over time == r2c ``atoms['values'][e, c, atom_idx, :].mean()``.
    atom_time_means = values_full.mean(axis=3)  # (epochs, spaces, atoms)

    if key == "InformationDynamics":
        metric_names = list(INFORMATION_DYNAMICS_METRICS.keys())
    elif key == "IntegratedInformationDecomposition":
        metric_names = list(atom_names)
    elif key == "IntegratedInformationTheory":
        metric_names = list(IIT_METRICS.keys())
    else:
        raise ValueError(f"Unknown key '{key}'.")

    the_final_vals = np.zeros((n_epochs, n_channels, len(metric_names)), dtype=np.float64)

    for e in range(n_epochs):
        for c in range(n_channels):
            for j, name in enumerate(metric_names):
                values = values_full[e, c, :, :]  # (atoms, times)

                if key == "InformationDynamics":
                    atom_indices = [atom_names.index(atom) for atom in INFORMATION_DYNAMICS_METRICS[name]]
                    if aggregation_mode == "sum-mean":
                        the_final_vals[e, c, j] = float(np.mean(np.sum(values[atom_indices, :], axis=0)))
                    elif aggregation_mode == "mean-sum":
                        the_final_vals[e, c, j] = float(np.sum([atom_time_means[e, c, ai] for ai in atom_indices]))
                elif key == "IntegratedInformationDecomposition":
                    atom_index = atom_names.index(name)
                    if aggregation_mode == "sum-mean":
                        the_final_vals[e, c, j] = float(np.mean(values[atom_index, :]))
                    elif aggregation_mode == "mean-sum":
                        the_final_vals[e, c, j] = float(atom_time_means[e, c, atom_index])
                elif key == "IntegratedInformationTheory":
                    atom_indices = [atom_names.index(atom) for atom in IIT_METRICS[name]]
                    if aggregation_mode == "sum-mean":
                        the_final_vals[e, c, j] = float(np.mean(np.sum(values[atom_indices, :], axis=0)))
                    elif aggregation_mode == "mean-sum":
                        the_final_vals[e, c, j] = float(np.sum([atom_time_means[e, c, ai] for ai in atom_indices]))
                    # NOTE: intentional DEVIATION from r2c. In r2c the metric key is
                    # "IntegratedInformation" (no space) but the guard checks
                    # "Integrated information" (space, lowercase i), so the rtr subtraction
                    # is DEAD CODE that never runs there. That is a bug: the IntegratedInformation
                    # metric is meant to subtract the intrinsic rtr term. We fix the guard so the
                    # subtraction actually fires. This makes the IIT "IntegratedInformation" column
                    # differ from r2c by exactly the rtr time-mean; every other metric is unchanged.
                    if name == "IntegratedInformation":
                        rtr_index = atom_names.index("rtr")
                        if aggregation_mode == "sum-mean":
                            the_final_vals[e, c, j] -= float(np.mean(values[rtr_index, :]))
                        elif aggregation_mode == "mean-sum":
                            the_final_vals[e, c, j] -= float(atom_time_means[e, c, rtr_index])

    out = xr.DataArray(
        the_final_vals,
        dims=("epochs", "spaces", "metrics"),
        coords={
            "epochs": np.arange(n_epochs),
            "spaces": space_names,
            "metrics": metric_names,
        },
        name=key,
    )
    out.attrs["metadata"] = json.dumps(
        {
            "type": key,
            "experimental": True,
            "ported_from": "raw_to_classification@meeg_refactor:atoms_results",
            "validated_against_r2c": False,
            "aggregation_mode": aggregation_mode,
        }
    )
    return _wrap(out)


# ============================================================================
# v1 — Fisher information
# ============================================================================

@register_node
def fisher_information_feature(epochs, delay: int = 1, dimension: int = 3) -> NodeResult:
    """EXPERIMENTAL — Fisher information per epoch / channel (r2c ``fisher_information_feature``).

    NEEDS-VALIDATION. Ported verbatim from raw_to_classification@meeg_refactor
    (features2.py:1279). For each epoch and channel, computes
    ``neurokit2.fisher_information(ts, delay=delay, dimension=dimension)``.

    Parameters
    ----------
    epochs : NodeResult(.fif) | path | mne.Epochs
        Epoched MEG (dims epochs x channels x times).
    delay : int
        Embedding delay (r2c default 1).
    dimension : int
        Embedding dimension (r2c default 3).

    Returns
    -------
    NodeResult
        ``.nc`` DataArray with dims ``(epochs, spaces)``.
    """
    import neurokit2 as nk2  # lazy optional dep

    epochs = _as_epochs(epochs)
    epochs = epochs.copy()
    space_names = list(epochs.info["ch_names"])
    data = epochs.get_data()
    n_epochs, n_channels, _ = data.shape
    values = np.empty((n_epochs, n_channels))
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            ts = data[epoch_idx, ch_idx, :]
            feature_val, _ = nk2.fisher_information(ts, delay=delay, dimension=dimension)
            values[epoch_idx, ch_idx] = feature_val

    da = xr.DataArray(
        values,
        dims=("epochs", "spaces"),
        coords={"epochs": np.arange(n_epochs), "spaces": space_names},
        name="FisherInformation",
    )
    da.attrs["metadata"] = json.dumps(
        {
            "type": "FisherInformation",
            "experimental": True,
            "ported_from": "raw_to_classification@meeg_refactor:fisher_information_feature",
            "validated_against_r2c": False,
            "delay": int(delay),
            "dimension": int(dimension),
        }
    )
    return _wrap(da)


# ============================================================================
# v1 — Power spectrum (feeds harmonicity) + Harmonicity
# ============================================================================

@register_node
def spectrum_multitaper(epochs, multitaper: dict | None = None) -> NodeResult:
    """EXPERIMENTAL — multitaper power spectrum (r2c ``spectrum_multitaper``).

    NEEDS-VALIDATION. Ported verbatim from raw_to_classification@meeg_refactor
    (features2.py:172). Computes ``mne.time_frequency.psd_array_multitaper`` per epoch
    and channel. In the r2c chain this ``PowerSpectrum`` feeds ``feature_harmonicity``.

    Parameters
    ----------
    epochs : NodeResult(.fif) | path | mne.Epochs
        Epoched MEG (dims epochs x channels x times).
    multitaper : dict, optional
        Keyword args forwarded to ``psd_array_multitaper``. Default matches the r2c
        ``PowerSpectrum`` config:
        ``{adaptive: False, low_bias: True, normalization: 'full', verbose: 0}``.

    Returns
    -------
    NodeResult
        ``.nc`` DataArray with dims ``(epochs, spaces, frequencies)``.
    """
    from mne.time_frequency import psd_array_multitaper

    epochs = _as_epochs(epochs)
    epochs = epochs.copy()
    sf = epochs.info["sfreq"]

    if multitaper is None:
        multitaper = {"adaptive": False, "low_bias": True, "normalization": "full", "verbose": 0}
    kwargs = dict(multitaper)

    space_names = list(epochs.info["ch_names"])
    psd, freqs = psd_array_multitaper(epochs.get_data(), sf, **kwargs)
    fullpsd = psd  # (epochs, spaces, frequencies)

    da = xr.DataArray(
        fullpsd,
        dims=("epochs", "spaces", "frequencies"),
        coords={
            "epochs": np.arange(fullpsd.shape[0]),
            "spaces": space_names,
            "frequencies": np.asarray(freqs),
        },
        name="PowerSpectrum",
    )
    da.attrs["metadata"] = json.dumps(
        {
            "type": "PowerSpectrum",
            "experimental": True,
            "ported_from": "raw_to_classification@meeg_refactor:spectrum_multitaper",
            "validated_against_r2c": False,
        }
    )
    return _wrap(da)


@register_node
def feature_harmonicity(
    power_spectrum,
    height=None,
    distance=None,
    bands=None,
) -> NodeResult:
    """EXPERIMENTAL — harmonicity metrics of spectral peaks (r2c ``feature_harmonicity``).

    NEEDS-VALIDATION. Ported verbatim from raw_to_classification@meeg_refactor
    (features2.py:1364, folding in ``process_harmonicity_output`` at :1442). Given a
    per-epoch/channel power spectrum, it finds the dominant peak in each frequency band,
    then computes three harmonicity metrics over the set of band peaks via ``biotuner``:

    * ``tenney``          : integral Tenney height (``integral_tenneyHeight``)
    * ``harmsim``         : mean harmonic similarity (``ratios2harmsim``)
    * ``subharm_tension`` : subharmonic tension (``compute_subharmonic_tension``)

    Combines ``feature_harmonicity`` + ``process_harmonicity_output`` (which in r2c drop
    the band axis and expose only the metrics array) into one node.

    Parameters
    ----------
    power_spectrum : NodeResult(.nc) | .nc path | xarray.DataArray
        Output of :func:`spectrum_multitaper`, dims ``(epochs, spaces, frequencies)``.
    height, distance : float | None
        ``scipy.signal.find_peaks`` args (r2c default None/None).
    bands : list[[name, [low, high]]] | None
        Frequency bands. Default (r2c) delta(2-4)/theta(4-8)/alpha(8-12)/beta(12-30)/
        gamma(30-60).

    Returns
    -------
    NodeResult
        ``.nc`` DataArray with dims ``(epochs, spaces, metrics)``,
        metrics = ['tenney', 'harmsim', 'subharm_tension'].

    Warnings
    --------
    ``biotuner.metrics.compute_subharmonic_tension`` emits ``RuntimeWarning: divide by
    zero`` internally (``1 / delta_norm`` when two peaks coincide / a harmonic delta is
    zero). This is benign — it still returns a valid ``subharm_tension`` — and is the
    same behaviour as r2c. These warnings are suppressed inside this node to keep the
    pipeline logs clean; the numeric result is unaffected.
    """
    from scipy.signal import find_peaks
    # harmonicity metrics are vendored above (no biotuner dep)

    da = _as_dataarray(power_spectrum).transpose("epochs", "spaces", "frequencies")
    psds = np.asarray(da.values)
    freqs = np.asarray(da.coords["frequencies"].values)
    space_names = [str(s) for s in np.asarray(da.coords["spaces"].values)]

    # --- r2c body (features2.py:1369) -------------------------------------
    if bands is None:
        bands = [
            ("delta", (2, 4)),
            ("theta", (4, 8)),
            ("alpha", (8, 12)),
            ("beta", (12, 30)),
            ("gamma", (30, 60)),
        ]
    # YAML may deliver bands as [name, [low, high]] pairs; normalise to (name, (low, high)).
    bands = [(str(name), (float(rng[0]), float(rng[1]))) for name, rng in bands]

    n_epochs, n_channels, _n_bins = psds.shape
    nbands = len(bands)
    n_features = 3  # Tenney, HarmSim, Subharmonic Tension
    metrics_list = ["tenney", "harmsim", "subharm_tension"]

    max_peaks = np.full((n_epochs, n_channels, nbands), np.nan)
    metrics = np.full((n_epochs, n_channels, n_features), np.nan)

    with warnings.catch_warnings():
        # biotuner's subharmonic-tension emits harmless divide-by-zero RuntimeWarnings.
        warnings.simplefilter("ignore", RuntimeWarning)
        for ep in range(n_epochs):
            for ch in range(n_channels):
                peaks_list = []
                for band_idx, (band_name, (low, high)) in enumerate(bands):
                    mask = (freqs >= low) & (freqs <= high)
                    band_psd = psds[ep, ch, mask]
                    band_freqs = freqs[mask]
                    if len(band_psd) > 0:
                        peaks, _ = find_peaks(band_psd, height=height, distance=distance)
                        if len(peaks) > 0:
                            max_peak_idx = peaks[np.argmax(band_psd[peaks])]
                            max_peaks[ep, ch, band_idx] = band_freqs[max_peak_idx]
                            peaks_list.append(band_freqs[max_peak_idx])
                        else:
                            peaks_list.append(np.nan)
                    else:
                        peaks_list.append(np.nan)
                valid_peaks = [p for p in peaks_list if not np.isnan(p)]
                valid_peaks = list(np.round(valid_peaks, 1))
                if len(valid_peaks) > 0:
                    try:
                        _, _, subharm, _ = compute_subharmonic_tension(
                            valid_peaks, n_harmonics=3, delta_lim=50
                        )
                        tenney = integral_tenneyHeight(valid_peaks)
                        harmsim = np.mean(ratios2harmsim(valid_peaks))
                        metrics[ep, ch, 0] = tenney
                        metrics[ep, ch, 1] = harmsim
                        metrics[ep, ch, 2] = subharm[0]
                    except Exception as ex:
                        log.warning(
                            "feature_harmonicity: metric error, leaving NaN",
                            epoch=ep, channel=ch, peaks=str(valid_peaks), error=str(ex),
                        )
    # ---------------------------------------------------------------------

    out = xr.DataArray(
        metrics,
        dims=("epochs", "spaces", "metrics"),
        coords={
            "epochs": np.arange(n_epochs),
            "spaces": space_names,
            "metrics": metrics_list,
        },
        name="Harmonicity",
    )
    out.attrs["metadata"] = json.dumps(
        {
            "type": "Harmonicity",
            "experimental": True,
            "ported_from": "raw_to_classification@meeg_refactor:feature_harmonicity+process_harmonicity_output",
            "validated_against_r2c": False,
            "bands": [[n, [lo, hi]] for n, (lo, hi) in bands],
            "height": height,
            "distance": distance,
        }
    )
    return _wrap(out)


__all__ = [
    "single_atoms",
    "atoms_results",
    "fisher_information_feature",
    "spectrum_multitaper",
    "feature_harmonicity",
]

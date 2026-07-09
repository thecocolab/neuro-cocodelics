"""Custom neurodags nodes for the cocodelics pipeline.

Loaded via `new_definitions: custom_nodes.py` in pipeline_cocodelics.yml.
These do NOT require any change to neurodags itself.
"""
from __future__ import annotations

import os
import re

from neurodags.definitions import Artifact, NodeResult
from neurodags.loaders import load_meeg
from neurodags.nodes import register_node


def _load_mne(mne_object):
    """Coerce a NodeResult(.fif) / path / MNE object to an MNE object."""
    if isinstance(mne_object, NodeResult):
        if ".fif" not in mne_object.artifacts:
            raise ValueError("NodeResult does not contain a .fif artifact to process.")
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)
    return mne_object


@register_node
def pick_meg_clean_names(
    mne_object,
    keep_regex: str = r"^M[LRZ][A-Z]",
    strip_suffix: str = r"-\d+$",
    save: bool = False,
) -> NodeResult:
    """Pick CTF MEG data sensors and normalize their names.

    Fixes two issues seen in the bidsified cocodelics data (smoke job 47578952):

    1. Non-data channels (reference coils ``BG*/BP*/BR*``, ``EEG*``, trigger ``UPPT*``,
       clock ``SCLK*``, head-loc ``HLC*``, …) get carried into feature extraction,
       producing junk columns and divide-by-zero NaNs. We keep only channels whose
       name matches ``keep_regex`` — default ``^M[LRZ][A-Z]`` = the CTF sensor pattern
       (side L/R/Z + region letter, e.g. MLC11, MRF67, MZO01). Picking by NAME rather
       than MNE channel *type* is deliberate: the bidsification mistyped some reference
       channels as grad/mag, so type-based picking would not exclude them.

    2. CTF channel names carry a per-acquisition ``-<runid>`` suffix (``MLC11-3305``)
       that varies across datasets, so the same sensor mis-aligns when datasets are
       concatenated. We strip the trailing ``strip_suffix`` (default ``-<digits>``) →
       ``MLC11``, matching the ML sensor list. Mirrors viz/plot_functions.py's
       ``rename_channels(lambda x: x.replace("-3305", ""))``.

    Parameters
    ----------
    mne_object : path | NodeResult | mne Raw/Epochs
        Input recording (typically the SourceFile).
    keep_regex : str
        Only channels whose name matches this (via ``re.match``) are kept.
    strip_suffix : str
        Regex removed from the end of each kept channel name (``""`` to disable).
    save : bool
        Persist the picked/renamed .fif. Default False (intermediate, passed in-memory).
    """
    if isinstance(mne_object, NodeResult):
        if ".fif" not in mne_object.artifacts:
            raise ValueError("NodeResult does not contain a .fif artifact to process.")
        mne_object = mne_object.artifacts[".fif"].item
    if isinstance(mne_object, (str, os.PathLike)):
        mne_object = load_meeg(mne_object)
    mne_object = mne_object.copy()

    keep = [ch for ch in mne_object.ch_names if re.match(keep_regex, ch)]
    if not keep:
        raise ValueError(
            f"pick_meg_clean_names: no channels matched keep_regex={keep_regex!r}; "
            f"first channels: {mne_object.ch_names[:8]}"
        )
    mne_object.pick(keep)

    if strip_suffix:
        mapping = {ch: re.sub(strip_suffix, "", ch) for ch in mne_object.ch_names}
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("pick_meg_clean_names: channel-name collision after stripping suffix.")
        mne_object.rename_channels(mapping)

    writer = (lambda path: mne_object.save(path, overwrite=True)) if save else None
    return NodeResult(artifacts={".fif": Artifact(item=mne_object, writer=writer)})


@register_node
def zapline_denoise(
    mne_object,
    line_freq: float = 60.0,
    n_remove="auto",
    n_harmonics=None,
    threshold: float = 3.0,
    adaptive: bool = False,
    save: bool = False,
) -> NodeResult:
    """Remove power-line noise with ZapLine (de Cheveigné 2020) via ``mne-denoise``.

    Preferred over a fixed notch: ZapLine removes the line component (and, with
    ``n_harmonics``, its harmonics) via DSS while preserving neural signal at the line
    frequency, instead of gouging a notch. Applied to the CONTINUOUS (pre-epoch) recording.

    Parameters
    ----------
    mne_object : path | NodeResult | mne Raw
        Continuous recording (typically the MEG-picked raw, before bandpass/epoch).
    line_freq : float
        Line frequency in Hz (cocodelics datasets list 60). Ignored when ``adaptive=True``
        (ZapLine-plus auto-detects). Use the raw-spectrum QC figure to confirm 50 vs 60.
    n_remove : int | 'auto'
        Number of line-noise components to project out ('auto' → z-score/``threshold``).
    n_harmonics : int | None
        Also clean this many harmonics of ``line_freq`` (None = fundamental only).
    threshold : float
        Z-score threshold for the 'auto' component count.
    adaptive : bool
        ZapLine-plus adaptive mode (auto line-frequency + chunking).
    save : bool
        Persist the cleaned .fif. Default False (intermediate, passed in-memory).
    """
    from mne_denoise.zapline import ZapLine  # lazy: optional prep dep
    import mne
    import numpy as np

    raw = _load_mne(mne_object).copy()
    data = raw.get_data()  # (n_channels, n_times)
    # MEG in Tesla is ~1e-13, so covariances (~1e-26) underflow ZapLine's reg (1e-9) ->
    # "Covariance matrix has no significant variance". ZapLine is a linear projector, so
    # scaling the data by a scalar in and back out is exact — do it to avoid the underflow.
    scale = float(np.std(data)) or 1.0
    est = ZapLine(
        sfreq=float(raw.info["sfreq"]),
        line_freq=line_freq,
        n_remove=n_remove,
        n_harmonics=n_harmonics,
        threshold=threshold,
        adaptive=adaptive,
    )
    cleaned = est.fit_transform(data / scale) * scale  # (n_channels, n_times)
    out = mne.io.RawArray(cleaned, raw.info, verbose="error")  # preserves ch names/types/sfreq

    writer = (lambda path: out.save(path, overwrite=True)) if save else None
    return NodeResult(artifacts={".fif": Artifact(item=out, writer=writer)})


@register_node
def meg_spectrum_fig(
    mne_object,
    fmin: float = 1.0,
    fmax=None,
    method: str = "welch",
    title: str = "MEG spectrum",
    save: bool = True,
) -> NodeResult:
    """QC: PSD of MEG data as a .png figure — mean across channels + per-channel spread.

    Works on a Raw or Epochs `.fif`. Use it BEFORE prep (raw, MEG-picked) to eyeball the true
    line frequency (50/60 Hz markers drawn) + data quality, and AFTER prep to confirm the
    ZapLine line removal + bandpass shape. Not a feature — a `.png` inspection artifact.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    raw = _load_mne(mne_object)
    sf = float(raw.info["sfreq"])
    if fmax is None:
        fmax = min(120.0, sf / 2.0 - 1.0)
    psd = raw.compute_psd(method=method, fmin=fmin, fmax=fmax, verbose="error")
    psds, freqs = psd.get_data(return_freqs=True)
    if psds.ndim == 3:  # Epochs input: (n_epochs, n_channels, n_freqs) -> average over epochs
        psds = psds.mean(axis=0)
    mean_psd = psds.mean(axis=0)  # -> mean across channels

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(freqs, psds.T, color="0.8", lw=0.4)
    ax.semilogy(freqs, mean_psd, color="C0", lw=1.8, label=f"mean of {psds.shape[0]} MEG ch")
    for lf, col in ((50.0, "tab:red"), (60.0, "tab:purple")):
        if fmin < lf < fmax:
            ax.axvline(lf, color=col, ls="--", lw=0.9, alpha=0.7, label=f"{lf:g} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (T²/Hz)")
    ax.set_title(f"{title} — {psds.shape[0]} MEG ch, sfreq {sf:g} Hz")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    def writer(path, _fig=fig):
        _fig.savefig(path, dpi=110)
        plt.close(_fig)

    return NodeResult(artifacts={".png": Artifact(item=fig, writer=writer if save else None)})

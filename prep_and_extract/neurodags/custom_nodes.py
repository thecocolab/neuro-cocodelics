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
    # Pick MEG sensors by NAME (^M[LRZ][A-Z]) + force compute_psd to use them regardless of
    # channel *type* — some FieldTrip-bidsified data mis-types MEG channels as 'misc', which the
    # default (data-type) picking would drop, showing far too few channels.
    meg_picks = [c for c in raw.ch_names if re.match(r"^M[LRZ][A-Z]", c)]
    psd = raw.compute_psd(
        method=method, fmin=fmin, fmax=fmax,
        picks=(meg_picks if meg_picks else "data"), verbose="error",
    )
    psds, freqs = psd.get_data(picks="all", return_freqs=True)  # 'all' -> not dropped by type
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


@register_node
def meg_report_qc(
    raw,
    epochs,
    title: str = "MEG QC",
    fmin: float = 1.0,
    fmax=None,
    psd_seconds: float = 60.0,
    save: bool = True,
) -> NodeResult:
    """Per-recording QC as a self-contained ``mne.Report`` HTML (raw vs prepped).

    Replaces the standalone PSD ``.png`` figures with a richer, standard MEG-QC HTML:
    a custom mean-MEG-PSD overlay (RAW pre-denoise vs PREPPED post-ZapLine+bandpass,
    50/100 Hz marked — the clearest view of the line removal), plus mne.Report's native
    per-channel PSD for the raw and the prepped epochs. Two derivative inputs:
    ``raw`` (PickedRaw) + ``epochs`` (PrepDur30Ov20).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import mne

    raw = _load_mne(raw)
    ep = _load_mne(epochs)

    def mean_meg_psd(obj):
        picks = [c for c in obj.ch_names if re.match(r"^M[LRZ][A-Z]", c)]
        sf = float(obj.info["sfreq"])
        fmx = fmax if fmax is not None else min(120.0, sf / 2.0 - 1.0)
        psd = obj.compute_psd(method="welch", fmin=fmin, fmax=fmx,
                              picks=(picks if picks else "data"), verbose="error")
        p, f = psd.get_data(picks="all", return_freqs=True)
        if p.ndim == 3:          # Epochs -> average over epochs
            p = p.mean(axis=0)
        return f, p.mean(axis=0), p.shape[0]

    fr, mr, nr = mean_meg_psd(raw)
    fe, me, ne = mean_meg_psd(ep)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(fr, mr, lw=1.6, color="0.5", label=f"raw pre-denoise ({nr} ch)")
    ax.semilogy(fe, me, lw=1.6, color="tab:red", label=f"prepped post-ZapLine ({ne} ch)")
    for lf, c in ((50.0, "tab:red"), (100.0, "tab:orange")):
        ax.axvline(lf, ls="--", lw=0.9, alpha=0.6, color=c, label=f"{lf:g} Hz")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (T²/Hz)")
    ax.set_title("Mean MEG PSD — raw vs prepped (ZapLine line removal)")
    ax.legend(fontsize=8, loc="upper right"); fig.tight_layout()

    rep = mne.Report(title=title, verbose="error")
    rep.add_figure(fig, title="Mean MEG PSD: raw vs prepped", section="Spectrum",
                   caption="Grey = raw (pre-denoise); red = prepped (ZapLine adaptive + n_harmonics=2, "
                           "bandpass 0.1–150, 30 s epochs). Dashed = 50/100 Hz line + harmonic.")
    plt.close(fig)
    rep.add_raw(raw, title="Raw picked (pre-denoise)", psd=psd_seconds, butterfly=False, projs=False)
    rep.add_epochs(ep, title="Prepped epochs (ZapLine + bandpass 0.1–150, 30 s)", psd=True, projs=False)

    def writer(path, _r=rep):
        _r.save(path, overwrite=True, open_browser=False)

    return NodeResult(artifacts={".html": Artifact(item=rep, writer=writer if save else None)})


def _dfa_exponent(x, boxes):
    """Detrended-fluctuation exponent (Peng 1994) of a 1-D series over given box sizes.

    Integrate the mean-removed signal, then for each box size ``n`` split into
    non-overlapping windows, linearly detrend each (vectorized via lstsq), take the
    RMS of residuals, and the fluctuation ``F(n)`` = RMS across windows. The exponent
    is the slope of ``log F(n)`` vs ``log n`` over the supplied boxes (the fit range).
    """
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.cumsum(x - x.mean())
    n_t = y.size
    logn, logF = [], []
    for n in boxes:
        n = int(n)
        if n < 4 or n > n_t // 2:
            continue
        nseg = n_t // n
        Y = y[: nseg * n].reshape(nseg, n)               # (nseg, n)
        t = np.arange(n)
        A = np.vstack([t, np.ones(n)]).T                 # (n, 2)
        coef, *_ = np.linalg.lstsq(A, Y.T, rcond=None)   # (2, nseg) linear fit per window
        resid = Y - (A @ coef).T                         # (nseg, n)
        F = np.sqrt(np.mean(resid ** 2))                 # fluctuation across all windows
        logn.append(np.log(n))
        logF.append(np.log(F))
    if len(logn) < 2:
        return float("nan")
    return float(np.polyfit(logn, logF, 1)[0])


@register_node
def alpha_envelope_dfa(
    mne_object,
    l_freq: float = 8.0,
    h_freq: float = 12.0,
    window_s: float = 240.0,
    crop_start_s: float = 2.0,
    env_sfreq: float = 100.0,
    dfa_lower_s: float = 1.0,
    n_boxes: int = 20,
    save: bool = True,
) -> NodeResult:
    """Alpha-band amplitude-envelope DFA on a single long continuous window.

    Canonical neural long-range-temporal-correlation (LRTC) DFA (Hardstone et al.
    2012): band-pass to alpha → Hilbert amplitude envelope → DFA on ONE continuous
    window, with the fluctuation fit restricted to box sizes in
    ``[dfa_lower_s, window_s/10]`` (Hardstone's upper bound = signal_length/10; the
    lower bound of a few seconds avoids the short-scale bias of the narrow-band
    filtered envelope). One exponent per MEG channel.

    A single ``window_s`` window (default 240 s = the cross-dataset common floor; the
    shortest recording across datasets is 244 s — see RECORDING_DURATIONS.md) is used
    instead of many short epochs so the exponent is comparable across datasets and
    the analysed segment is genuinely continuous. The envelope is downsampled to
    ``env_sfreq`` (its bandwidth is only a few Hz) purely for DFA speed.

    NEEDS VALIDATION: the fit range (``dfa_lower_s``, ``window_s/10``), envelope
    downsample, and edge pad (``crop_start_s``) are principled defaults, not yet
    tuned against a reference implementation (e.g. NBT).

    Returns a ``.nc`` DataArray with dims ``(epochs, spaces)`` (epochs=1) so it feeds
    the standard ``aggregate_across_dimension`` → for_dataframe path unchanged.
    """
    import numpy as np
    import xarray as xr

    raw = _load_mne(mne_object).copy()
    meg = [c for c in raw.ch_names if re.match(r"^M[LRZ][A-Z]", c)]
    if meg:
        raw.pick(meg)
    raw.filter(l_freq, h_freq, verbose="error")            # alpha band
    raw.apply_hilbert(envelope=True, verbose="error")      # data -> amplitude envelope
    if env_sfreq and env_sfreq < float(raw.info["sfreq"]):
        raw.resample(env_sfreq, verbose="error")
    sf = float(raw.info["sfreq"])

    data = raw.get_data()                                  # (n_ch, n_times) envelope
    start = int(round(crop_start_s * sf))
    length = int(round(window_s * sf))
    if data.shape[1] < start + length:                     # shorter than pad+window
        start = max(0, data.shape[1] - length)             # fall back to last `length`
    seg = data[:, start:start + length]

    lo = max(4, int(round(dfa_lower_s * sf)))
    hi = max(lo + 1, int(round((window_s / 10.0) * sf)))   # Hardstone upper bound = len/10
    boxes = np.unique(np.round(np.logspace(np.log10(lo), np.log10(hi), n_boxes)).astype(int))
    exps = np.array([_dfa_exponent(seg[i], boxes) for i in range(seg.shape[0])])

    da = xr.DataArray(
        exps[None, :], dims=("epochs", "spaces"),
        coords={"spaces": list(raw.ch_names)}, name="alphaEnvelopeDfa",
    )
    writer = (lambda path, arr=da: arr.to_netcdf(path, engine="netcdf4", format="NETCDF4")) if save else None
    return NodeResult(artifacts={".nc": Artifact(item=da, writer=writer)})

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
def notch_denoise(
    mne_object,
    freqs=(50.0, 100.0, 150.0),
    save: bool = False,
) -> NodeResult:
    """Remove power-line noise with a notch filter (classic r2c approach).

    Chosen over ZapLine (2026-07-11): ZapLine adaptive only reduced the 50 Hz line ~40%
    (verified — see git history / the A/B + sweep), leaving a visible residual peak; the
    notch removes the line + harmonics essentially completely, at the cost of gouging a
    narrow band around each ``freqs`` entry. Applied to the CONTINUOUS raw so both the
    30 s battery prep and the alpha-envelope DFA branch share one denoised source.
    (The alpha band 8–12 Hz is untouched by a 50/100/150 notch.)
    """
    raw = _load_mne(mne_object).copy()
    raw.notch_filter(list(freqs), verbose="error")
    writer = (lambda path: raw.save(path, overwrite=True)) if save else None
    return NodeResult(artifacts={".fif": Artifact(item=raw, writer=writer)})


@register_node
def meg_report_qc(
    raw,
    denoised,
    epochs,
    title: str = "MEG QC",
    denoise_desc: str = "denoise",
    fmin: float = 1.0,
    fmax=None,
    win_s: float = 2.0,
    psd_seconds: float = 60.0,
    save: bool = True,
) -> NodeResult:
    """Per-recording QC as a self-contained ``mne.Report`` HTML.

    The primary figure overlays the ZapLine before/after at MATCHED spectral resolution:
    ``PickedRaw`` (pre-denoise) vs ``DenoisedRaw`` (post-ZapLine) — both continuous, same
    Welch window (``win_s``), so the peak heights ARE comparable and the true line
    reduction is visible (an earlier version compared raw-continuous vs prepped-epochs at
    different resolutions, which hid the effect). The final ``PrepDur30Ov20`` (post
    bandpass+resample+epoch) is drawn as a third reference curve. Plus mne.Report's native
    per-channel PSD. Three derivative inputs: ``raw`` + ``denoised`` + ``epochs``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import mne

    raw = _load_mne(raw)
    dn = _load_mne(denoised)
    ep = _load_mne(epochs)

    def mean_meg_psd(obj):
        picks = [c for c in obj.ch_names if re.match(r"^M[LRZ][A-Z]", c)]
        sf = float(obj.info["sfreq"])
        fmx = fmax if fmax is not None else min(120.0, sf / 2.0 - 1.0)
        nperseg = int(round(win_s * sf))   # fixed-DURATION window -> same Hz resolution for all
        psd = obj.compute_psd(method="welch", fmin=fmin, fmax=fmx, n_fft=nperseg,
                              n_per_seg=nperseg, picks=(picks if picks else "data"),
                              verbose="error")
        p, f = psd.get_data(picks="all", return_freqs=True)
        if p.ndim == 3:          # Epochs -> average over epochs
            p = p.mean(axis=0)
        return f, p.mean(axis=0), p.shape[0]

    def line_ratio(f, m, f0):
        peak = m[(f >= f0 - 1) & (f <= f0 + 1)].max()
        base = np.median(m[((f >= f0 - 5) & (f <= f0 - 2)) | ((f >= f0 + 2) & (f <= f0 + 5))])
        return peak / base

    fr, mr, nr = mean_meg_psd(raw)
    fd, md, nd = mean_meg_psd(dn)
    fe, me, ne = mean_meg_psd(ep)
    r50_raw, r50_dn = line_ratio(fr, mr, 50.0), line_ratio(fd, md, 50.0)
    red = 100.0 * (1.0 - r50_dn / r50_raw) if r50_raw else 0.0

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(fr, mr, lw=1.8, color="0.55", label=f"PickedRaw (pre-denoise, {nr} ch)")
    ax.semilogy(fd, md, lw=1.8, color="tab:red", label=f"DenoisedRaw (post-denoise, {nd} ch)")
    ax.semilogy(fe, me, lw=1.1, color="tab:blue", ls=":", label=f"Prepped (+bandpass+600Hz, {ne} ch)")
    for lf, c in ((50.0, "tab:red"), (100.0, "tab:orange")):
        ax.axvline(lf, ls="--", lw=0.9, alpha=0.5, color=c)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (T²/Hz)")
    ax.set_title(f"{denoise_desc} — 50 Hz reduced {red:.0f}% (matched {win_s:g}s Welch)")
    ax.legend(fontsize=8, loc="upper right"); fig.tight_layout()

    rep = mne.Report(title=title, verbose="error")
    rep.add_figure(fig, title="Mean MEG PSD: denoise before/after (matched resolution)", section="Spectrum",
                   caption=f"Grey = PickedRaw (pre-denoise); red = DenoisedRaw (post {denoise_desc}) — both "
                           f"continuous at the same {win_s:g}s Welch window, so the 50 Hz reduction "
                           f"({red:.0f}%: peak/baseline {r50_raw:.1f}→{r50_dn:.1f}) is directly comparable. "
                           f"Dotted blue = final prepped (adds bandpass 0.1–150 + resample 600 + 30 s "
                           f"epoching). Dashed verticals = 50/100 Hz.")
    plt.close(fig)
    rep.add_raw(dn, title="DenoisedRaw (post-denoise, continuous)", psd=psd_seconds, butterfly=False, projs=False)
    rep.add_epochs(ep, title="Prepped epochs (denoise + bandpass 0.1–150, 30 s)", psd=True, projs=False)

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

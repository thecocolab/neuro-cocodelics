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

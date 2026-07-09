#!/usr/bin/env python3
"""Generate a ZapLine A/B comparison pipeline from the base cocodelics pipeline.

Why: the QC spectrum figures (job 47822781) showed the current *adaptive* ZapLine
leaves a 50 Hz residual and does NOT touch the 100 Hz harmonic across all four
FieldTrip datasets. Before committing to a final denoising config we want to
compare, at the FEATURE level (not just the QC PSD), two candidates that differ
in exactly ONE knob — the `adaptive` flag — with harmonic cleaning on in both:

    Fixed     : zapline_denoise(line_freq=50, n_harmonics=2, adaptive=False)
    Adaptive  : zapline_denoise(line_freq=50, n_harmonics=2, adaptive=True)

This script reads the base pipeline (the single source of truth for the classical
battery), clones the classical prep -> per-epoch -> aggregate chain into two
suffixed variants (`...Fixed` / `...Adaptive`), rewiring each node's internal
`derivative:` references to its own variant, and writes a standalone comparison
pipeline. The base pipeline is left untouched so production runs stay lean; once
we pick a winner we set that config in the base pipeline and delete this artifact.

The EXPERIMENTAL phi/IIT/Fisher/Harmonicity nodes are intentionally dropped — the
classical battery is what the ML consumes and is enough to decide the denoising.

Usage:
    .venv/bin/python make_zapcompare_pipeline.py            # -> pipeline_cocodelics_zapcompare.yml
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE / "pipeline_cocodelics.yml"
OUT = HERE / "pipeline_cocodelics_zapcompare.yml"

# Shared (variant-independent) derivatives copied verbatim from the base.
SHARED = ["PickedRaw", "RawMegSpectrum"]

# The classical chain we clone per variant. `PrepDur30Ov20` is handled specially
# (its zapline args are swapped); everything else is cloned + rewired mechanically.
PREP = "PrepDur30Ov20"
CLASSICAL = [
    "Ep_XarrayEpochs",
    "Ep_LZivComplexity", "Ep_HiguchiFD", "Ep_KatzFD", "Ep_PetrosianFD",
    "Ep_SVDEntropy", "Ep_NumZeroCross", "Ep_PermEntropy", "Ep_SpectralEntropy",
    "Ep_HjorthParams", "Ep_DetrendedFluctuation",
    "lzivComplexityMeanEpochs", "higuchiFdMeanEpochs", "higuchiFdVarEpochs",
    "katzFdMeanEpochs", "katzFdSDEpochs", "petrosianFdMeanEpochs",
    "svdEntropyMeanEpochs", "numZerocrossMeanEpochs", "permEntropyMeanEpochs",
    "spectralEntropyMeanEpochs", "hjorthMeanEpochs", "detrendedFluctuationMeanEpochs",
]

# Every derivative name that gets a per-variant suffix (so we know which
# `derivative:` refs to rewrite). PrepDur30Ov20 + all classical nodes.
RENAMED = {PREP, *CLASSICAL}

VARIANTS = {
    "Fixed":    {"line_freq": 50.0, "n_harmonics": 2, "adaptive": False},
    "Adaptive": {"line_freq": 50.0, "n_harmonics": 2, "adaptive": True},
}


def suffix_ref(ref: str, variant: str) -> str:
    """Rewrite a `derivative:` reference to point at the variant's copy.

    Refs look like `PrepDur30Ov20.fif` or `Ep_LZivComplexity.nc`. Only refs whose
    base name is in RENAMED get the suffix inserted before the extension; shared
    refs (PickedRaw.fif, SourceFile) are left alone.
    """
    if "." in ref:
        base, ext = ref.rsplit(".", 1)
        if base in RENAMED:
            return f"{base}{variant}.{ext}"
        return ref
    return f"{ref}{variant}" if ref in RENAMED else ref


def clone_node_def(defn: dict, variant: str) -> dict:
    """Deep-copy a derivative definition, rewiring internal derivative refs."""
    out = copy.deepcopy(defn)
    for node in out.get("nodes", []):
        if "derivative" in node:
            node["derivative"] = suffix_ref(node["derivative"], variant)
    return out


def main() -> None:
    base = yaml.safe_load(BASE.read_text())
    src = base["DerivativeDefinitions"]

    out_defs: dict = {}

    # 1) shared derivatives verbatim
    for name in SHARED:
        out_defs[name] = copy.deepcopy(src[name])

    df_features: list[str] = []  # for_dataframe aggregates, both variants

    # 2) per-variant prep (+ its QC spectrum) and the cloned classical chain
    for variant, zap in VARIANTS.items():
        # prep: clone, swap zapline args, keep everything else
        prep = clone_node_def(src[PREP], variant)
        for node in prep["nodes"]:
            if node.get("node") == "zapline_denoise":
                node["args"] = {"mne_object": "id.0", **zap}
        out_defs[f"{PREP}{variant}"] = prep

        # QC spectrum figure for this variant's prep
        out_defs[f"PrepSpectrum{variant}"] = {
            "overwrite": False, "save": True, "for_dataframe": False,
            "nodes": [
                {"id": 0, "derivative": f"{PREP}{variant}.fif"},
                {"id": 1, "node": "meg_spectrum_fig",
                 "args": {"mne_object": "id.0",
                          "title": f"Prepped MEG spectrum — ZapLine {variant} (n_harm=2)"}},
            ],
        }

        # classical Ep_* + aggregates
        for name in CLASSICAL:
            defn = clone_node_def(src[name], variant)
            out_defs[f"{name}{variant}"] = defn
            if defn.get("for_dataframe"):
                df_features.append(f"{name}{variant}")

    # 3) assemble the pipeline doc, reusing base top-level config
    doc = {
        "datasets": base["datasets"],
        "mount_point": base["mount_point"],
        "new_definitions": ["custom_nodes.py"],  # classical only -> no experimental module
        "n_jobs": base.get("n_jobs", 1),
        "joblib_backend": base.get("joblib_backend", "loky"),
        "joblib_prefer": base.get("joblib_prefer", "processes"),
        "DerivativeDefinitions": out_defs,
        "DerivativeList": list(out_defs.keys()),
    }

    header = (
        "# AUTO-GENERATED by make_zapcompare_pipeline.py — DO NOT EDIT BY HAND.\n"
        "# ZapLine A/B: Fixed (adaptive=False) vs Adaptive (adaptive=True), both\n"
        "# line_freq=50, n_harmonics=2. Two full classical feature sets with distinct\n"
        "# names (…Fixed / …Adaptive) + a PrepSpectrum QC figure each, so the 50/100 Hz\n"
        "# residual can be compared at both the PSD and feature level on the same subjects.\n"
        "# Regenerate: .venv/bin/python make_zapcompare_pipeline.py\n\n"
    )
    OUT.write_text(header + yaml.safe_dump(doc, sort_keys=False, width=100))
    n_feats = len(df_features)
    print(f"wrote {OUT.name}: {len(out_defs)} derivatives, {n_feats} dataframe features "
          f"({n_feats // 2}/variant)")
    print("dataframe features:", ", ".join(df_features))


if __name__ == "__main__":
    main()

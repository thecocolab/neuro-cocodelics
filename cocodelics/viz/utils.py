from pathlib import Path

from mne.io import read_raw_fif

INFO = read_raw_fif(Path(__file__).parent / "info_for_plot_topomap_function.fif", preload=False).info
ROI2SENSOR = {
    "Frontal_Left": [ch for ch in INFO.ch_names if ch.startswith("MLF")],
    "Frontal_Right": [ch for ch in INFO.ch_names if ch.startswith("MRF")],
    "Central_Left": [ch for ch in INFO.ch_names if ch.startswith("MLC")],
    "Central_Right": [ch for ch in INFO.ch_names if ch.startswith("MRC")],
    "Parietal_Left": [ch for ch in INFO.ch_names if ch.startswith("MLP")],
    "Parietal_Right": [ch for ch in INFO.ch_names if ch.startswith("MRP")],
    "Temporal_Left": [ch for ch in INFO.ch_names if ch.startswith("MLT")],
    "Temporal_Right": [ch for ch in INFO.ch_names if ch.startswith("MRT")],
    "Occipital_Left": [ch for ch in INFO.ch_names if ch.startswith("MLO")],
    "Occipital_Right": [ch for ch in INFO.ch_names if ch.startswith("MRO")],
    "Midline": [ch for ch in INFO.ch_names if ch.startswith("MZ")],
}
SENSOR2ROI = {ch: roi for roi, sensors in ROI2SENSOR.items() for ch in sensors}

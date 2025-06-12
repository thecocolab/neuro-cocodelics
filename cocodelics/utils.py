from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.io import read_raw_fif
from mne.stats import fdr_correction
from mne.viz import plot_topomap
from scipy.stats import ttest_1samp

# Load INFO object for topomaps
INFO = read_raw_fif(Path(__file__).parent / "info_for_plot_topomap_function.fif", preload=False).info
INFO.rename_channels(lambda x: x.replace("-3305", ""))

# ROI mappings
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


def get_feature_names(df):
    """Extract feature names from dataframe columns."""
    return list(set([col.replace("feature-", "").split(".")[0] for col in df.columns if col.startswith("feature")]))


def get_channel_names(df):
    """Extract channel names from dataframe columns."""
    return list(set([col[-5:] for col in df.columns if ".spaces-" in col]))


def load_data(data_dir, ignore_features=None, act_minus_pcb=True, normalize=False):
    """
    Load and process data from CSV files.

    Parameters:
    -----------
    data_dir : str
        Directory containing CSV files
    ignore_features : list, optional
        List of feature names to ignore
    act_minus_pcb : bool, default True
        If True, compute (active - placebo) / (placebo + 1e-4)
        If False, keep active and placebo separate
    normalize : bool, default True
        If True, apply z-score normalization

    Returns:
    --------
    tuple
        (data_dict, feature_names, channel_names)
    """
    if ignore_features is None:
        ignore_features = []

    ft_names, ch_names = None, None
    data = {}

    for path in sorted(glob(data_dir + "/*.csv")):
        name = path.split("/")[-1].split(".")[0]
        if name.startswith("aggregate"):
            continue

        df = pd.read_csv(path, index_col=0)

        if ft_names is None:
            ft_names = get_feature_names(df)
            ch_names = get_channel_names(df)
        else:
            assert ft_names == get_feature_names(df), "Feature names do not match across datasets."
            assert ch_names == get_channel_names(df), "Channel names do not match across datasets."

        target = df["target"]
        df = df.drop(columns="target")

        # Apply normalization if requested
        if normalize:
            for col in df.columns:
                if df[col].std() > 1e-8:  # avoid division by zero
                    df[col] = (df[col] - df[col].mean()) / df[col].std()

        if act_minus_pcb:
            if name not in ignore_features:
                data[name] = (df[target == 1] - df[target == 0]) / (df[target == 0] + 1e-4)
        else:
            if name not in ignore_features:
                data[name] = df[target == 1]
            if name + "-pcb" not in ignore_features:
                data[name + "-pcb"] = df[target == 0]

    return data, ft_names, ch_names


def get_data(df, ft_name, ch_names, avg_subjs=False, avg_chs=False):
    """
    Extract data for a specific feature.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    ft_name : str
        Feature name to extract
    ch_names : list
        List of channel names
    avg_subjs : bool, default False
        If True, average across subjects
    avg_chs : bool, default False
        If True, average across channels

    Returns:
    --------
    numpy.ndarray
        Extracted data array
    """
    col_names = [f"feature-{ft_name}.spaces-{ch}" for ch in ch_names]
    data = df[col_names].values
    if avg_subjs:
        data = data.mean(axis=0, keepdims=True)
    if avg_chs:
        data = data.mean(axis=1, keepdims=True)
    return data


def average_across_subjects(data, ft_names, ch_names):
    """
    Average data across subjects for each condition and feature.

    Parameters:
    -----------
    data : dict
        Dictionary with condition names as keys and DataFrames as values
    ft_names : list
        List of feature names
    ch_names : list
        List of channel names

    Returns:
    --------
    dict
        Dictionary with averaged data for each condition and feature
    """
    averaged_data = {}

    for condition, df in data.items():
        averaged_data[condition] = {}
        for ft_name in ft_names:
            col_names = [f"feature-{ft_name}.spaces-{ch}" for ch in ch_names]
            feature_data = df[col_names].values
            averaged_data[condition][ft_name] = feature_data.mean(axis=0)

    return averaged_data


def ttest_across_subjects(data, ft_names, ch_names):
    """
    Perform t-test across subjects for each condition and feature, with FDR correction.

    Parameters:
    -----------
    data : dict
        Dictionary with condition names as keys and DataFrames as values
    ft_names : list
        List of feature names
    ch_names : list
        List of channel names

    Returns:
    --------
    tuple of dict
        (t_values, p_values) where each is:
        {condition: {ft_name: array}}
    """

    t_values = {}
    p_values = {}

    for condition, df in data.items():
        t_values[condition] = {}
        p_values[condition] = {}
        for ft_name in ft_names:
            col_names = [f"feature-{ft_name}.spaces-{ch}" for ch in ch_names]
            feature_data = df[col_names].values  # shape: (subjects, channels)
            t_vals, p_vals = ttest_1samp(feature_data, popmean=0, axis=0, nan_policy="omit")
            _, p_vals_fdr = fdr_correction(p_vals)
            t_values[condition][ft_name] = t_vals
            p_values[condition][ft_name] = p_vals_fdr

    return t_values, p_values


def plot_topomaps_by_feature(tval, pval, ft_names, ch_names, axes=None, figsize=(15, 10), p_thresh=0.05):
    """
    Plot topomaps for each feature across all conditions using all channels.
    Significant (FDR-corrected) channels are masked.

    Parameters:
    -----------
    tval : dict
        Dictionary with condition names as keys and t-values as values
    pval : dict
        Dictionary with condition names as keys and p-values as values
    ft_names : list
        List of feature names
    ch_names : list
        List of channel names
    axes : list, optional
        List of axes to plot on. If None, new axes will be created.
    figsize : tuple, default (15, 10)
        Figure size
    p_thresh : float, default 0.05
        Significance threshold for p-values (after FDR correction)
    """
    n_features = len(ft_names)
    n_conditions = len(tval)

    # Handle axes argument
    if axes is None:
        _, axes = plt.subplots(n_features, n_conditions, figsize=figsize)
    else:
        axes = np.asarray(axes)
        assert axes.shape[0] == n_features, f"axes must have {n_features} rows (features), got {axes.shape[0]}"
        if axes.ndim == 1:
            if n_features == 1:
                axes = axes.reshape(1, -1)
            elif n_conditions == 1:
                axes = axes.reshape(-1, 1)

    # Get indices of ch_names in INFO.ch_names
    ch_indices = [INFO.ch_names.index(ch) for ch in ch_names if ch in INFO.ch_names]

    for i, ft_name in enumerate(ft_names):
        for j, (condition, condition_data) in enumerate(tval.items()):
            ax = axes[i, j] if n_features > 1 and n_conditions > 1 else axes[i] if n_conditions == 1 else axes[j]

            # Get feature data and p-values for this condition
            feature_values = condition_data[ft_name]
            feature_pvals = pval[condition][ft_name]

            # Create data array for topomap (all channels)
            topo_data = np.zeros(len(INFO.ch_names))
            mask = np.zeros(len(INFO.ch_names), dtype=bool)
            for k, idx in enumerate(ch_indices):
                topo_data[idx] = feature_values[k]
                mask[idx] = feature_pvals[k] < p_thresh

            # Plot topomap with mask for significant channels
            im, _ = plot_topomap(
                topo_data,
                INFO,
                axes=ax,
                show=False,
                cmap="RdBu_r",
                sensors=True,
                names=None,
                mask=mask,
                # mask_params=dict(marker="*"),
                contours=False,
            )

            # Set title
            if i == 0:
                ax.set_title(f"{condition}", fontsize=12, fontweight="bold")
            if j == 0:
                ax.text(
                    -0.1,
                    0.5,
                    ft_name,
                    rotation=0,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                )

    plt.tight_layout()
    plt.show()


# Color map for different conditions
COLOR_MAP = {
    "lsd-Closed1": "#c6b4e3",
    "lsd-Closed2": "#b19bdc",
    "lsd-Music": "#9b83d5",
    "lsd-Open2": "#856bcc",
    "lsd-Open1": "#6f54c3",
    "lsd-Video": "#593ebb",
    "lsd-avg": "#4327b2",
    "ketamine": "#1dbc7c",
    "psilocybin": "#bf00ee",
    "perampanel": "#bfa900",
    "tiagabine": "#e61a1a",
}

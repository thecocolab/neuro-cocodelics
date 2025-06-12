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
    "ROI_Frontal_Left": [ch for ch in INFO.ch_names if ch.startswith("MLF")],
    "ROI_Frontal_Right": [ch for ch in INFO.ch_names if ch.startswith("MRF")],
    "ROI_Central_Left": [ch for ch in INFO.ch_names if ch.startswith("MLC")],
    "ROI_Central_Right": [ch for ch in INFO.ch_names if ch.startswith("MRC")],
    "ROI_Parietal_Left": [ch for ch in INFO.ch_names if ch.startswith("MLP")],
    "ROI_Parietal_Right": [ch for ch in INFO.ch_names if ch.startswith("MRP")],
    "ROI_Temporal_Left": [ch for ch in INFO.ch_names if ch.startswith("MLT")],
    "ROI_Temporal_Right": [ch for ch in INFO.ch_names if ch.startswith("MRT")],
    "ROI_Occipital_Left": [ch for ch in INFO.ch_names if ch.startswith("MLO")],
    "ROI_Occipital_Right": [ch for ch in INFO.ch_names if ch.startswith("MRO")],
    "ROI_Midline": [ch for ch in INFO.ch_names if ch.startswith("MZ")],
}
SENSOR2ROI = {ch: roi for roi, sensors in ROI2SENSOR.items() for ch in sensors}


def get_feature_names(df):
    """Extract feature names from dataframe columns."""
    return list(set([col.replace("feature-", "").split(".")[0] for col in df.columns if col.startswith("feature")]))


def get_channel_names(df):
    """Extract channel names from dataframe columns."""
    return list(set([col[-5:] for col in df.columns if ".spaces-" in col]))


def load_data(data_dir, ignore_features=[], act_minus_pcb=True, normalize=False, rois=False, ignore_rois=[]):
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
    rois : bool, default False
        If True, return data in ROI format
    ignore_rois : list, optional
        List of ROIs to ignore (not used in current implementation)

    Returns:
    --------
    tuple
        (data_dict, feature_names, channel_names)
    """
    ft_names, ch_names, col_names = None, None, None
    data = {}

    paths = glob(data_dir + "/*.csv")

    # Sort paths by DATASET_ORDER
    def dataset_key(path):
        name = Path(path).stem
        try:
            return DATASET_ORDER.index(name)
        except ValueError:
            return len(DATASET_ORDER)  # put unknowns at the end

    paths = sorted(paths, key=dataset_key)

    for path in paths:
        name = path.split("/")[-1].split(".")[0]
        if name.startswith("aggregate"):
            continue

        df = pd.read_csv(path, index_col=0)
        target = df["target"]
        df = df.drop(columns="target")

        if ft_names is None:
            ft_names = get_feature_names(df)
            ch_names = get_channel_names(df)
            col_names = df.columns.tolist()
        else:
            assert ft_names == get_feature_names(df), "Feature names do not match across datasets."
            assert ch_names == get_channel_names(df), "Channel names do not match across datasets."
            assert col_names == df.columns.tolist(), "Column names do not match across datasets."

        if rois:
            new_df = {}
            for roi, sensors in ROI2SENSOR.items():
                if roi in ignore_rois:
                    continue
                roi_data = {}
                for col in df.columns:
                    for sensor in sensors:
                        if sensor in col:
                            feat = col.replace("feature-", "").split(".")[0]
                            if feat not in roi_data:
                                roi_data[feat] = []
                            roi_data[feat].append(df[col].values)
                for feat in roi_data:
                    new_df[f"feature-{feat}.spaces-{roi}"] = np.mean(roi_data[feat], axis=0)
            df = pd.DataFrame(new_df, index=df.index)

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

    if rois:
        return data, ft_names, [roi for roi in ROI2SENSOR.keys() if roi not in ignore_rois], df.columns.tolist()
    return data, ft_names, ch_names, col_names


def get_feature_data(df, ft_name, ch_names, avg_subjs=False, avg_chs=False):
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


def plot_topomaps_by_feature(tval, pval, ft_names, ch_names, axes=None, figsize=(15, 10), p_thresh=0.05, vlim=5):
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
    vlim : float, default 5
        Value limit for color scale in topomap plots
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
                contours=False,
                sphere=(0, 0, -0.11, 0.1),
                vlim=(-vlim, vlim),
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
                    ha="right",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                )

    plt.tight_layout()
    plt.show()


def plot_spider_by_feature(tval, ft_names, ch_names, figsize=(8, 8)):
    """
    Plot a spider (radar) plot for each condition.
    Each branch is a feature, each color is a condition.
    The value is the average t-value across sensors for that feature and condition.

    Parameters:
    -----------
    tval : dict
        Dictionary with condition names as keys and t-values as values
    ft_names : list
        List of feature names
    ch_names : list
        List of channel names
    figsize : tuple, default (8, 8)
        Figure size
    vlim : float, default 5
        Value limit for radial axis
    """
    n_features = len(ft_names)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    plt.figure(figsize=figsize)
    ax = plt.subplot(111, polar=True)

    # Add bold contour at 0
    ax.plot(angles, [0] * len(angles), color="black", linestyle="-", linewidth=5, zorder=0)

    for condition, ft_dict in tval.items():
        values = [np.mean(ft_dict[ft]) for ft in ft_names]
        values += values[:1]  # close the loop
        color = COLOR_MAP.get(condition, None)
        ax.plot(angles, values, label=condition, color=color, linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(ft_names, fontsize=12)
    ax.set_title("Average t-values across sensors (by feature)", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(tval, ft_names, ch_names, figsize=(10, 6), title="", kind="violin"):
    """
    Violin or bar plots of t-values grouped by feature, one color per condition.

    Parameters:
    -----------
    tval : dict
        {condition: {feature: tval_array (channels,)}}
    ft_names : list
        List of feature names
    ch_names : list
        List of channel names (unused here)
    figsize : tuple
        Figure size
    title : str
        Title for the plot
    kind : str
        Type of plot ("violin" or "bar")
    """
    n_features = len(ft_names)
    n_conditions = len(tval)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    width = 0.8 / n_conditions  # violin width
    positions = np.arange(n_features)

    for idx, (condition, ft_dict) in enumerate(tval.items()):
        data = [np.asarray(ft_dict[ft]) for ft in ft_names]
        pos = positions + (idx - (n_conditions - 1) / 2) * width

        if kind == "violin":
            vp = ax.violinplot(data, positions=pos, widths=width, showmeans=False, showextrema=False, showmedians=False)
            color = COLOR_MAP.get(condition, f"C{idx}")
            for pc in vp["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
                pc.set_edgecolor("black")
                pc.set_linewidth(0.8)
        elif kind == "bar":
            means = [np.mean(ft_dict[ft]) for ft in ft_names]
            stds = [np.std(ft_dict[ft]) for ft in ft_names]
            ax.bar(pos, means, width=width, yerr=stds, label=condition, color=COLOR_MAP.get(condition, f"C{idx}"), alpha=0.7)
        else:
            raise ValueError("kind must be either 'violin' or 'bar'.")

    # Add vertical dashed lines between features
    for i in range(1, n_features):
        ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=1, zorder=0)

    ax.axhline(0, color="k", linewidth=1.5, linestyle="--", zorder=0)
    ax.set_xticks(positions)
    ax.set_xticklabels(ft_names, rotation=30, ha="right")
    ax.set_ylabel("t-value")
    ax.set_title(title, fontweight="bold")

    # Create legend manually
    from matplotlib.patches import Patch

    legend_handles = [Patch(color=COLOR_MAP.get(cond, f"C{i}"), label=cond) for i, cond in enumerate(tval.keys())]
    plt.legend(handles=legend_handles)

    plt.tight_layout()
    plt.show()


# Color map for different conditions
COLOR_MAP = {
    "lsd-Closed1": "#8B0000",  # dark red
    "lsd-Closed2": "#B22222",  # firebrick
    "lsd-Music": "#FF4500",  # orange red
    "lsd-Open2": "#FF8C00",  # dark orange
    "lsd-Open1": "#EB0000",  # gold/yellow
    "lsd-Video": "#FFA500",  # orange
    "lsd-avg": "#FFB300",  # yellow-orange
    "ketamine": "#FFD700",  # yellow
    "psilocybin": "#FF8C00",  # dark orange
    "perampanel": "#2979ff",  # blue
    "tiagabine": "#43a047",  # green
}

FEATURE_ORDER = [
    "Detrended Fluctuation",
    "Higuchi Fd",
    "Petrosian Fd",
    "Katz Fd",
    "Hjorth Complexity",
    "Hjorth Mobility",
    "Lziv Complexity",
    "numZerocross",
    "Spectral Entropy",
    "Svd Entropy",
]

DATASET_ORDER = [
    "lsd-Closed1",
    "lsd-Closed1-pcb",
    "lsd-Closed2",
    "lsd-Closed2-pcb",
    "lsd-Music",
    "lsd-Music-pcb",
    "lsd-Open1",
    "lsd-Open1-pcb",
    "lsd-Open2",
    "lsd-Open2-pcb",
    "lsd-Video",
    "lsd-Video-pcb",
    "lsd-avg",
    "lsd-avg-pcb",
    "psilocybin",
    "psilocybin-pcb",
    "ketamine",
    "ketamine-pcb",
    "perampanel",
    "perampanel-pcb",
    "tiagabine",
    "tiagabine-pcb",
]

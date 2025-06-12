import os
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import mne
import sys
sys.path.insert(0, os.path.dirname(__file__))
# from utils import load_aggregated_pickle

def load_aggregated_pickle(
    aggregated_dir, global_experiment_id, dataset, analysis_type, model_name
):
    """Loads a standard aggregated results pickle file for performance."""
    file_name = (
        f"{global_experiment_id}_{dataset}_{analysis_type}_{model_name}.pkl"
    )
    full_path = os.path.join(aggregated_dir, file_name)
    if not os.path.exists(full_path):
        # Fallback for older naming convention if needed
        file_name_alt = f"{global_experiment_id}_{dataset}_{model_name}.pkl"
        full_path_alt = os.path.join(aggregated_dir, file_name_alt)
        if not os.path.exists(full_path_alt):
            raise FileNotFoundError(f"Aggregated pickle not found in either format: {full_path} or {full_path_alt}")
        full_path = full_path_alt
    
    with open(full_path, "rb") as f:
        data = pickle.load(f)
    return data

# --- Globals and Constants ---
CTF_MEG_INFO_FILE = 'viz/info_for_plot_topomap_function.fif'

# --- Data Handling ---

def load_ctf_meg_info():
    """Loads the CTF MEG sensor information from the FIF file."""
    if not os.path.exists(CTF_MEG_INFO_FILE):
        raise FileNotFoundError(f"CTF MEG info file not found: {CTF_MEG_INFO_FILE}")
    return mne.io.read_info(CTF_MEG_INFO_FILE, verbose=False)

def extract_performance_metrics(aggregated_results):
    """
    Extracts performance metrics and reshapes them for plotting.
    
    Input format: {sensor: {metric_name: value}}
    Output format: {metric_name: {sensor: value}}
    """
    performance_metrics = {}
    if not aggregated_results:
        return performance_metrics

    first_sensor_data = next(iter(aggregated_results.values()))
    metric_names = list(first_sensor_data.keys())

    for metric_name in metric_names:
        performance_metrics[metric_name] = {}
        for sensor_name, sensor_data in aggregated_results.items():
            if metric_name in sensor_data:
                performance_metrics[metric_name][sensor_name] = sensor_data[metric_name]
    
    return performance_metrics

def _prepare_data_for_topomap(data_dict, info):
    """
    Prepares the data dictionary so that keys correspond to the **base** sensor names
    (e.g. 'MRC41') expected by the MNE layout used later for plotting.

    Parameters
    ----------
    data_dict : dict
        Mapping ``{sensor_name_base: value}`` obtained from the aggregated
        performance pickle.
    info : mne.Info
        The measurement info read from the CTF FIF file. Currently not used but
        kept as a parameter to preserve the public signature.

    Returns
    -------
    dict
        ``{base_sensor_name: value}`` – ready to be consumed by
        ``_plot_topomap_single``.
    """
    # The aggregated results already come with base sensor names (e.g. 'MRC41').
    # We therefore simply return a *copy* of the incoming dict to make it clear
    # that this function performs no further conversion.
    return dict(data_dict)

# --- Core Plotting Function ---

def _plot_topomap_single(ax, data_dict, info, vmin, vmax, cmap, title, plot_style, show_sensors):
    """A private helper function to plot a single topomap."""

    # Obtain MEG sensor layout for CTF system
    layout = mne.find_layout(info, ch_type='meg')
    layout_base_names = [name.split('-')[0] if '-' in name else name for name in layout.names]

    # Build an array with the data values in the *layout* order
    layout_data_np = np.array([data_dict.get(base_name, np.nan) for base_name in layout_base_names])

    # Convert to standard Python floats to ensure compatibility
    layout_data = [float(x) if not np.isnan(x) else np.nan for x in layout_data_np]
    layout_data = np.array(layout_data)
    # Plot
    im, _ = mne.viz.plot_topomap(
        layout_data,
        layout.pos,
        show=False,
        vlim=(vmin, vmax),
        cmap=None,
        axes=ax,
        contours=6,
        outlines='head',
        sensors=show_sensors,
        sphere=None,
        extrapolate='auto'
    )

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    return im

# --- Main Public Function ---

def make_performance_topomap_grid(
    aggregated_dir, global_experiment_id, dataset, analysis_type, model_name,
    metrics_to_plot=None, save_path=None, figsize=(10, 8), dpi=300, 
    file_format='png', show_sensors=False, colorbar_label='Metric Value',
    plot_style='interpolate', cmap='RdYlBu_r'
):
    """
    Creates a publication-quality grid of topomaps for model performance metrics.
    """
    # 1. Load data and sensor info
    info = load_ctf_meg_info()
    aggregated_results = load_aggregated_pickle(
        aggregated_dir, global_experiment_id, dataset, analysis_type, model_name
    )

    all_metrics = extract_performance_metrics(aggregated_results)
    
    # 2. Prepare data for all plots
    if metrics_to_plot is None:
        metrics_to_plot = [m for m in all_metrics.keys() if 'mean' in m]

    all_plots_data = {}
    for name in metrics_to_plot:
        if name in all_metrics:
            all_plots_data[name] = _prepare_data_for_topomap(all_metrics[name], info)
        else:
            print(f"Warning: Metric '{name}' not found in data. Skipping.")

    if not all_plots_data:
        print("Error: No valid metrics to plot. Aborting.")
        return None

    # 3. Setup grid and color scale
    plot_titles_raw = list(all_plots_data.keys())
    plot_titles = [title.replace('metric_scores.', '').replace('.mean', '').replace('_', ' ').title() for title in plot_titles_raw]
    n_plots = len(plot_titles)
    
    if n_plots == 0:
        print("No plots to generate.")
        return None
        
    if n_plots <= 4: nrows, ncols = 2, 2
    elif n_plots <= 6: nrows, ncols = 2, 3
    else: nrows, ncols = 3, 3

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='white')
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    # Correctly extract values from the dictionaries for color scale calculation
    all_values = np.concatenate([list(data.values()) for data in all_plots_data.values() if data])
    vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)

    # 4. Create all plots
    im = None
    for i, title_key in enumerate(plot_titles_raw):
        im = _plot_topomap_single(axes[i], all_plots_data[title_key], info, vmin, vmax, 
                                  cmap, plot_titles[i], plot_style, show_sensors)

    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    # 5. Finalize figure layout, titles, and colorbar
    fig.suptitle(f'Model Performance Topomaps\nDataset: {dataset} | Model: {model_name}',
                 fontsize=14, fontweight='bold', y=0.96)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.15, wspace=0.2, hspace=0.3)
    
    if im:
        cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.03])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(colorbar_label, fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.text(0.99, 0.01, f'Generated: {timestamp}', ha='right', va='bottom', 
             fontsize=8, alpha=0.7)

    # 6. Save if requested
    if save_path:
        final_save_path = f"{save_path}.{file_format}"
        print(f"Saving figure to: {final_save_path}")
        fig.savefig(final_save_path, dpi=dpi, facecolor='white', transparent=False)
    
    return fig



def _plot_radar_single(ax, categories, values, vmin=None, vmax=None,
                       fill_alpha=0.25, line_kwargs=None, fill_kwargs=None,
                       title=None, label=None):
    """Plot a single radar / spider chart on ``ax``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes created with ``polar=True``.
    categories : list[str]
        Names of each spoke.
    values : list[float] | np.ndarray
        Data value for each spoke (must be same length as *categories*).
    vmin, vmax : float | None
        Global radial limits.  If *None* the limits are derived from *values*.
    fill_alpha : float
        Transparency of the filled polygon.
    line_kwargs, fill_kwargs : dict | None
        Extra keyword arguments forwarded to ``ax.plot`` / ``ax.fill``.
    title : str | None
        Optional title placed above this radar chart.
    label : str | None
        Label for the polygon (used in legends).
    """
    N = len(categories)
    if len(values) != N:
        raise ValueError("Length of values does not match categories.")

    # Compute angles for each spoke
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    theta += theta[:1]  # close the loop

    vals = np.asarray(values, dtype=float).tolist()
    vals += vals[:1]

    # Rotation so the first axis is at the top and clockwise direction
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Grid labels
    ax.set_thetagrids(np.degrees(theta[:-1]), categories)

    if vmin is None:
        vmin = np.nanmin(vals)
    if vmax is None:
        vmax = np.nanmax(vals)

    ax.set_ylim(vmin, vmax)

    # Draw polygon
    line_kwargs = dict(line_kwargs or {})
    if label is not None:
        line_kwargs.setdefault("label", label)
    line, = ax.plot(theta, vals, **line_kwargs)

    # Fill polygon
    fill_kwargs = dict(fill_kwargs or {})
    ax.fill(theta, vals, alpha=fill_alpha, **fill_kwargs)

    # Title and aesthetics
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    return line




def _pivot_radar_data(model_results_dict):
    """Pivot raw aggregated results into *metric -> model -> category -> value*.

    Parameters
    ----------
    model_results_dict : dict[str, dict]
        Mapping ``{model_name: aggregated_results}`` where *aggregated_results*
        follows the schema returned by :pyfunc:`utils.load_aggregated_pickle`:
        ``{category: {metric_name: value}}``.

    Returns
    -------
    dict
        ``{metric_name: {model_name: {category: value}}}``.
    """
    pivot = {}
    for model_name, aggregated_results in model_results_dict.items():
        for category, metrics in aggregated_results.items():
            for metric_name, value in metrics.items():
                pivot.setdefault(metric_name, {}).setdefault(model_name, {})[category] = value
    return pivot


def _pick_grid(n_plots):
    """Return sensible (nrows, ncols) for *n_plots* small grid."""
    if n_plots <= 4:
        return 2, 2
    if n_plots <= 6:
        return 2, 3
    return 3, 3



def make_performance_radar_grid(
    aggregated_dir,
    global_experiment_id,
    dataset,
    analysis_type,
    model_names,
    metrics_to_plot=None,
    categories_to_show=None,
    save_path=None,
    figsize=(10, 8),
    dpi=300,
    file_format='png',
    fill_alpha=0.25,
    color_cycle=None,
    vmin=None,
    vmax=None,
):
    """Create a grid of radar plots comparing *model_names* on each *metric*.

    The function mirrors :pyfunc:`make_performance_topomap_grid` so it can be
    used interchangeably in notebooks / scripts.
    """

    # ------------------------------------------------------------------
    # 1) Load all aggregated results (one pickle per model)
    # ------------------------------------------------------------------
    model_results = {}
    for model_name in model_names:
        try:
            model_results[model_name] = load_aggregated_pickle(
                aggregated_dir,
                global_experiment_id,
                dataset,
                analysis_type,
                model_name,
            )
        except FileNotFoundError as err:
            print(f"Warning: results for model '{model_name}' not found – skipping. ({err})")
            continue

    if not model_results:
        print("Error: No model results loaded; aborting radar plot.")
        return None

    # ------------------------------------------------------------------
    # 2) Pivot to metric -> model -> category dictionary
    # ------------------------------------------------------------------
    pivoted = _pivot_radar_data(model_results)

    # Determine default metrics if none provided
    if metrics_to_plot is None:
        metrics_to_plot = [m for m in pivoted.keys() if 'mean' in m]

    # Consistent category ordering
    # Take categories from the first available metric & model
    first_metric = metrics_to_plot[0]
    first_model = next(iter(pivoted[first_metric].keys()))
    categories = sorted(pivoted[first_metric][first_model].keys())

    if categories_to_show is not None:
        # Ensure requested categories exist
        categories = [c for c in categories_to_show if c in categories]

    if not categories:
        print("Error: No categories available for radar plot.")
        return None

    # ------------------------------------------------------------------
    # 3) Global vmin / vmax across all metrics & models unless provided
    # ------------------------------------------------------------------
    if vmin is None or vmax is None:
        all_vals = [val
                    for metric in metrics_to_plot
                    for model in model_names
                    if model in pivoted.get(metric, {})
                    for val in pivoted[metric][model].values()]
        if vmin is None:
            vmin = np.nanmin(all_vals)
        if vmax is None:
            vmax = np.nanmax(all_vals)

    # ------------------------------------------------------------------
    # 4) Prepare figure grid
    # ------------------------------------------------------------------
    n_plots = len(metrics_to_plot)
    nrows, ncols = _pick_grid(n_plots)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        subplot_kw={'polar': True},
        figsize=figsize,
        facecolor='white',
    )
    axes = axes.flatten() if n_plots > 1 else [axes]

    color_cycle = color_cycle or plt.rcParams['axes.prop_cycle'].by_key()['color']

    # ------------------------------------------------------------------
    # 5) Draw each radar chart
    # ------------------------------------------------------------------
    for ax, metric_name in zip(axes, metrics_to_plot):
        # Title nice formatting
        pretty_metric = (
            metric_name.replace('metric_scores.', '')
            .replace('.mean', '')
            .replace('_', ' ')
            .title()
        )

        # Draw one polygon per model
        for i, model_name in enumerate(model_names):
            if model_name not in pivoted.get(metric_name, {}):
                continue  # model missing this metric

            values_per_cat = [pivoted[metric_name][model_name].get(cat, np.nan) for cat in categories]

            _plot_radar_single(
                ax,
                categories,
                values_per_cat,
                vmin=vmin,
                vmax=vmax,
                fill_alpha=fill_alpha,
                line_kwargs={'color': color_cycle[i % len(color_cycle)], 'linewidth': 2},
                fill_kwargs={'color': color_cycle[i % len(color_cycle)]},
                title=None,  # set common title later
                label=model_name,
            )

        ax.set_title(pretty_metric, fontsize=11, fontweight='bold', pad=10)
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05), fontsize=8, frameon=False)

    # Hide unused axes
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    # ------------------------------------------------------------------
    # 6) Finalise layout and optional save
    # ------------------------------------------------------------------
    fig.suptitle(
        f'Model Performance Radar Charts\nDataset: {dataset}',
        fontsize=14,
        fontweight='bold',
        y=0.96,
    )

    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.88,
        bottom=0.05,
        wspace=0.3,
        hspace=0.3,
    )

    # Timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.text(
        0.99,
        0.01,
        f'Generated: {timestamp}',
        ha='right',
        va='bottom',
        fontsize=8,
        alpha=0.7,
    )

    if save_path:
        final_save = f"{save_path}.{file_format}"
        print(f"Saving radar figure to: {final_save}")
        fig.savefig(final_save, dpi=dpi, facecolor='white', transparent=False)

    return fig


def load_feature_importance_pickle(
    aggregated_dir, global_experiment_id, drug_name, analysis_type, model_name
):
    """Loads a feature-importance pickle file for a given drug and model."""
    file_name = (
        f"{global_experiment_id}_{drug_name}_{analysis_type}_"
        f"{model_name}_feat_importance.pkl"
    )
    full_path = os.path.join(aggregated_dir, file_name)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Feature importance file not found: {full_path}")
    with open(full_path, "rb") as f:
        data = pickle.load(f)
    return data


def _pivot_feature_importance_data(importance_data_dict):
    """
    Pivot and aggregate feature importance data.
    Input: {drug: {sensor: {feature: value}}}
    Output: {drug: {feature: avg_value_across_sensors}}
    """
    pivot = {}
    agg_temp = {}  # {drug: {feature: [values]}}

    for drug_name, sensor_data in importance_data_dict.items():
        for sensor, feature_data in sensor_data.items():
            for feature_name, value in feature_data.items():
                agg_temp.setdefault(drug_name, {}).setdefault(feature_name, []).append(value)

    # Average the aggregated values
    for drug_name, feature_data in agg_temp.items():
        for feature_name, values in feature_data.items():
            avg_value = np.mean(values)
            pivot.setdefault(drug_name, {})[feature_name] = avg_value
            
    return pivot


def make_feature_importance_radar_grid(
    aggregated_dir,
    global_experiment_id,
    drug_names,
    analysis_type,
    model_name,
    features_to_show=None,
    save_path=None,
    figsize=(10, 8),
    dpi=300,
    file_format="png",
    fill_alpha=0.25,
    color_cycle=None,
    vmin=None,
    vmax=None,
):
    """Create a radar plot comparing feature importance across drugs.
 
    For a single ML model, this function generates a radar plot
    where the radar's spokes are the features and each colored
    polygon represents a different drug.
 
    Parameters
    ----------
    aggregated_dir : str
        Path to the directory containing aggregated pickle files.
    global_experiment_id : str
        The base name for the experiment files.
    drug_names : list[str]
        A list of drug names (datasets) to compare.
    analysis_type : str
        The analysis type (e.g., 'baseline').
    model_name : str
        The machine learning model to use (e.g., 'Logistic Regression').
    features_to_show : list[str], optional
        A subset of features to show on the radar spokes. If None, shows all.
    save_path : str, optional
        Base path to save the figure. The format will be appended.
    figsize, dpi, file_format, fill_alpha, color_cycle, vmin, vmax
        Styling parameters for the plot.
    """
    # 1. Load data for all requested drugs
    importance_data = {}
    for drug in drug_names:
        try:
            importance_data[drug] = load_feature_importance_pickle(
                aggregated_dir, global_experiment_id, drug, analysis_type, model_name
            )
        except FileNotFoundError as e:
            print(f"Warning: Could not load data for drug '{drug}'. Skipping. Details: {e}")
            continue
 
    if not importance_data:
        print("Error: No feature importance data could be loaded. Aborting.")
        return None
 
    # 2. Pivot data into the required structure for plotting
    pivoted_data = _pivot_feature_importance_data(importance_data)

    # No metric loop needed, we generate one plot
    features = sorted(next(iter(pivoted_data.values())).keys())

    if features_to_show:
        features = [f for f in features_to_show if f in features]
 
    if not features:
        print("Error: No features available to plot.")
        return None
 
    # 3. Determine global value range for consistent axes
    if vmin is None or vmax is None:
        all_vals = [
            val
            for drug_data in pivoted_data.values()
            for val in drug_data.values()
        ]
        if not all_vals:
            vmin, vmax = 0, 1 # Default if no data
        else:
            if vmin is None:
                vmin = np.nanmin(all_vals)
            if vmax is None:
                vmax = np.nanmax(all_vals)

    # 4. Set up figure
    fig, ax = plt.subplots(
        subplot_kw={"polar": True}, figsize=figsize, facecolor="white"
    )

    color_cycle = color_cycle or plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Draw one polygon per drug
    for i, drug_name in enumerate(drug_names):
        if drug_name not in pivoted_data:
            continue

        values = [pivoted_data[drug_name].get(f, np.nan) for f in features]

        _plot_radar_single(
            ax,
            features,
            values,
            vmin=vmin,
            vmax=vmax,
            fill_alpha=fill_alpha,
            line_kwargs={"color": color_cycle[i % len(color_cycle)], "linewidth": 2},
            fill_kwargs={"color": color_cycle[i % len(color_cycle)]},
            label=drug_name.title(),
        )

    ax.legend(
        loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9, frameon=False
    )

    # Finalize and save
    fig.suptitle(
        f'Mean Feature Importance Radar: {model_name}\nAnalysis: {analysis_type.title()}',
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(pad=3.0, h_pad=4.0, w_pad=4.0)
    fig.subplots_adjust(top=0.85) # Adjust top to make space for suptitle
 
    if save_path:
        final_save_path = f"{save_path}.{file_format}"
        print(f"Saving figure to: {final_save_path}")
        fig.savefig(final_save_path, dpi=dpi, facecolor='white', transparent=False)
 
    return fig


# --- Example Usage ---
if __name__ == '__main__':
    AGGREGATED_DIR = "/home/sesma/projects/def-kjerbi/data_neurococodelics/aggregated"
    GLOBAL_EXPERIMENT_ID = "neuro_cocodelics_single_sensor_all_features"
    ANALYSIS_TYPE = "baseline"
    MODEL_NAME = "Logistic Regression"

    # --- Example for Feature Importance Radar Plot ---
    try:
        print("\n--- Generating Feature Importance Radar Grid ---")
        drug_list = [
            "psilocybin",
            "tiagabine",
            "perampanel",
            "lsd-avg",
            "lsd-Closed1",
            "lsd-Closed2",
            "lsd-Music",
            "lsd-Open1",
            "lsd-Open2",
            "lsd-Video",
        ]
        fig_feat = make_feature_importance_radar_grid(
            aggregated_dir=AGGREGATED_DIR,
            global_experiment_id=GLOBAL_EXPERIMENT_ID,
            drug_names=drug_list,
            analysis_type=ANALYSIS_TYPE,
            model_name=MODEL_NAME,
            save_path=f'fig_feat_importance_{MODEL_NAME.replace(" ", "_")}',
        )
        if fig_feat:
            print("✅ Successfully generated feature importance radar figure.")
            plt.show()
        else:
            print("❌ Feature importance radar figure generation failed.")

    except Exception as e:
        print(f"❌ Error generating feature importance figure: {e}")
        import traceback
        traceback.print_exc()


    # --- Original Example for Topomap ---
    try:
        print("\n--- Generating Performance Topomap Grid ---")
        fig = make_performance_topomap_grid(
            aggregated_dir=AGGREGATED_DIR,
            global_experiment_id=GLOBAL_EXPERIMENT_ID,
            dataset="psilocybin",
            analysis_type=ANALYSIS_TYPE,
            model_name=MODEL_NAME,
            save_path=f'fig_psilocybin_{MODEL_NAME.replace(" ", "_")}_performance',
            metrics_to_plot=[
                'metric_scores.accuracy.mean',
                'metric_scores.roc_auc.mean',
                'metric_scores.f1.mean'
            ],
            colorbar_label="Mean Score",
            cmap='RdYlBu_r',
            show_sensors=True
        )
        if fig:
            print("✅ Successfully generated performance figure.")
        else:
            print("❌ Figure generation failed.")
            
    except Exception as e:
        print(f"❌ Error generating figure: {e}")
        import traceback
        traceback.print_exc()
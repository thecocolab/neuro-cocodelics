import os
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import mne
import sys
sys.path.append(os.path.dirname(__file__))
from utils import load_aggregated_pickle

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
    Creates a dictionary mapping full channel names to their corresponding data values.
    MNE's plot_topomap can directly use a dictionary for robust plotting.
    """
    sensor_to_value = {name: val for name, val in data_dict.items()}
    
    # Create a mapping from the base sensor name (e.g., 'MRC41') to the full channel name (e.g., 'MRC41-1571')
    base_name_to_full_name = {ch_name.split('-')[0]: ch_name for ch_name in info['ch_names']}
    
    # Map the data values to the full channel names
    data_for_mne = {}
    for base_name, value in sensor_to_value.items():
        if base_name in base_name_to_full_name:
            full_name = base_name_to_full_name[base_name]
            data_for_mne[full_name] = value
            
    return data_for_mne

# --- Core Plotting Function ---

def _plot_topomap_single(ax, data_dict, info, vmin, vmax, cmap, title, plot_style, show_sensors):
    """A private helper function to plot a single topomap."""
    
    # Extract values and names in the correct order for MNE
    ordered_names = info['ch_names']
    data_values = np.array([data_dict.get(name, np.nan) for name in ordered_names])

    # Get layout and create proper mapping
    layout = mne.find_layout(info, ch_type='meg')
    
    # Map data to layout positions by matching base names
    layout_base_names = [name.split('-')[0] if '-' in name else name for name in layout.names]
    info_base_names = [name.split('-')[0] for name in ordered_names]
    
    # Create arrays that match layout dimensions
    layout_data = np.full(len(layout.names), np.nan)
    
    for i, layout_base in enumerate(layout_base_names):
        # Find corresponding data value
        if layout_base in data_dict:
            layout_data[i] = data_dict[layout_base]
        else:
            # Try to find in info mapping
            for j, info_base in enumerate(info_base_names):
                if info_base == layout_base and not np.isnan(data_values[j]):
                    layout_data[i] = data_values[j]
                    break

    # Now plot with properly matched dimensions
    im, _ = mne.viz.plot_topomap(
        layout_data, layout.pos, 
        show=False, 
        vlim=(vmin, vmax), 
        cmap=cmap, 
        axes=ax,
        contours=6,
        sensors=show_sensors,
        sphere=None  # Let MNE determine from layout
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
    plot_titles = list(all_plots_data.keys())
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
    for i, title in enumerate(plot_titles):
        im = _plot_topomap_single(axes[i], all_plots_data[title], info, vmin, vmax, 
                                  cmap, title, plot_style, show_sensors)

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

# --- Example Usage ---
if __name__ == '__main__':
    AGGREGATED_DIR = "/home/sesma/projects/def-kjerbi/data_neurococodelics/aggregated"
    GLOBAL_EXPERIMENT_ID = "neuro_cocodelics_single_sensor_all_features"
    DATASET = "psilocybin"
    ANALYSIS_TYPE = "baseline"
    MODEL_NAME = "Logistic Regression"
    
    try:
        print("\n--- Generating Performance Topomap Grid ---")
        fig = make_performance_topomap_grid(
            aggregated_dir=AGGREGATED_DIR,
            global_experiment_id=GLOBAL_EXPERIMENT_ID,
            dataset=DATASET,
            analysis_type=ANALYSIS_TYPE,
            model_name=MODEL_NAME,
            save_path=f'fig_{DATASET}_{MODEL_NAME.replace(" ", "_")}_performance',
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
            plt.show()
        else:
            print("❌ Figure generation failed.")
            
    except Exception as e:
        print(f"❌ Error generating figure: {e}")
        import traceback
        traceback.print_exc() 
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as mpatches   # for legend handles

# Use a valid matplotlib style
plt.style.use('seaborn-v0_8-colorblind')

# Publication-ready rcParams
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.color': '#dddddd',
    'grid.linestyle': '--',
    'grid.alpha': 0.7
})

# Mapping from raw feature keys to friendly names
MAPPING = {
    'feature_importances.feature-detrendedFluctuationMeanEpochs.spaces': 'DFA',
    'feature_importances.feature-higuchiFdMeanEpochs.spaces': 'Higuchi FD Mean',
    'feature_importances.feature-higuchiFdVarEpochs.spaces': 'Higuchi FD Var',
    'feature_importances.feature-hjorthComplexityMeanEpochs.spaces': 'Hjorth Complexity',
    'feature_importances.feature-hjorthMobilityMeanEpochs.spaces': 'Hjorth Mobility',
    'feature_importances.feature-katzFdMeanEpochs.spaces': 'Katz FD Mean',
    'feature_importances.feature-katzFdSDEpochs.spaces': 'Katz FD SD',
    'feature_importances.feature-lzivComplexityMeanEpochs.spaces': 'LZIV',
    'feature_importances.feature-numZerocrossMeanEpochs.spaces': 'Num Zero Crossings',
    'feature_importances.feature-petrosianFdMeanEpochs.spaces': 'Petrosian FD',
    'feature_importances.feature-spectralEntropyMeanEpochs.spaces': 'Spectral Entropy',
    'feature_importances.feature-svdEntropyMeanEpochs.spaces': 'SVD Entropy',
}

# Manual color mapping for features
cmap = plt.get_cmap('tab20').colors
FEATURE_COLORS = {raw: cmap[i % len(cmap)] for i, raw in enumerate(MAPPING)}

# Pre-build a fixed set of legend handles (one patch per feature, in MAPPING’s order)
LEGEND_HANDLES = [
    mpatches.Patch(color=FEATURE_COLORS[feat], label=name)
    for feat, name in MAPPING.items()
]

def load_results(pickle_path):
    """Load pickle file containing sensor feature importances."""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def parse_feature_data(data):
    """
    Parse raw sensor data dict into {feature: (mean, std)}.
    Takes absolute value of mean importances.
    """
    feature_data = {}
    for key, value in data.items():
        if key.endswith('.mean'):
            feat = key[:-5]
            mean = abs(value)
            std = data.get(f'{feat}.std', 0)
            feature_data[feat] = (mean, std)
    return feature_data

def plot_feature_importances(feature_data, sensor_name=None, save=False, outdir=None):
    """
    Plot feature importances with error bars and custom colors.
    Legend is fixed and consistent across plots.
    """
    # Sort by importance descending
    sorted_items = sorted(feature_data.items(), key=lambda x: x[1][0], reverse=True)
    raw_feats, stats = zip(*sorted_items)
    means, stds = zip(*stats)

    # Pull friendly labels and colors
    colors = [FEATURE_COLORS[f[:-6]] for f in raw_feats]
    labels = [MAPPING.get(f[:-6], f) for f in raw_feats]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(axis='y')

    # Plot bars
    ax.bar(range(len(raw_feats)), means, yerr=stds,
           color=colors, capsize=4)

    # Clean x-axis
    ax.set_xticks([])
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)

    # Fixed legend
    ax.legend(handles=LEGEND_HANDLES,
              title='Features',
              bbox_to_anchor=(1.05, 1),
              loc='upper left')

    # Labels & title
    ax.set_ylabel('Feature Importance')
    title = f"Feature Importances for {sensor_name}" if sensor_name else "Feature Importances"
    ax.set_title(title)

    # Save if requested
    if save:
        filename = f"{sensor_name or 'features'}_importance.png"
        filepath = os.path.join(outdir or os.getcwd(), filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')

    plt.show()

def main():
    save_plots = True
    outdir = os.getcwd()
    # <-- Update this path to point to your actual pickle file:
    pickle_file = "/home/hamza97/scratch/neuro-cocodelics/aggregated/neuro_cocodelics_single_sensor_all_features_tiagabine_baseline_Logistic Regression_feat_importance.pkl"

    results = load_results(pickle_file)

    for idx, (sensor_name, data) in enumerate(results.items()):
        feature_data = parse_feature_data(data)
        plot_feature_importances(feature_data,
                                 sensor_name=sensor_name,
                                 save=save_plots,
                                 outdir=outdir)
        # limit to first 10 sensors
        if idx == 9:
            break

if __name__ == "__main__":
    main()

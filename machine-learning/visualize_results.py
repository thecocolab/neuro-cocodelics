import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

def plot_model_performance(results_data, analysis_id, output_dir):
    """
    Plots model performance metrics for a single analysis.
    """
    plot_data = []
    for model_name, model_results in results_data.items():
        if isinstance(model_results, dict) and 'metric_scores' in model_results:
            for metric, scores in model_results['metric_scores'].items():
                plot_data.append({
                    'analysis_id': analysis_id,
                    'model': model_name,
                    'metric': metric,
                    'score': scores['mean'],
                    'std': scores['std']
                })

    if not plot_data:
        print("No metric data found in the results.")
        return

    df_plot = pd.DataFrame(plot_data)

    # Create the plot
    g = sns.catplot(
        data=df_plot,
        x='model',
        y='score',
        hue='model',
        col='metric',
        kind='bar',
        errorbar='sd',
        palette='viridis',
        height=5,
        aspect=1.2,
        dodge=False
    )
    
    g.fig.suptitle(f'Model Performance for {analysis_id}', y=1.03)
    g.set_axis_labels("Model", "Score")
    g.set_titles("{col_name}")
    g.despine(left=True)

    # Save the figure
    figure_path = os.path.join(output_dir, f"{analysis_id}_model_performance.png")
    plt.savefig(figure_path, bbox_inches='tight')
    print(f"Figure saved to {figure_path}")
    plt.close()

def plot_feature_importance(results_data, analysis_id, output_dir):
    """
    Plots feature importances for each model in a single analysis.
    """
    for model_name, model_results in results_data.items():
        if isinstance(model_results, dict) and 'feature_importances' in model_results and model_results['feature_importances']:
            importances = model_results['feature_importances']
            
            # Check if importances are in the expected format
            if 'mean' not in importances or 'feature_names' not in importances:
                print(f"Skipping feature importance plot for {model_name} in {analysis_id}: data format incorrect.")
                continue

            df_imp = pd.DataFrame({
                'feature': importances['feature_names'],
                'importance': importances['mean']
            }).sort_values(by='importance', ascending=False).head(20) # Top 20 features

            if df_imp.empty:
                continue

            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=df_imp, palette='rocket')
            plt.title(f'Feature Importance for {model_name} in {analysis_id}')
            plt.tight_layout()
            
            figure_path = os.path.join(output_dir, f"{analysis_id}_{model_name}_feature_importance.png")
            plt.savefig(figure_path)
            print(f"Figure saved to {figure_path}")
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize ML experiment results.")
    parser.add_argument(
        "--results", "-r", 
        required=True, 
        help="Path to a results pickle file."
    )
    parser.add_argument(
        "--aggregated",
        action="store_true",
        help="Set this flag if the results file is an aggregated pickle with multiple analyses."
    )
    args = parser.parse_args()
    
    try:
        results_data = pd.read_pickle(args.results)
    except FileNotFoundError:
        print(f"Error: Results file not found at {args.results}")
        return

    output_dir = os.path.dirname(args.results)
    
    if args.aggregated:
        # Aggregated file with multiple analyses
        for analysis_id, analysis_results in results_data.items():
            plot_model_performance(analysis_results, analysis_id, output_dir)
            plot_feature_importance(analysis_results, analysis_id, output_dir)
    else:
        # Single analysis file
        filename = os.path.basename(args.results)
        analysis_id = re.sub(r'\.pkl$', '', filename) # Clean up filename for title
        plot_model_performance(results_data, analysis_id, output_dir)
        plot_feature_importance(results_data, analysis_id, output_dir)


if __name__ == "__main__":
    main() 
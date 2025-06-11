import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import os
import re
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# --- GLOBAL RC SETTINGS FOR PUBLISHING ---
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,          # high-res
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Set publication-quality style and color-blind friendly palette
plt.style.use('seaborn-v0_8-whitegrid')
palette = sns.color_palette("colorblind")
sns.set_palette(palette)

def debug_results_structure(results_data, analysis_id="Unknown"):
    """
    Debug function to inspect the structure of results data.
    """
    print(f"\n=== DEBUGGING RESULTS STRUCTURE FOR {analysis_id} ===")
    print(f"Results data type: {type(results_data)}")
    
    if isinstance(results_data, dict):
        print(f"Top-level keys: {list(results_data.keys())}")
        
        for key, value in results_data.items():
            print(f"\n--- Model/Key: {key} ---")
            print(f"Value type: {type(value)}")
            
            if isinstance(value, dict):
                print(f"Sub-keys: {list(value.keys())}")
                
                # Check specific keys we're interested in
                for subkey in ['metric_scores', 'feature_importances', 'predictions', 'selected_features']:
                    if subkey in value:
                        print(f"  {subkey}: {type(value[subkey])}")
                        if isinstance(value[subkey], dict):
                            print(f"    Keys: {list(value[subkey].keys())}")
                        elif hasattr(value[subkey], 'shape'):
                            print(f"    Shape: {value[subkey].shape}")
    print("=" * 60)


def plot_model_performance(results_data, analysis_id, output_dir):
    """
    Plots model performance metrics for a single analysis with publication quality.
    """
    plot_data = []
    cv_data = []  # For cross-validation score distributions
    
    for model_name, model_results in results_data.items():
        if isinstance(model_results, dict) and 'metric_scores' in model_results:
            for metric, scores in model_results['metric_scores'].items():
                # Main performance plot data
                plot_data.append({
                    'analysis_id': analysis_id,
                    'model': model_name,
                    'metric': metric,
                    'score': scores['mean'],
                    'std': scores['std']
                })
                
                # CV fold scores for distribution plots
                if 'fold_scores' in scores:
                    for fold_idx, fold_score in enumerate(scores['fold_scores']):
                        cv_data.append({
                            'model': model_name,
                            'metric': metric,
                            'fold': fold_idx,
                            'score': fold_score
                        })

    if not plot_data:
        print("No metric data found in the results.")
        return

    df_plot = pd.DataFrame(plot_data)
    df_cv = pd.DataFrame(cv_data)

    # 1. Main performance comparison plot
    n_metrics = len(df_plot['metric'].unique())
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(df_plot['metric'].unique()):
        metric_data = df_plot[df_plot['metric'] == metric]
        
        ax = axes[i]
        
        # Use color-blind friendly colors
        bars = ax.bar(range(len(metric_data)), metric_data['score'], 
                     yerr=metric_data['std'], capsize=6, alpha=0.8,
                     color=[palette[j] for j in range(len(metric_data))],
                     edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars with better positioning
        for j, (bar, score, std) in enumerate(zip(bars, metric_data['score'], metric_data['std'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_title(f'{metric.upper()}', fontsize=18, fontweight='bold', pad=20)
        ax.set_ylabel('Score', fontsize=16, fontweight='bold')
        ax.set_xlabel('Model', fontsize=16, fontweight='bold')
        ax.set_xticks(range(len(metric_data)))
        ax.set_xticklabels(metric_data['model'], rotation=30, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, min(1.0, metric_data['score'].max() + metric_data['std'].max() + 0.1))
    
    plt.suptitle(f'Model Performance Comparison', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    figure_path = os.path.join(output_dir, f"{analysis_id}_model_performance.png")
    plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Performance plot saved to {figure_path}")
    plt.close()

    # 2. Cross-validation score distributions with improved styling
    if not df_cv.empty:
        n_metrics = len(df_cv['metric'].unique())
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        # Determine global y-axis limits for consistency
        global_min = df_cv['score'].min() - 0.05
        global_max = df_cv['score'].max() + 0.05
        
        for i, metric in enumerate(df_cv['metric'].unique()):
            metric_cv_data = df_cv[df_cv['metric'] == metric]
            
            ax = axes[i]
            
            # Use different colors for different models
            sns.boxplot(data=metric_cv_data, x='model', y='score', ax=ax, 
                       palette=palette[:len(metric_cv_data['model'].unique())],
                       width=0.6)
            sns.stripplot(data=metric_cv_data, x='model', y='score', ax=ax, 
                         color='darkred', alpha=0.8, size=6, jitter=0.2,
                         edgecolor='white', linewidth=0.5)
            
            ax.set_title(f'CV Score Distribution - {metric.upper()}', fontsize=18, fontweight='bold', pad=20)
            ax.set_ylabel('Score', fontsize=16, fontweight='bold')
            ax.set_xlabel('Model', fontsize=16, fontweight='bold')
            ax.tick_params(axis='x', rotation=0)
            ax.grid(False, axis='y')
            ax.set_ylim(global_min, global_max)
        
        plt.suptitle(f'Cross-Validation Score Distributions', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        cv_figure_path = os.path.join(output_dir, f"{analysis_id}_cv_distributions.png")
        plt.savefig(cv_figure_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"CV distributions plot saved to {cv_figure_path}")
        plt.close()


def plot_feature_importance(results_data, analysis_id, output_dir):
    """
    Plots feature importances for each model with enhanced styling.
    """
    for model_name, model_results in results_data.items():
        if isinstance(model_results, dict) and 'feature_importances' in model_results:
            importances = model_results['feature_importances']
            
            if not importances:
                print(f"No feature importances for {model_name}")
                continue
            
            # Try different possible structures for feature importances
            feature_names = None
            importance_values = None
            
            if isinstance(importances, dict):
                if 'mean' in importances and 'feature_names' in importances:
                    feature_names = importances['feature_names']
                    importance_values = importances['mean']
                elif 'weighted_mean' in importances and 'feature_names' in importances:
                    feature_names = importances['feature_names']
                    importance_values = importances['weighted_mean']
                else:
                    print(f"Feature importance format not recognized for {model_name}:")
                    print(f"Available keys: {list(importances.keys())}")
                    continue
            elif isinstance(importances, np.ndarray):
                importance_values = importances
                # Generate generic feature names if not available
                feature_names = [f"Feature_{i}" for i in range(len(importance_values))]
            else:
                print(f"Unsupported feature importance type: {type(importances)}")
                continue

            if feature_names is None or importance_values is None:
                print(f"Could not extract feature names/values for {model_name}")
                continue

            # Create DataFrame and sort by importance
            df_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_values
            }).sort_values(by='importance', ascending=False).head(20)  # Top 20 features

            if df_imp.empty:
                continue

            plt.figure(figsize=(12, 10))
            
            # Create horizontal bar plot with gradient colors
            y_pos = np.arange(len(df_imp))
            colors = plt.cm.viridis(np.linspace(0, 1, len(df_imp)))
            
            bars = plt.barh(y_pos, df_imp['importance'], alpha=0.8, 
                           color=colors, edgecolor='black', linewidth=0.5)
            
            plt.yticks(y_pos, df_imp['feature'], fontsize=12)
            plt.xlabel('Importance Score', fontsize=16, fontweight='bold')
            plt.ylabel('Features', fontsize=16, fontweight='bold')
            plt.title(f'Top 20 Feature Importances - {model_name}', 
                     fontsize=18, fontweight='bold', pad=20)
            
            # Add value labels on bars
            for i, (bar, importance) in enumerate(zip(bars, df_imp['importance'])):
                plt.text(bar.get_width() + importance*0.01, bar.get_y() + bar.get_height()/2,
                        f'{importance:.4f}', va='center', fontweight='bold', fontsize=10)
            
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            
            figure_path = os.path.join(output_dir, f"{analysis_id}_{model_name}_feature_importance.png")
            plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Feature importance plot saved to {figure_path}")
            plt.close()


def plot_feature_selection_frequency(results_data, analysis_id, output_dir):
    """
    Plots feature selection frequency with enhanced styling.
    """
    for model_name, model_results in results_data.items():
        if isinstance(model_results, dict) and 'feature_frequency' in model_results:
            freq_data = model_results['feature_frequency']
            
            if not freq_data:
                continue
                
            df_freq = pd.DataFrame(list(freq_data.items()), 
                                 columns=['feature', 'frequency']).sort_values(
                                     by='frequency', ascending=False).head(20)
            
            if df_freq.empty:
                continue

            plt.figure(figsize=(12, 10))
            
            # Create horizontal bar plot with distinct colors
            y_pos = np.arange(len(df_freq))
            colors = plt.cm.plasma(np.linspace(0, 1, len(df_freq)))
            
            bars = plt.barh(y_pos, df_freq['frequency'], alpha=0.8,
                           color=colors, edgecolor='black', linewidth=0.5)
            
            plt.yticks(y_pos, df_freq['feature'], fontsize=12)
            plt.xlabel('Selection Frequency (across CV folds)', fontsize=16, fontweight='bold')
            plt.ylabel('Features', fontsize=16, fontweight='bold')
            plt.title(f'Feature Selection Frequency - {model_name}', 
                     fontsize=18, fontweight='bold', pad=20)
            
            # Add frequency labels
            for bar, freq in zip(bars, df_freq['frequency']):
                plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{freq}', va='center', fontweight='bold', fontsize=11)
            
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            
            figure_path = os.path.join(output_dir, f"{analysis_id}_{model_name}_feature_selection_freq.png")
            plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Feature selection frequency plot saved to {figure_path}")
            plt.close()


def plot_roc_curves(results_data, analysis_id, output_dir):
    """
    Plots ROC curves with publication-quality styling.
    """
    plt.figure(figsize=(10, 8))
    
    for i, (model_name, model_results) in enumerate(results_data.items()):
        if isinstance(model_results, dict) and 'predictions' in model_results:
            predictions = model_results['predictions']
            
            if 'y_true' in predictions and 'y_proba' in predictions:
                y_true = predictions['y_true']
                y_proba = predictions['y_proba']
                
                # Handle binary classification
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    y_proba = y_proba[:, 1]  # Probability of positive class
                elif y_proba.ndim == 1:
                    pass  # Already 1D
                else:
                    print(f"Skipping ROC for {model_name}: unsupported probability shape {y_proba.shape}")
                    continue
                
                try:
                    fpr, tpr, _ = roc_curve(y_true, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    # Use thicker lines and distinct colors
                    plt.plot(fpr, tpr, linewidth=3, color=palette[i % len(palette)],
                            label=f'{model_name} (AUC = {roc_auc:.3f})',
                            marker='o', markersize=4, markevery=0.1, alpha=0.8)
                except Exception as e:
                    print(f"Error plotting ROC for {model_name}: {e}")
                    continue
    
    # Random classifier line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=16, fontweight='bold')
    plt.title('ROC Curves', fontsize=18, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=13, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    figure_path = os.path.join(output_dir, f"{analysis_id}_roc_curves.png")
    plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"ROC curves plot saved to {figure_path}")
    plt.close()


def plot_confusion_matrices(results_data, analysis_id, output_dir):
    """
    Plots confusion matrices with shared colorbar and improved styling.
    """
    n_models = len(results_data)
    if n_models == 0:
        return
        
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    # Shared colorbar settings
    vmin, vmax = 0, 1
    
    for idx, (model_name, model_results) in enumerate(results_data.items()):
        if isinstance(model_results, dict) and 'predictions' in model_results:
            predictions = model_results['predictions']
            
            if 'y_true' in predictions and 'y_pred' in predictions:
                y_true = predictions['y_true']
                y_pred = predictions['y_pred']
                
                cm = confusion_matrix(y_true, y_pred)
                
                # Normalize confusion matrix
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                ax = axes[idx]
                im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues',
                              vmin=vmin, vmax=vmax, aspect='equal')
                ax.set_title(f'{model_name}', fontsize=16, fontweight='bold', pad=15)
                
                # Remove background grid and spines for a cleaner heatmap
                ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                # Add text annotations with better contrast
                thresh = cm_normalized.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        text_color = "white" if cm_normalized[i, j] > thresh else "black"
                        ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                               ha="center", va="center", fontweight='bold', fontsize=12,
                               color=text_color)
                
                ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
                ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
                
                # Set tick labels
                classes = np.unique(np.concatenate([y_true, y_pred]))
                ax.set_xticks(range(len(classes)))
                ax.set_yticks(range(len(classes)))
                ax.set_xticklabels(classes, fontsize=12)
                ax.set_yticklabels(classes, fontsize=12)
    
    # Add shared colorbar to the right of all subplots
    fig.tight_layout(rect=[0,0,0.88,1])  # make room on the right
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Row-normalized proportion", fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    plt.suptitle('Confusion Matrices', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    
    figure_path = os.path.join(output_dir, f"{analysis_id}_confusion_matrices.png")
    plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Confusion matrices plot saved to {figure_path}")
    plt.close()


def generate_summary_report(results_data, analysis_id, output_dir):
    """
    Generate a text summary report of the analysis.
    """
    report_lines = []
    report_lines.append(f"ML Analysis Summary Report: {analysis_id}")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Model performance summary
    report_lines.append("MODEL PERFORMANCE SUMMARY:")
    report_lines.append("-" * 30)
    
    for model_name, model_results in results_data.items():
        if isinstance(model_results, dict) and 'metric_scores' in model_results:
            report_lines.append(f"\n{model_name}:")
            for metric, scores in model_results['metric_scores'].items():
                report_lines.append(f"  {metric}: {scores['mean']:.4f} ± {scores['std']:.4f}")
    
    # Feature selection summary (if available)
    report_lines.append("\n\nFEATURE SELECTION SUMMARY:")
    report_lines.append("-" * 30)
    
    for model_name, model_results in results_data.items():
        if isinstance(model_results, dict) and 'selected_features' in model_results:
            selected = model_results['selected_features']
            report_lines.append(f"\n{model_name}:")
            report_lines.append(f"  Number of features selected: {len(selected)}")
            report_lines.append(f"  Selected features: {selected[:10]}{'...' if len(selected) > 10 else ''}")
    
    report_lines.append("\n" + "=" * 60)
    
    # Save report
    report_path = os.path.join(output_dir, f"{analysis_id}_summary_report.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Summary report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive ML experiment results visualization.")
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to inspect data structure."
    )
    parser.add_argument(
        "--plots",
        nargs='+',
        default=['all'],
        choices=['performance', 'cv_dist', 'feature_importance', 'feature_selection', 'roc', 'confusion', 'all'],
        help="Specify which plots to generate (default: all)"
    )
    
    args = parser.parse_args()
    
    try:
        results_data = pd.read_pickle(args.results)
    except FileNotFoundError:
        print(f"Error: Results file not found at {args.results}")
        return
    except Exception as e:
        print(f"Error loading results file: {e}")
        return

    output_dir = os.path.dirname(args.results)
    
    def run_visualization(analysis_results, analysis_id):
        """Run visualization for a single analysis."""
        if args.debug:
            debug_results_structure(analysis_results, analysis_id)
        
        plot_functions = {
            'performance': plot_model_performance,
            'feature_importance': plot_feature_importance,
            'feature_selection': plot_feature_selection_frequency,
            'roc': plot_roc_curves,
            'confusion': plot_confusion_matrices,
        }
        
        plots_to_run = args.plots if 'all' not in args.plots else list(plot_functions.keys())
        
        for plot_type in plots_to_run:
            if plot_type in plot_functions:
                try:
                    plot_functions[plot_type](analysis_results, analysis_id, output_dir)
                except Exception as e:
                    print(f"Error generating {plot_type} plot: {e}")
        
        # Always generate summary report
        try:
            generate_summary_report(analysis_results, analysis_id, output_dir)
        except Exception as e:
            print(f"Error generating summary report: {e}")
    
    if args.aggregated:
        # Aggregated file with multiple analyses
        for analysis_id, analysis_results in results_data.items():
            print(f"\nProcessing analysis: {analysis_id}")
            run_visualization(analysis_results, analysis_id)
    else:
        # Single analysis file
        filename = os.path.basename(args.results)
        analysis_id = re.sub(r'\.pkl$', '', filename)
        print(f"\nProcessing single analysis: {analysis_id}")
        run_visualization(results_data, analysis_id)
    
    print(f"\nVisualization complete! All figures saved to: {output_dir}")


if __name__ == "__main__":
    main() 
    
    #python machine-learning/visualize_results.py --results results/lsd_closed/lsd_closed_baseline_binary_baseline_rs42.pkl
# fetch results pickle from /home/hamza97/projects/neuro-cocodelics/results
import os
import pickle
import pandas as pd
import numpy as np
import json

def load_single_pickle(results_dir, global_experiment_id, analysis_type, task_type, random_state, model_name=None, slice_name=None):
    """
    Load the results pickle file based on the provided experiment configuration.
    Loads single model results for a specific experiment if the model is not None.
       f"{model_name}_{global_experiment_id}_{task_type}_{analysis_type}_rs{random_state}.pkl"
    Loads the final results for the experiment if the model is None.
       f"{global_experiment_id}_{task_type}_{analysis_type}_rs{random_state}.pkl"
    If slice_name is provided, it will load the results for that specific slice.
      f"{global_experiment_id}_{slice_name}_{task_type}_{analysis_type}_rs{random_state}.pkl"
      f"{model_name}_{global_experiment_id}_{slice_name}_{task_type}_{analysis_type}_rs{random_state}.pkl"

    Parameters:
    - results_dir (str): Directory where the results pickle files are stored.
    - global_experiment_id (str): Unique identifier for the experiment.
    - analysis_type (str): Type of analysis performed (e.g., "baseline", "feature_selection").
    - task_type (str): Type of task (e.g., "binary", "multiclass").
    - random_state (int): Random seed used for reproducibility.
    - model_name (str): Name of the model used in the experiment (e.g., "SVC", "Random Forest"). If None, loads final results.
    - slice_name (str): Optional name of a specific slice to load results for. If provided, it will modify the filename accordingly.

    Returns:
    - The loaded results object.
    """
    # Construct the pickle filename by combining results_file, experiment, and model identifiers.
    if slice_name is not None:
        global_experiment_id = f"{global_experiment_id}_{slice_name}"
    pickle_filename = f"{global_experiment_id}_{task_type}_{analysis_type}_rs{random_state}.pkl"

    if model_name is not None:
        pickle_filename = f"{model_name}_{pickle_filename}"
    pickle_path = os.path.join(results_dir, pickle_filename)
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Results file not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)
    return results
    
def load_aggregated_pickle(aggregated_dir, global_experiment_id, dataset, analysis_type, model_name):
    """
    Load the aggregated results pickle file.
    Filename format:
    f"{global_experiment_id}_{dataset}_{analysis_type}_{model_name}_perf_metrics.pkl"

    Parameters:
    - aggregated_dir (str): Directory where the aggregated results pickle files are stored.
    - global_experiment_id (str): Unique identifier for the experiment.
    - dataset (str): The dataset name (e.g., "ketamine", "lsd-avg").
    - analysis_type (str): Type of analysis performed (e.g., "baseline").
    - model_name (str): Name of the model used in the experiment.

    Returns:
    - The loaded results object.
    """
    pickle_filename = f"{global_experiment_id}_{dataset}_{analysis_type}_{model_name}_perf_metrics.pkl"
    pickle_path = os.path.join(aggregated_dir, pickle_filename)
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Aggregated results file not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)
    return results

def extract_results(results, keys=None, include_fold_scores=False, include_fold_preds=False, include_fold_estimators=False,
                include_fold_feature_importances=False):
    """
    Extract slices from `results` by key-path.

    Parameters
    ----------
    results : dict
        The full dict you loaded from pickle.
    keys : list of str, optional
        If provided, should be a list of top-level or nested key-paths,
        e.g. ['model_name',
               'metric_scores.accuracy.mean',
               'predictions.y_true'].
        Returns exactly those paths.
    include_fold_scores : bool
        If False, will drop any `fold_scores` under metric_scores.*.
    include_fold_preds : bool
        If False, will drop the `fold_preds` entry under predictions.

    Returns
    -------
    dict
        A new dict containing only the requested slices.
    """
    def get_by_path(d, path):
        for part in path.split('.'):
            d = d[part]
        return d

    # If user explicitly listed key-paths, just grab those
    if keys is not None:
        out = {}
        for path in keys:
            out[path] = get_by_path(results, path)
        return out

    # Otherwise, return all top-level keys,
    # but optionally strip out fold-level arrays
    out = {}
    for top, val in results.items():
        if (top == 'folds_estimators' and include_fold_estimators==False) or top == "model_name":
            continue
        if top == 'metric_scores':
            trimmed = {}
            for metric, mv in val.items():
                trimmed[metric] = {
                    k: v
                    for k, v in mv.items()
                    if include_fold_scores or k != 'fold_scores'
                }
            out[top] = trimmed

        elif top == 'predictions':
            trimmed = {
                k: v
                for k, v in val.items()
                if include_fold_preds or k != 'fold_preds'
            }
            out[top] = trimmed

        elif top == 'feature_importances':
            trimmed = {
                k: v
                for k, v in val.items()
                if include_fold_feature_importances or k != 'fold_importances'
            }
        else:
            out[top] = val

    return out

def aggregate_model_results(slices, results_dir, global_experiment_id, analysis_type, task_type, random_state, model_name,
        paths):
    aggregate_perf = {}
    aggregate_feat_import = {}

    for slice in slices:
        slice_results = load_single_pickle(results_dir, global_experiment_id, analysis_type, task_type,  random_state, model_name, slice)
        slice_results_extracted_performance = extract_results(slice_results, keys=paths_perfer)
        slice_results_extracted_performance = extract_results(slice_results, keys=paths_feat_imp)    
        aggregate[slice] = slice_results_extracted
        # aggregate_feat_import[]

    return aggregate

def save(file, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump( file, f)


if __name__ == "__main__":
    # Example of loading a single, non-aggregated file
    # results = load_single_pickle(
    #     results_dir="/home/hamza97/scratch/neuro-cocodelics/results",
    #     global_experiment_id="neuro_cocodelics_single_sensor_all_features_tiagabine",
    #     analysis_type="baseline",
    #     task_type="binary",
    #     slice_name="MZP01",
    #     random_state=42,
    #     model_name="Random Forest"
    # )
    # print("--- Single Result ---")
    # print(results)

    # Example of loading an aggregated file
    aggregated_results = load_aggregated_pickle(
        aggregated_dir="/home/sesma/projects/def-kjerbi/data_neurococodelics/aggregated",
        global_experiment_id="neuro_cocodelics_single_sensor_all_features",
        dataset="ketamine",
        analysis_type="baseline",
        model_name="Gradient Boosting"
    )
    print("\n--- Inspected Aggregated Result (ketamine, Gradient Boosting) ---")
    
    print(f"\nType of loaded data: {type(aggregated_results)}")
    
    if isinstance(aggregated_results, dict):
        keys = list(aggregated_results.keys())
        print(f"\nNumber of sensors (top-level keys): {len(keys)}")
        print(f"Example sensor names: {keys[:5]}")

        first_sensor_key = keys[0]
        print(f"\nShowing structure for one sensor ('{first_sensor_key}'):")
        
        print(json.dumps(aggregated_results[first_sensor_key], indent=4))





# fetch results pickle from /home/hamza97/projects/neuro-cocodelics/results
import os
import pickle

def load_results_pickle(results_dir, global_experiment_id, analysis_type, task_type, random_state, model_name=None):
    """
    Load the results pickle file based on the provided experiment configuration.
    Loads single model results for a specific experiment if the model is not None.
       f"{model_name}_{global_experiment_id}_{task_type}_{analysis_type}_{random_state}.pkl"
    Loads the final results for the experiment if the model is None.
       f"{global_experiment_id}_{task_type}_{analysis_type}_{random_state}.pkl"

    Parameters:
    - results_dir (str): Directory where the results pickle files are stored.
    - global_experiment_id (str): Unique identifier for the experiment.
    - analysis_type (str): Type of analysis performed (e.g., "baseline", "feature_selection").
    - task_type (str): Type of task (e.g., "binary", "multiclass").
    - random_state (int): Random seed used for reproducibility.
    - model_name (str): Name of the model used in the experiment (e.g., "SVC", "Random Forest"). If None, loads final results.

    Returns:
    - The loaded results object.
    """
    # Construct the pickle filename by combining results_file, experiment, and model identifiers.
    pickle_filename = f"{global_experiment_id}_{task_type}_{analysis_type}_{random_state}.pkl"

    if model_name is not None:
        pickle_filename = f"{model_name}_{pickle_filename}"
    pickle_path = os.path.join(results_dir, pickle_filename)
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Results file not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)
    
    return results


if __name__ == "__main__":
    results = load_results_pickle(
        results_dir="/home/hamza97/projects/neuro-cocodelics/results",
        global_experiment_id="neuro_cocodelics_single_sensor_all_features_ketamine",
        analysis_type="baseline",
        task_type="binary",
        random_state=42,
        model_name="Random Forest"
    )
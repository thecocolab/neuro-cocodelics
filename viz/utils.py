


# fetch results pickle from /home/hamza97/projects/neuro-cocodelics/results
import os
import pickle

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
        aggregate_feat_import[]

    return aggregate

def save(file, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump( file, f)


if __name__ == "__main__":
    results = load_single_pickle(
        results_dir="/home/hamza97/scratch/neuro-cocodelics/results",
        global_experiment_id="neuro_cocodelics_single_sensor_all_features_tiagabine",
        analysis_type="baseline",
        task_type="binary",
        slice_name="MZP01",
        random_state=42,
        model_name="Random Forest"
    )
    paths = [
        'metric_scores.accuracy.mean',
        'metric_scores.accuracy.std',
        'metric_scores.roc_auc.mean',
        'metric_scores.roc_auc.std',
        'metric_scores.f1.mean',
        'metric_scores.f1.std',
    ]
    paths_importances = [
        f'feature_importances.feature-detrendedFluctuationMeanEpochs.spaces-{slice}.mean',
        f'feature_importances.feature-detrendedFluctuationMeanEpochs.spaces-{slice}.std',
        f'feature_importances.feature-higuchiFdMeanEpochs.spaces-{slice}.mean',
        f'feature_importances.feature-higuchiFdMeanEpochs.spaces-{slice}.std',
        f'feature_importances.feature-higuchiFdVarEpochs.spaces-{slice}.mean',
        f'feature_importances.feature-higuchiFdVarEpochs.spaces-{slice}.std',
        f'feature_importances.feature-hjorthComplexityMeanEpochs.spaces-{slice}.mean',
        f'feature_importances.feature-hjorthComplexityMeanEpochs.spaces-{slice}.std',
        f'feature_importances.feature-hjorthMobilityMeanEpochs.spaces-{slice}.mean',
        f'feature_importances.feature-hjorthMobilityMeanEpochs.spaces-{slice}.std',
        f'feature_importances.feature-katzFdMeanEpochs.spaces-{slice}.mean',
        f'feature_importances.feature-katzFdMeanEpochs.spaces-{slice}.std',
        f'feature_importances.feature-katzFdSDEpochs.spaces-{slice}.mean',
        f'feature_importances.feature-katzFdSDEpochs.spaces-{slice}.std',
        f'feature_importances.feature-lzivComplexityMeanEpochs.spaces-{slice}.mean',
        f'feature_importances.feature-lzivComplexityMeanEpochs.spaces-{slice}.std',
        f'feature_importances.feature-multiscaleEntropyMeanEpochs.spaces-{slice}.mean',
        f'feature_importances.feature-multiscaleEntropyMeanEpochs.spaces-{slice}.std',
        f'feature_importances.feature-numZerocrossMeanEpochs.spaces-{slice}.mean',
        f'feature_importances.feature-numZerocrossMeanEpochs.spaces-{slice}.std',
        f'feature_importances.feature-petrosianFdMeanEpochs.spaces-{slice}.mean',
        f'feature_importances.feature-petrosianFdMeanEpochs.spaces-{slice}.std',
        f'feature_importances.feature-spectralEntropyMeanEpochs.spaces-{slice}.mean',
        f'feature_importances.feature-spectralEntropyMeanEpochs.spaces-{slice}.std',
        f'feature_importances.feature-svdEntropyMeanEpochs.spaces-{slice}.mean',
        f'feature_importances.feature-svdEntropyMeanEpochs.spaces-{slice}.std',
    ]
    slices = ['MRC41', 'MLF61', 'MLC15', 'MRP32', 'MRT41', 'MLF41', 'MLT55', 'MRF25', 'MRC16', 'MRO14', 'MRF33', 'MRT53', 'MRC63', 'MLF32', 'MRF23', 'MRT56', 'MRO32', 'MZC04', 'MRC53', 'MRT14', 'MRP33', 'MLF67', 'MLC31', 'MLO24', 'MLT56', 'MLP54', 'MRT51', 'MRC31', 'MLO34', 'MLP35', 'MZF01', 'MZF03', 'MLC14', 'MRF66', 'MRC23', 'MLO44', 'MRT42', 'MRT46', 'MRF14', 'MRO44', 'MZC02', 'MLT25', 'MRP21', 'MRP52', 'MRP35', 'MRT13', 'MRP12', 'MRO42', 'MRF13', 'MLC12', 'MLT33', 'MRC55', 'MRF52', 'MRF54', 'MRT23', 'MLC16', 'MRC42', 'MLO33', 'MRP11', 'MLT57', 'MLT14', 'MRP34', 'MRC14', 'MRC15', 'MLF12', 'MRF46', 'MLF35', 'MLF63', 'MZC01', 'MLC25', 'MLF11', 'MLT16', 'MRO53', 'MRT27', 'MLP51', 'MRC51', 'MLT47', 'MLC17', 'MLP53', 'MLP12', 'MLT27', 'MRT24', 'MRF62', 'MRT45', 'MRF55', 'MLP52', 'MLO52', 'MRT12', 'MRT55', 'MLC62', 'MLP55', 'MLO53', 'MRF42', 'MLF56', 'MLF25', 'MLP11', 'MRT26', 'MLT54', 'MLT23', 'MLC32', 'MLP32', 'MRO43', 'MRT35', 'MRP22', 'MLP41', 'MLP22', 'MRT44', 'MLF65', 'MRF12', 'MLF23', 'MLT41', 'MRF64', 'MRO12', 'MRT43', 'MRC12', 'MRF61', 'MZP01', 'MRF45', 'MLT15', 'MRF44', 'MLP57', 'MRT47', 'MRF24', 'MRP57', 'MZO03', 'MLF14', 'MRP51', 'MRT22', 'MLT12', 'MRT52', 'MRT34', 'MLF43', 'MLF33', 'MRF56', 'MLT52', 'MLO13', 'MRC61', 'MLF64', 'MRC54', 'MLF53', 'MLC11', 'MLO14', 'MLF54', 'MLF55', 'MRP45', 'MLT21', 'MLF24', 'MLT35', 'MLT42', 'MLC63', 'MLC61', 'MLF13', 'MRF21', 'MLF44', 'MRC62', 'MRP54', 'MRF22', 'MRC32', 'MRO31', 'MRP42', 'MRP56', 'MLO43', 'MLC23', 'MLC54', 'MRC52', 'MRP43', 'MZO01', 'MLO41', 'MRF65', 'MLF66', 'MLF22', 'MRC21', 'MLC52', 'MRT31', 'MRF41', 'MLO22', 'MLP33', 'MLC24', 'MLF46', 'MLO31', 'MLP44', 'MLP45', 'MLP56', 'MRT33', 'MRT32', 'MLT24', 'MLT31', 'MRC11', 'MRT57', 'MZO02', 'MLF42', 'MRC17', 'MLT46', 'MLO42', 'MLF21', 'MLP23', 'MRF34', 'MZC03', 'MLC22', 'MLO51', 'MRO33', 'MLC53', 'MRF32', 'MLO21', 'MRF35', 'MRP55', 'MLC21', 'MRF31', 'MRP41', 'MRF43', 'MLO12', 'MRF51', 'MLF62', 'MRF53', 'MLP42', 'MRO13', 'MLF34', 'MZF02', 'MLT26', 'MLC51', 'MRC13', 'MLF51', 'MRO34', 'MRC22', 'MLT53', 'MRO51', 'MRC24', 'MRP53', 'MRO22', 'MLC13', 'MLO23', 'MLP21', 'MLP43', 'MRC25', 'MLP34', 'MLT51', 'MLT34', 'MLF52', 'MRP23', 'MLT11', 'MLF31', 'MRF67', 'MRT25', 'MRT37', 'MLP31', 'MRP44', 'MLT43', 'MRT11', 'MLC41', 'MRO52', 'MLO32', 'MRO11', 'MRO24', 'MLC55', 'MRO23', 'MLF45', 'MRT21', 'MLT45', 'MRT15', 'MRT16', 'MRP31', 'MRF63', 'MLC42', 'MRO21', 'MRT54', 'MLT22', 'MRF11', 'MLT13', 'MLT44']

    aggregated = aggregate_model_results(slices, 
        results_dir="/home/hamza97/scratch/neuro-cocodelics/results",
        global_experiment_id="neuro_cocodelics_single_sensor_all_features_tiagabine",
        analysis_type="baseline",
        task_type="binary",
        random_state=42,
        model_name="SVC",
        paths=paths)
    global_experiment_id="neuro_cocodelics_single_sensor_all_features"
    analysis_type="baseline"
    task_type="binary"
    random_state=42
    model_name="SVC"
    for model_name in ["SVC", "Random Forest", "Logistic Regression", "Gradient Boosting"]:
        for dataset in ["lsd-Video", "lsd-Music", "lsd-Open2", "lsd-Open1", "lsd-Closed1", "lsd-Closed2", "lsd-avg", "tiagabine", "perampanel", "ketamine","psilocybin"]:
            aggregated_path = os.path.join("/home/hamza97/scratch/neuro-cocodelics/aggregated/", f"{global_experiment_id}_{dataset}_{analysis_type}_{model_name}_perf_metrics.pkl")
            aggregated = aggregate_model_results(
                    results_dir="/home/hamza97/scratch/neuro-cocodelics/results",
                    global_experiment_id=f"{global_experiment_id}_{dataset}",
                    analysis_type="baseline",
                    task_type="binary",
                    random_state=42,
                    model_name=model_name,
                    slices=slices,
                    paths=paths)
            save(aggregated, aggregated_path)
            print(f" Done {model_name} {dataset}")
    


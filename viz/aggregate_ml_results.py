#!/usr/bin/env python3
"""
Utilities for loading, extracting, and aggregating experiment results from pickle files.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def load_single_pickle(
    results_dir: str,
    global_experiment_id: str,
    analysis_type: str,
    task_type: str,
    random_state: int,
    model_name: Optional[str] = None,
    slice_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load the results dict from a pickle file constructed from the experiment identifiers.

    Filename patterns:
    - With slice & model:   {model_name}_{exp_id}_{slice_name}_{task_type}_{analysis_type}_rs{random_state}.pkl
    - With slice only:      {exp_id}_{slice_name}_{task_type}_{analysis_type}_rs{random_state}.pkl
    - With model only:      {model_name}_{exp_id}_{task_type}_{analysis_type}_rs{random_state}.pkl
    - Neither (final):      {exp_id}_{task_type}_{analysis_type}_rs{random_state}.pkl

    """
    exp_id = global_experiment_id
    if slice_name:
        exp_id = f"{exp_id}_{slice_name}"
    filename = f"{exp_id}_{task_type}_{analysis_type}_rs{random_state}.pkl"
    if model_name:
        filename = f"{model_name}_{filename}"
    file_path = Path(results_dir) / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")
    with file_path.open("rb") as f:
        return pickle.load(f)


def extract_results(
    results: Dict[str, Any],
    keys: Optional[List[str]] = None,
    include_fold_scores: bool = False,
    include_fold_preds: bool = False,
    include_fold_estimators: bool = False,
    include_fold_feature_importances: bool = False
) -> Dict[str, Any]:
    """
    Slice out requested fields from a results dict.

    - If `keys` is provided, interpret each as a dot-path into the dict.
    - Otherwise, returns all top-level keys except fold-level details,
      which are omitted unless explicitly included.
    """
    def get_by_path(d: Dict[str, Any], path: str) -> Any:
        parts = path.split('.')
        if len(parts) == 1:
            return d[parts[0]]
        if len(parts) == 2:
            return d[parts[0]][parts[1]]
        # for nested keys that themselves contain dots, e.g. feature_importances.feature-name.spaces-slice.mean
        top_key = parts[0]
        last_key = parts[-1]
        mid_key = '.'.join(parts[1:-1])
        return d[top_key][mid_key][last_key]

    # Direct extraction by key-paths
    if keys:
        return {path: get_by_path(results, path) for path in keys}

    # Full extraction with optional trimming
    extracted: Dict[str, Any] = {}
    for key, val in results.items():
        if key == "model_name":
            continue
        if key == "folds_estimators" and not include_fold_estimators:
            continue
        if key == "metric_scores" and isinstance(val, dict):
            trimmed = {
                metric: {
                    sub_k: sub_v
                    for sub_k, sub_v in stats.items()
                    if include_fold_scores or sub_k != "fold_scores"
                }
                for metric, stats in val.items()
            }
            extracted[key] = trimmed
        elif key == "predictions" and isinstance(val, dict):
            trimmed = {
                pred_k: pred_v
                for pred_k, pred_v in val.items()
                if include_fold_preds or pred_k != "fold_preds"
            }
            extracted[key] = trimmed
        elif key == "feature_importances" and isinstance(val, dict):
            trimmed = {
                feat_k: feat_v
                for feat_k, feat_v in val.items()
                if include_fold_feature_importances or feat_k != "fold_importances"
            }
            extracted[key] = trimmed
        else:
            extracted[key] = val  # type: ignore
    return extracted


def aggregate_model_results(
    slices: List[str],
    results_dir: str,
    global_experiment_id: str,
    analysis_type: str,
    task_type: str,
    random_state: int,
    model_name: str,
    paths_perf: List[str],
    paths_feat_imp_templates: List[str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Aggregate performance metrics and (optionally) feature importances across slices.

    Returns two dicts mapping slice name -> extracted fields.
    """
    agg_perf: Dict[str, Any] = {}
    agg_feat_imp: Dict[str, Any] = {}

    for slice_id in slices:
        results = load_single_pickle(
            results_dir, global_experiment_id, analysis_type,
            task_type, random_state, model_name, slice_id
        )
        # Performance extraction
        perf = extract_results(results, keys=paths_perf)
        agg_perf[slice_id] = perf

        # Feature importances for non-SVC models
        if model_name != "SVC" and paths_feat_imp_templates:
            feat_paths = [
                tmpl.format(slice=slice_id) for tmpl in paths_feat_imp_templates
            ]
            feat_imp = extract_results(results, keys=feat_paths)
            agg_feat_imp[slice_id] = feat_imp

    return agg_perf, agg_feat_imp


def save_object(obj: Any, file_path: str) -> None:
    """Pickle `obj` to `file_path`, creating parent dirs if needed."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def main():
    # Example usage
    RESULTS_DIR = "/home/hamza97/scratch/neuro-cocodelics/results"
    AGG_DIR = "/home/hamza97/scratch/neuro-cocodelics/aggregated"
    SLICES = [
        'MRC41', 'MLF61', 'MLC15', 'MRP32', 'MRT41', 'MLF41', 'MLT55', 'MRF25', 'MRC16', 'MRO14', 'MRF33', 'MRT53', 'MRC63', 'MLF32', 'MRF23', 'MRT56', 'MRO32', 'MZC04', 'MRC53', 'MRT14', 'MRP33', 'MLF67', 'MLC31', 'MLO24', 'MLT56', 'MLP54', 'MRT51', 'MRC31', 'MLO34', 'MLP35', 'MZF01', 'MZF03', 'MLC14', 'MRF66', 'MRC23', 'MLO44', 'MRT42', 'MRT46', 'MRF14', 'MRO44', 'MZC02', 'MLT25', 'MRP21', 'MRP52', 'MRP35', 'MRT13', 'MRP12', 'MRO42', 'MRF13', 'MLC12', 'MLT33', 'MRC55', 'MRF52', 'MRF54', 'MRT23', 'MLC16', 'MRC42', 'MLO33', 'MRP11', 'MLT57', 'MLT14', 'MRP34', 'MRC14', 'MRC15', 'MLF12', 'MRF46', 'MLF35', 'MLF63', 'MZC01', 'MLC25', 'MLF11', 'MLT16', 'MRO53', 'MRT27', 'MLP51', 'MRC51', 'MLT47', 'MLC17', 'MLP53', 'MLP12', 'MLT27', 'MRT24', 'MRF62', 'MRT45', 'MRF55', 'MLP52', 'MLO52', 'MRT12', 'MRT55', 'MLC62', 'MLP55', 'MLO53', 'MRF42', 'MLF56', 'MLF25', 'MLP11', 'MRT26', 'MLT54', 'MLT23', 'MLC32', 'MLP32', 'MRO43', 'MRT35', 'MRP22', 'MLP41', 'MLP22', 'MRT44', 'MLF65', 'MRF12', 'MLF23', 'MLT41', 'MRF64', 'MRO12', 'MRT43', 'MRC12', 'MRF61', 'MZP01', 'MRF45', 'MLT15', 'MRF44', 'MLP57', 'MRT47', 'MRF24', 'MRP57', 'MZO03', 'MLF14', 'MRP51', 'MRT22', 'MLT12', 'MRT52', 'MRT34', 'MLF43', 'MLF33', 'MRF56', 'MLT52', 'MLO13', 'MRC61', 'MLF64', 'MRC54', 'MLF53', 'MLC11', 'MLO14', 'MLF54', 'MLF55', 'MRP45', 'MLT21', 'MLF24', 'MLT35', 'MLT42', 'MLC63', 'MLC61', 'MLF13', 'MRF21', 'MLF44', 'MRC62', 'MRP54', 'MRF22', 'MRC32', 'MRO31', 'MRP42', 'MRP56', 'MLO43', 'MLC23', 'MLC54', 'MRC52', 'MRP43', 'MZO01', 'MLO41', 'MRF65', 'MLF66', 'MLF22', 'MRC21', 'MLC52', 'MRT31', 'MRF41', 'MLO22', 'MLP33', 'MLC24', 'MLF46', 'MLO31', 'MLP44', 'MLP45', 'MLP56', 'MRT33', 'MRT32', 'MLT24', 'MLT31', 'MRC11', 'MRT57', 'MZO02', 'MLF42', 'MRC17', 'MLT46', 'MLO42', 'MLF21', 'MLP23', 'MRF34', 'MZC03', 'MLC22', 'MLO51', 'MRO33', 'MLC53', 'MRF32', 'MLO21', 'MRF35', 'MRP55', 'MLC21', 'MRF31', 'MRP41', 'MRF43', 'MLO12', 'MRF51', 'MLF62', 'MRF53', 'MLP42', 'MRO13', 'MLF34', 'MZF02', 'MLT26', 'MLC51', 'MRC13', 'MLF51', 'MRO34', 'MRC22', 'MLT53', 'MRO51', 'MRC24', 'MRP53', 'MRO22', 'MLC13', 'MLO23', 'MLP21', 'MLP43', 'MRC25', 'MLP34', 'MLT51', 'MLT34', 'MLF52', 'MRP23', 'MLT11', 'MLF31', 'MRF67', 'MRT25', 'MRT37', 'MLP31', 'MRP44', 'MLT43', 'MRT11', 'MLC41', 'MRO52', 'MLO32', 'MRO11', 'MRO24', 'MLC55', 'MRO23', 'MLF45', 'MRT21', 'MLT45', 'MRT15', 'MRT16', 'MRP31', 'MRF63', 'MLC42', 'MRO21', 'MRT54', 'MLT22', 'MRF11', 'MLT13', 'MLT44'
    ]

    paths_perf = [
        "metric_scores.accuracy.mean",
        "metric_scores.accuracy.std",
        "metric_scores.roc_auc.mean",
        "metric_scores.roc_auc.std",
        "metric_scores.f1.mean",
        "metric_scores.f1.std",
    ]
    feat_imp_templates = [
        'feature_importances.feature-detrendedFluctuationMeanEpochs.spaces-{slice}.mean',
        'feature_importances.feature-detrendedFluctuationMeanEpochs.spaces-{slice}.std',
        'feature_importances.feature-higuchiFdMeanEpochs.spaces-{slice}.mean',
        'feature_importances.feature-higuchiFdMeanEpochs.spaces-{slice}.std',
        'feature_importances.feature-higuchiFdVarEpochs.spaces-{slice}.mean',
        'feature_importances.feature-higuchiFdVarEpochs.spaces-{slice}.std',
        'feature_importances.feature-hjorthComplexityMeanEpochs.spaces-{slice}.mean',
        'feature_importances.feature-hjorthComplexityMeanEpochs.spaces-{slice}.std',
        'feature_importances.feature-hjorthMobilityMeanEpochs.spaces-{slice}.mean',
        'feature_importances.feature-hjorthMobilityMeanEpochs.spaces-{slice}.std',
        'feature_importances.feature-katzFdMeanEpochs.spaces-{slice}.mean',
        'feature_importances.feature-katzFdMeanEpochs.spaces-{slice}.std',
        'feature_importances.feature-katzFdSDEpochs.spaces-{slice}.mean',
        'feature_importances.feature-katzFdSDEpochs.spaces-{slice}.std',
        'feature_importances.feature-lzivComplexityMeanEpochs.spaces-{slice}.mean',
        'feature_importances.feature-lzivComplexityMeanEpochs.spaces-{slice}.std',
        'feature_importances.feature-numZerocrossMeanEpochs.spaces-{slice}.mean',
        'feature_importances.feature-numZerocrossMeanEpochs.spaces-{slice}.std',
        'feature_importances.feature-petrosianFdMeanEpochs.spaces-{slice}.mean',
        'feature_importances.feature-petrosianFdMeanEpochs.spaces-{slice}.std',
        'feature_importances.feature-spectralEntropyMeanEpochs.spaces-{slice}.mean',
        'feature_importances.feature-spectralEntropyMeanEpochs.spaces-{slice}.std',
        'feature_importances.feature-svdEntropyMeanEpochs.spaces-{slice}.mean',
        'feature_importances.feature-svdEntropyMeanEpochs.spaces-{slice}.std',
    ]

    for model in [ "Random Forest", "Gradient Boosting","SVC"]:
        for dataset in ["psilocybin", "lsd-Video", "lsd-Music", "lsd-Open2", "lsd-Open1",
                        "lsd-Closed1", "lsd-Closed2", "lsd-avg",
                        "tiagabine", "perampanel"]: #ketamine
            exp_id = f"neuro_cocodelics_single_sensor_all_features_{dataset}"
            perf, feat_imp = aggregate_model_results(
                SLICES, RESULTS_DIR, exp_id, "baseline", "binary", 42,
                model, paths_perf, feat_imp_templates
            )
            save_object(perf, f"{AGG_DIR}/{exp_id}_baseline_{model}_perf_metrics.pkl")
            if model != "SVC":
                save_object(feat_imp, f"{AGG_DIR}/{exp_id}_baseline_{model}_feat_importance.pkl")
            print(f"Done: {model} {dataset}")


if __name__ == "__main__":
    main()

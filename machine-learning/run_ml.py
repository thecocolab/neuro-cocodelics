#!/usr/bin/env python3
import argparse
import logging
import os
from copy import deepcopy

import yaml
import pandas as pd

from coco_pipe.io import load, select_features
from coco_pipe.ml.pipeline import MLPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_analysis(X, y, groups, analysis_cfg):
    """
    Run one or more MLPipeline runs according to analysis_cfg.

    Supports three multivariate variants (all / per-sensor / per-feature) plus
    univariate mode inside each slice handled by MLPipeline.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix, shape (n_samples, n_sensors * n_features) if DataFrame,
        or generic 2D array.
    y : pd.Series or np.ndarray
        Target vector or matrix.
    groups : pd.Series or np.ndarray
        Group labels for cross-validation (optional).
    analysis_cfg : dict
        Must include keys:
          - task, analysis_type, models, metrics, cv_kwargs, n_features, direction,
            search_type, n_iter, scoring, n_jobs, save_intermediate,
            results_dir, results_file, mode
        And for slicing:
          - spatial_units : list of sensor names or "all"
          - feature_names : list of feature names or "all"
          - analysis_unit: one of "all", "sensor", "feature"
          - sep           : string separator in column names
          - reverse       : bool, if True swap sensor/feature in names
    """
    # 1) Prepare X as DataFrame for easy column-based slicing
    X_df = X if hasattr(X, "columns") else pd.DataFrame(X)
    y_arr = y.values if hasattr(y, "values") else y
    groups_arr = groups.values if hasattr(groups, "values") else groups

    # 2) Extract slicing parameters
    spatial_units = analysis_cfg.get("spatial_units", "all")
    feature_names = analysis_cfg.get("feature_names", "all")
    sep = analysis_cfg.get("sep", "_")
    reverse = analysis_cfg.get("reverse", False)
    unit = analysis_cfg.get("analysis_unit", "all")  # "all", "sensor", or "feature"

    # 3) Build base config for MLPipeline (exclude X, y, groups)
    base_cfg = {
        "task":            analysis_cfg.get("task"),
        "analysis_type":   analysis_cfg.get("analysis_type"),
        "models":          analysis_cfg.get("models"),
        "metrics":         analysis_cfg.get("metrics"),
        "cv_strategy":     analysis_cfg.get("cv_kwargs", {}).get("cv_strategy"),
        "n_splits":        analysis_cfg.get("cv_kwargs", {}).get("n_splits"),
        "cv_kwargs":       analysis_cfg.get("cv_kwargs"),
        "n_features":      analysis_cfg.get("n_features"),
        "direction":       analysis_cfg.get("direction"),
        "search_type":     analysis_cfg.get("search_type"),
        "n_iter":          analysis_cfg.get("n_iter"),
        "scoring":         analysis_cfg.get("scoring"),
        "n_jobs":          analysis_cfg.get("n_jobs"),
        "save_intermediate": analysis_cfg.get("save_intermediate"),
        "results_dir":     analysis_cfg.get("results_dir"),
        "results_file":    analysis_cfg.get("results_file"),
        "mode":            analysis_cfg.get("mode", "multivariate"),
    }
    # Drop any None values so defaults inside MLPipeline apply
    base_cfg = {k: v for k, v in base_cfg.items() if v is not None}

    logger.info(
        "Launching '%s' (%s, unit=%s) on data %s",
        base_cfg["task"],
        base_cfg["mode"],
        unit,
        X_df.shape,
    )

    # Helper to run one slice of X through MLPipeline
    def _run_slice(X_sub, label):
        logger.info("  • pipeline slice=%r, shape=%s", label, X_sub.shape)
        X_arr = X_sub.values if hasattr(X_sub, "values") else X_sub
        pipeline = MLPipeline(
            X=X_arr,
            y=y_arr,
            groups=groups_arr,
            config=base_cfg
        )
        return pipeline.run()

    # 4) Build map of label → DataFrame slice
    slice_map = {}

    if unit == "all":
        slice_map["all"] = X_df

    elif unit == "sensor":
        # one run per sensor name in spatial_units
        for sensor in spatial_units:
            if not reverse:
                cols = [c for c in X_df.columns if c.startswith(f"{sensor}{sep}")]
            else:
                cols = [c for c in X_df.columns if c.endswith(f"{sep}{sensor}")]
            if not cols:
                logger.warning("No columns found for sensor=%r", sensor)
            else:
                slice_map[sensor] = X_df[cols]

    elif unit == "feature":
        # one run per feature name in feature_names
        for feat in feature_names:
            if not reverse:
                cols = [c for c in X_df.columns if c.endswith(f"{sep}{feat}")]
            else:
                cols = [c for c in X_df.columns if c.startswith(f"{feat}{sep}")]
            if not cols:
                logger.warning("No columns found for feature=%r", feat)
            else:
                slice_map[feat] = X_df[cols]

    else:
        raise ValueError(
            "Invalid analysis_unit %r; must be one of 'all', 'sensor', 'feature'",
            unit
        )

    if not slice_map:
        raise ValueError(f"No valid slices generated for unit={unit!r}")

    # 5) Run each slice and collect results
    results = {
        label: _run_slice(X_sub, label)
        for label, X_sub in slice_map.items()
    }

    # 6) If only the "all" slice was requested, unwrap the dict
    if list(slice_map.keys()) == ["all"]:
        return results["all"]
    return results


def main():
    parser = argparse.ArgumentParser(description="Run ML analyses as defined in a YAML config")
    parser.add_argument(
        "--config", "-c", required=True, help="Path to YAML file with defaults + analyses"
    )
    args = parser.parse_args()

    # 0) Load global config and data
    cfg = yaml.safe_load(open(args.config, "r"))
    df = load("tabular", cfg["data_path"])
    all_results = {}
    defaults = cfg.get("defaults", {})

    # 1) Loop over each analysis block
    for analysis in cfg["analyses"]:
        analysis_cfg = deepcopy(defaults)
        analysis_cfg.update(analysis)

        # 2) Feature selection & target extraction
        X, y, groups = select_features(
            df,
            target_columns=analysis_cfg["target_columns"],
            covariates=analysis_cfg.get("covariates"),
            spatial_units=analysis_cfg.get("spatial_units"),
            feature_names=analysis_cfg.get("feature_names", "all"),
            row_filter=analysis_cfg.get("row_filter"),
            sep=analysis_cfg.get("sep", "_"),
            reverse=analysis_cfg.get("reverse", False),
            groups_column=analysis_cfg.get("groups_column", None),
            verbose=True,
        )

        logger.info(
            "Analysis %r selected %d features × %d samples, target=%r",
            analysis["id"], X.shape[1], X.shape[0], getattr(y, "name", None)
        )
        logger.info("  First features: %s", X.columns.tolist()[:5])

        # ensure global defaults for results_dir/file
        analysis_cfg["results_dir"]  = cfg.get("results_dir", analysis_cfg.get("results_dir"))
        analysis_cfg["results_file"] = cfg.get("results_file", analysis_cfg.get("results_file"))

        # 3) Run the configured analysis
        results = run_analysis(X, y, groups, analysis_cfg)
        all_results[analysis["id"]] = results

    # 4) Save aggregated results
    out_dir = cfg.get("results_dir", ".")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cfg['global_experiment_id']}.pkl")
    logger.info("Saving all results to %r", out_path)
    pd.to_pickle(all_results, out_path)


if __name__ == "__main__":
    main()

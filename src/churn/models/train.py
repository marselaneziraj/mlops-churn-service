"""Stage 4: train, validate, calibrate, threshold, log.

Flow
----
1. Load the raw CSV and select features through the explicit allow-list.
2. Split. ``stratified`` for a dataset without an event time; ``time`` (all rows
   before a cut-off train, after it test) whenever a timestamp exists -- a random
   split on temporal data reports a number you will never see in production.
3. Cross-validate all three candidates on the training set only (5-fold stratified),
   reporting ROC-AUC, PR-AUC, Brier and log loss with fold standard deviations.
4. Fit the winner, then wrap it in ``CalibratedClassifierCV`` so the probabilities
   mean what they say.
5. Choose the decision threshold by sweeping out-of-fold probabilities from the
   *training* set against the cost matrix -- never on the test set, which would make
   the reported cost optimistic.
6. Evaluate on the untouched test set at that fixed threshold.
7. Log params, metrics, figures and the model to MLflow; save a serving bundle.

Outputs: models/champion.joblib, reports/metrics.json, data/reference/reference.parquet,
docs/figures/{reliability_curve,cost_vs_threshold,shap_summary}.png
"""

from __future__ import annotations

import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from churn.config import Config, config_arg_parser, load_config
from churn.features.build_features import split_xy
from churn.logging_utils import configure_logging, get_logger
from churn.models.baselines import make_candidates
from churn.models.evaluate import (
    decision_metrics,
    expected_calibration_error,
    plot_cost_curve,
    plot_reliability_with_hist,
    plot_shap_summary,
    probability_metrics,
)
from churn.models.threshold import CostMatrix, optimal_threshold

log = get_logger(__name__)

CV_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "brier": "neg_brier_score",
    "log_loss": "neg_log_loss",
}


def make_split(
    frame: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Stratified or time-based holdout, as configured."""
    split_cfg = cfg.section("split")
    strategy = str(split_cfg.get("strategy", "stratified")).lower()
    test_size = float(split_cfg["test_size"])

    if strategy == "time":
        time_col = split_cfg.get("time_column")
        if not time_col or time_col not in frame.columns:
            raise KeyError(
                "split.strategy='time' needs split.time_column present in the data; "
                f"got {time_col!r}"
            )
        ordered = frame.sort_values(time_col).reset_index(drop=True)
        cut = int(len(ordered) * (1 - test_size))
        train, test = ordered.iloc[:cut], ordered.iloc[cut:]
        meta = {
            "strategy": "time",
            "time_column": time_col,
            "cutoff": str(ordered.loc[cut, time_col]),
        }
    else:
        train, test = train_test_split(
            frame,
            test_size=test_size,
            random_state=cfg.seed,
            stratify=frame[cfg.target],
        )
        meta = {
            "strategy": "stratified",
            "note": (
                "This dataset carries no event timestamp, so a stratified random holdout is "
                "used. It assumes the churn process is stationary; with a timestamp, switch "
                "split.strategy to 'time'."
            ),
        }
    meta.update(
        {
            "n_train": len(train),
            "n_test": len(test),
            "train_positive_rate": float((train[cfg.target] == cfg.positive_label).mean()),
            "test_positive_rate": float((test[cfg.target] == cfg.positive_label).mean()),
        }
    )
    return train.reset_index(drop=True), test.reset_index(drop=True), meta


def cross_validate_candidates(
    candidates: dict[str, Pipeline], x: pd.DataFrame, y: np.ndarray, cfg: Config
) -> dict[str, dict[str, float]]:
    """5-fold stratified CV for every candidate, mean +/- std per metric."""
    folds = StratifiedKFold(
        n_splits=int(cfg.section("split")["cv_folds"]), shuffle=True, random_state=cfg.seed
    )
    results: dict[str, dict[str, float]] = {}
    for name, pipeline in candidates.items():
        started = time.perf_counter()
        scores = cross_validate(
            clone(pipeline), x, y, cv=folds, scoring=CV_SCORING, n_jobs=1, error_score="raise"
        )
        row: dict[str, float] = {"fit_seconds": float(time.perf_counter() - started)}
        for metric in CV_SCORING:
            values = scores[f"test_{metric}"]
            # neg_* scorers are returned negated; flip them back to natural units.
            values = -values if metric in {"brier", "log_loss"} else values
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        results[name] = row
        log.info(
            "cv complete",
            extra={
                "model": name,
                "roc_auc": round(row["roc_auc_mean"], 4),
                "pr_auc": round(row["pr_auc_mean"], 4),
                "brier": round(row["brier_mean"], 4),
            },
        )
    return results


def fit_calibrated(
    pipeline: Pipeline, x: pd.DataFrame, y: np.ndarray, cfg: Config
) -> CalibratedClassifierCV:
    """Wrap the pipeline in cross-fitted calibration and fit it.

    ``CalibratedClassifierCV`` learns the probability remap on held-out folds, so
    no row is used to both fit the model and fit its own calibrator. With
    ``calibration.ensemble: false`` the result is a single pipeline plus a single
    calibrator -- the serving path then runs one model instead of ``cv`` of them.
    """
    cal_cfg = cfg.section("calibration")
    calibrated = CalibratedClassifierCV(
        estimator=clone(pipeline),
        method=str(cal_cfg.get("method", "isotonic")),
        cv=int(cal_cfg.get("cv", 5)),
        ensemble=bool(cal_cfg.get("ensemble", False)),
    )
    calibrated.fit(x, y)
    return calibrated


def _cv_probabilities(estimator: Any, x: pd.DataFrame, y: np.ndarray, cfg: Config) -> np.ndarray:
    """Out-of-fold positive-class probabilities on the training set."""
    folds = StratifiedKFold(
        n_splits=int(cfg.section("split")["cv_folds"]), shuffle=True, random_state=cfg.seed
    )
    probs = cross_val_predict(clone(estimator), x, y, cv=folds, method="predict_proba", n_jobs=1)
    return np.asarray(probs)[:, 1]


def train(cfg: Config) -> dict[str, Any]:
    rng_seed = cfg.seed
    np.random.seed(rng_seed)
    figures_dir = cfg.path_for("figures_dir")
    reports_dir = cfg.path_for("reports_dir")
    models_dir = cfg.path_for("models_dir")
    for d in (figures_dir, reports_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(cfg.section("data")["raw_path"])
    train_df, test_df, split_meta = make_split(frame, cfg)
    x_train, y_train = split_xy(train_df, cfg)
    x_test, y_test = split_xy(test_df, cfg)
    log.info("split", extra=split_meta)

    candidates = make_candidates(cfg)
    cv_results = cross_validate_candidates(candidates, x_train, y_train, cfg)

    best_name = max(cv_results, key=lambda k: cv_results[k]["roc_auc_mean"])
    log.info("selected candidate", extra={"model": best_name})

    uncalibrated = clone(candidates[best_name])
    uncalibrated.fit(x_train, y_train)
    calibrated = fit_calibrated(candidates[best_name], x_train, y_train, cfg)

    # --- threshold from out-of-fold training predictions only -------------------
    costs = CostMatrix.from_config(cfg.section("costs"))
    oof_prob = _cv_probabilities(calibrated, x_train, y_train, cfg)
    threshold_result = optimal_threshold(y_train, oof_prob, costs)
    log.info(
        "threshold selected",
        extra={
            "threshold": round(threshold_result.threshold, 4),
            "break_even": round(threshold_result.break_even_probability, 4),
            "oof_cost_per_customer": round(threshold_result.cost_per_customer, 2),
        },
    )

    # --- held-out evaluation ----------------------------------------------------
    test_prob_cal = calibrated.predict_proba(x_test)[:, 1]
    test_prob_raw = uncalibrated.predict_proba(x_test)[:, 1]
    n_bins = int(cfg.section("calibration").get("n_bins", 10))

    metrics: dict[str, Any] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "seed": rng_seed,
        "python": platform.python_version(),
        "split": split_meta,
        "cv": cv_results,
        "selected_model": best_name,
        "test": {
            "uncalibrated": probability_metrics(y_test, test_prob_raw),
            "calibrated": probability_metrics(y_test, test_prob_cal),
        },
        "calibration": {
            "method": cfg.section("calibration")["method"],
            "ece_uncalibrated": expected_calibration_error(y_test, test_prob_raw, n_bins),
            "ece_calibrated": expected_calibration_error(y_test, test_prob_cal, n_bins),
        },
        "threshold": {
            "value": threshold_result.threshold,
            "selected_on": "out-of-fold training predictions",
            "break_even_probability": threshold_result.break_even_probability,
            "cost_matrix": {
                "false_negative": costs.false_negative,
                "false_positive": costs.false_positive,
                "true_positive": costs.true_positive,
                "true_negative": costs.true_negative,
            },
            "oof_confusion": threshold_result.confusion,
            "oof_cost_per_customer": threshold_result.cost_per_customer,
        },
        "decision": {
            "at_selected_threshold": decision_metrics(
                y_test, test_prob_cal, threshold_result.threshold, costs
            ),
            "at_half": decision_metrics(y_test, test_prob_cal, 0.5, costs),
            "treat_none": decision_metrics(y_test, test_prob_cal, 1.01, costs),
            "treat_all": decision_metrics(y_test, test_prob_cal, 0.0, costs),
        },
    }
    chosen = metrics["decision"]["at_selected_threshold"]["cost_per_customer"]
    metrics["decision"]["savings_per_customer_vs_half"] = (
        metrics["decision"]["at_half"]["cost_per_customer"] - chosen
    )
    metrics["decision"]["savings_per_customer_vs_treat_none"] = (
        metrics["decision"]["treat_none"]["cost_per_customer"] - chosen
    )

    # --- figures ----------------------------------------------------------------
    reliability_path = plot_reliability_with_hist(
        y_test,
        {f"{best_name} (raw)": test_prob_raw, f"{best_name} (calibrated)": test_prob_cal},
        figures_dir / "reliability_curve.png",
        n_bins=n_bins,
    )
    cost_path = plot_cost_curve(
        threshold_result, figures_dir / "cost_vs_threshold.png", len(y_train)
    )
    shap_path: Path | None = None
    if cfg.section("tracking").get("log_shap", True) and best_name == "lightgbm":
        sample_n = min(int(cfg.section("tracking").get("shap_sample", 2000)), len(x_train))
        shap_path = plot_shap_summary(
            uncalibrated,
            x_train.sample(sample_n, random_state=rng_seed),
            figures_dir / "shap_summary.png",
        )
        if shap_path is None:
            log.warning("shap not installed; skipping the SHAP summary plot")
    figures = [p for p in (reliability_path, cost_path, shap_path) if p is not None]

    # --- serving bundle ----------------------------------------------------------
    bundle = {
        "pipeline": calibrated,
        "threshold": float(threshold_result.threshold),
        "metadata": {
            "model_name": best_name,
            "model_version": f"{best_name}-{datetime.now(tz=UTC):%Y%m%d%H%M%S}",
            "trained_at": metrics["generated_at"],
            "seed": rng_seed,
            "calibration": cfg.section("calibration")["method"],
            "feature_columns": cfg.feature_columns,
            "numeric_features": cfg.numeric_features,
            "categorical_features": cfg.categorical_features,
            "target": cfg.target,
            "positive_label": cfg.positive_label,
            "test_roc_auc": metrics["test"]["calibrated"]["roc_auc"],
            "test_brier": metrics["test"]["calibrated"]["brier"],
            "cost_matrix": metrics["threshold"]["cost_matrix"],
        },
    }
    model_path = Path(cfg.section("serving")["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    # Reference window for drift monitoring: the training features plus the
    # predictions the model made on them. PSI compares live traffic against this.
    reference = x_train.copy()
    reference["prediction_proba"] = calibrated.predict_proba(x_train)[:, 1]
    reference["target"] = y_train
    reference_path = Path(cfg.section("data")["reference_path"])
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(reference_path, index=False)

    metrics_path = reports_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    log_to_mlflow(cfg, metrics, figures, model_path, calibrated, x_train.head(5))

    log.info(
        "training complete",
        extra={
            "model": best_name,
            "test_roc_auc": round(metrics["test"]["calibrated"]["roc_auc"], 4),
            "threshold": round(threshold_result.threshold, 4),
            "artefact": str(model_path),
        },
    )
    return metrics


def log_to_mlflow(
    cfg: Config,
    metrics: dict[str, Any],
    figures: list[Path],
    model_path: Path,
    estimator: Any,
    input_example: pd.DataFrame,
) -> None:
    """Log the run to MLflow if it is installed and enabled.

    Why a tracking server at all: a notebook gives you one number at a time with no
    link between the number, the parameters and the artefact that produced it. MLflow
    makes every run addressable -- params, metrics, figures and the serialised model
    under one run id -- so "which config produced the champion?" has an answer.
    """
    tracking = cfg.section("tracking")
    if not tracking.get("enabled", True):
        return
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:  # pragma: no cover - optional extra
        log.warning("mlflow not installed; skipping tracking (pip install '.[tracking]')")
        return

    uri = tracking.get("tracking_uri")
    if uri:
        mlflow.set_tracking_uri(str(uri))
    mlflow.set_experiment(str(tracking.get("experiment", "churn")))

    with mlflow.start_run(run_name=f"{metrics['selected_model']}-{metrics['generated_at'][:19]}"):
        mlflow.log_params(
            {
                "seed": cfg.seed,
                "selected_model": metrics["selected_model"],
                "split_strategy": metrics["split"]["strategy"],
                "test_size": cfg.section("split")["test_size"],
                "cv_folds": cfg.section("split")["cv_folds"],
                "calibration_method": cfg.section("calibration")["method"],
                "calibration_ensemble": cfg.section("calibration").get("ensemble", False),
                "cost_fn": cfg.section("costs")["false_negative"],
                "cost_fp": cfg.section("costs")["false_positive"],
                **{f"lgbm_{k}": v for k, v in cfg.section("model")["params"].items()},
            }
        )
        flat: dict[str, float] = {}
        for model_name, row in metrics["cv"].items():
            for key, value in row.items():
                flat[f"cv_{model_name}_{key}"] = float(value)
        for variant, row in metrics["test"].items():
            for key, value in row.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    flat[f"test_{variant}_{key}"] = float(value)
        for key, value in metrics["decision"]["at_selected_threshold"].items():
            flat[f"decision_{key}"] = float(value)
        flat["ece_calibrated"] = float(metrics["calibration"]["ece_calibrated"])
        flat["ece_uncalibrated"] = float(metrics["calibration"]["ece_uncalibrated"])
        mlflow.log_metrics(flat)

        mlflow.log_artifact(str(cfg.path), artifact_path="config")
        for fig in figures:
            mlflow.log_artifact(str(fig), artifact_path="figures")
        mlflow.log_artifact(
            str(cfg.path_for("reports_dir") / "metrics.json"), artifact_path="reports"
        )
        mlflow.log_artifact(str(model_path), artifact_path="bundle")
        mlflow.sklearn.log_model(
            sk_model=estimator,
            artifact_path="model",
            input_example=input_example,
            registered_model_name=str(tracking.get("registered_model", "churn-classifier")),
        )
        log.info("logged run to mlflow", extra={"experiment": tracking.get("experiment")})


def main() -> None:
    configure_logging()
    args = config_arg_parser("Train, calibrate and threshold the churn model").parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()

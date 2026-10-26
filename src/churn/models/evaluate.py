"""Metrics, calibration assessment and the two decision figures.

Calibration, briefly
--------------------
Ranking quality (AUC) and probability quality are different things. A model can
rank perfectly while reporting 0.9 for cases that churn 40% of the time. The
Brier score (mean squared error of the probability) and the reliability curve
(predicted probability vs observed frequency, bucketed) measure the second.
We care because the decision threshold is derived from a cost matrix in
probability units -- see :mod:`churn.models.threshold`.

Isotonic regression is the default remap: non-parametric, monotone, and with
~5600 training rows there is enough data for it. Below roughly 1000 rows prefer
``sigmoid`` (Platt), which fits two parameters instead of a step function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn.models.threshold import CostMatrix, ThresholdResult, total_cost


def probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Threshold-free metrics: ranking (AUC, PR-AUC) and probability quality."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-9, 1 - 1e-9)
    single_class = len(np.unique(y_true)) < 2
    return {
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float("nan") if single_class else float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "mean_predicted": float(y_prob.mean()),
        "base_rate": float(y_true.mean()),
    }


def decision_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, costs: CostMatrix
) -> dict[str, float]:
    """Metrics at an operating point, including the business cost."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    n = max(int(y_true.size), 1)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "alert_rate": float(y_pred.mean()),
        "total_cost": float(total_cost(y_true, y_pred, costs)),
        "cost_per_customer": float(total_cost(y_true, y_pred, costs) / n),
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Weighted mean |predicted - observed| across equal-width probability bins."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)
    error = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        error += mask.mean() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(error)


def plot_reliability_with_hist(
    y_true: np.ndarray,
    probs: dict[str, np.ndarray],
    out_path: Path,
    n_bins: int = 10,
) -> Path:
    """Reliability curve plus the predicted-probability histogram beneath it.

    The histogram matters: a curve can look good in a bin that holds 12 customers.
    """
    y_true = np.asarray(y_true).astype(int)
    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(5.6, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot([0, 1], [0, 1], ls="--", color="#666666", lw=1, label="perfectly calibrated")
    colors = ["#C44E52", "#4C72B0", "#55A868", "#8172B2"]
    for (name, prob), color in zip(probs.items(), colors, strict=False):
        prob_true, prob_pred = calibration_curve(y_true, prob, n_bins=n_bins, strategy="quantile")
        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            ms=4,
            lw=1.6,
            color=color,
            label=f"{name} (Brier {brier_score_loss(y_true, prob):.4f}, "
            f"ECE {expected_calibration_error(y_true, prob, n_bins):.4f})",
        )
        ax_hist.hist(prob, bins=40, alpha=0.55, color=color, label=name)
    ax.set_ylabel("observed churn frequency")
    ax.set_title(f"Reliability curve ({n_bins} quantile bins)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax_hist.set_xlabel("predicted probability")
    ax_hist.set_ylabel("count")
    ax_hist.set_yscale("log")
    ax_hist.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_cost_curve(result: ThresholdResult, out_path: Path, n_customers: int) -> Path:
    """Total cost as a function of the decision threshold, with the chosen point marked."""
    thresholds = np.asarray(result.sweep["threshold"], dtype=float)
    cost = np.asarray(result.sweep["cost"], dtype=float) / max(n_customers, 1)
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(thresholds, cost, color="#4C72B0", lw=1.8)
    ax.axvline(
        result.threshold,
        color="#C44E52",
        lw=1.6,
        label=f"chosen t* = {result.threshold:.3f} ({result.cost_per_customer:.2f}/customer)",
    )
    ax.axvline(
        result.break_even_probability,
        color="#55A868",
        ls="--",
        lw=1.4,
        label=f"cost-matrix break-even = {result.break_even_probability:.3f}",
    )
    ax.axvline(0.5, color="#888888", ls=":", lw=1.4, label="naive 0.5")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("cost per customer")
    ax.set_title("Cost vs decision threshold (out-of-fold predictions)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_shap_summary(
    pipeline: Any, x_sample: Any, out_path: Path, max_display: int = 20
) -> Path | None:
    """SHAP beeswarm for the tree model inside a fitted Pipeline.

    Returns None if ``shap`` is not installed -- it is an optional extra, and a
    missing explainability library should not fail a training run.
    """
    try:
        import shap
    except ImportError:  # pragma: no cover - optional dependency
        return None

    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    matrix = pre.transform(x_sample)
    names = [str(n) for n in pre.get_feature_names_out()]
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(matrix)
    if isinstance(values, list):  # older shap returns one array per class
        values = values[1]
    fig = plt.figure(figsize=(7.5, 6.5))
    shap.summary_plot(
        values, matrix, feature_names=names, max_display=max_display, show=False, plot_size=None
    )
    plt.title("SHAP summary - contribution to log-odds of churn", fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path

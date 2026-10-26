"""Population Stability Index.

What PSI measures
-----------------
PSI compares the distribution of a variable today against a reference window
(here: the training features). Bin both, then sum over bins

    PSI = sum_i (actual_i - expected_i) * ln(actual_i / expected_i)

It is the symmetrised KL divergence of two discretised distributions. The number
is unitless and the industry convention, from credit scoring, is:

    PSI < 0.10   no meaningful shift
    0.10 - 0.25  moderate shift, investigate
    PSI > 0.25   significant shift, act

Why it is the first alert you build: PSI needs no labels. Churn labels arrive
weeks after the prediction, so accuracy-based alerting is always late. An input or
prediction distribution that has moved is actionable the same day.

Bin edges come from the *reference* quantiles and are then frozen, which is what
makes successive days comparable. Empty bins are floored at ``eps`` so a single
missing category does not send PSI to infinity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WARN = 0.10
ALERT = 0.25
EPS = 1e-6


@dataclass(frozen=True)
class PSIResult:
    feature: str
    psi: float
    kind: str  # numeric | categorical
    severity: str  # none | moderate | significant
    n_reference: int
    n_current: int

    @property
    def is_alert(self) -> bool:
        return self.severity == "significant"


def _severity(value: float, warn: float = WARN, alert: float = ALERT) -> str:
    if value >= alert:
        return "significant"
    if value >= warn:
        return "moderate"
    return "none"


def _psi_from_shares(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected, EPS, None)
    actual = np.clip(actual, EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI for a continuous variable using frozen reference quantile edges."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    if ref.size == 0 or cur.size == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    return _psi_from_shares(ref_counts / ref.size, cur_counts / cur.size)


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """PSI over category levels; levels unseen in the reference land in an 'other' bin."""
    ref = reference.astype(str)
    cur = current.astype(str)
    levels = sorted(set(ref.unique()) | set(cur.unique()))
    ref_shares = np.array([float((ref == lv).mean()) for lv in levels])
    cur_shares = np.array([float((cur == lv).mean()) for lv in levels])
    return _psi_from_shares(ref_shares, cur_shares)


def feature_psi(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    bins: int = 10,
    warn: float = WARN,
    alert: float = ALERT,
) -> list[PSIResult]:
    """PSI per feature, sorted worst first."""
    results: list[PSIResult] = []
    for col in numeric:
        if col in reference.columns and col in current.columns:
            value = numeric_psi(reference[col], current[col], bins=bins)
            results.append(
                PSIResult(
                    col,
                    value,
                    "numeric",
                    _severity(value, warn, alert),
                    len(reference),
                    len(current),
                )
            )
    for col in categorical:
        if col in reference.columns and col in current.columns:
            value = categorical_psi(reference[col], current[col])
            results.append(
                PSIResult(
                    col,
                    value,
                    "categorical",
                    _severity(value, warn, alert),
                    len(reference),
                    len(current),
                )
            )
    return sorted(results, key=lambda r: -r.psi)


def prediction_psi(
    reference_proba: pd.Series,
    current_proba: pd.Series,
    bins: int = 10,
    warn: float = WARN,
    alert: float = ALERT,
) -> PSIResult:
    """PSI on the model's own output -- the single most informative drift signal."""
    value = numeric_psi(reference_proba, current_proba, bins=bins)
    return PSIResult(
        "prediction_proba",
        value,
        "numeric",
        _severity(value, warn, alert),
        len(reference_proba),
        len(current_proba),
    )

"""Cost-based decision thresholding.

A classifier outputs a probability; a business needs a yes/no. 0.5 is the right
cut-off only when a false positive and a false negative cost the same, which for
churn they emphatically do not: letting a customer leave costs their remaining
margin, while a needless retention offer costs the offer. With
``FN = 500`` and ``FP = 50`` the break-even probability is

    p* = C_FP / (C_FP + C_FN) = 50 / 550 = 0.0909

so you should intervene on anyone with a churn probability above ~9%. We do not
use that closed form directly -- we sweep the threshold on held-out predictions
and pick the minimum-cost point, which also accounts for the empirical shape of
the probability distribution -- but the closed form is reported next to it as a
sanity check, and the two should agree closely once the model is calibrated.

This is also *why calibration matters*: the cost-optimal threshold is a statement
about probabilities. If the model's 0.09 does not mean "9 out of 100 such
customers churn", the threshold derived from the cost matrix is meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CostMatrix:
    """Cost in currency units incurred per prediction outcome."""

    false_negative: float
    false_positive: float
    true_positive: float = 0.0
    true_negative: float = 0.0

    @classmethod
    def from_config(cls, costs: dict[str, float]) -> CostMatrix:
        return cls(
            false_negative=float(costs["false_negative"]),
            false_positive=float(costs["false_positive"]),
            true_positive=float(costs.get("true_positive", 0.0)),
            true_negative=float(costs.get("true_negative", 0.0)),
        )

    @property
    def break_even_probability(self) -> float:
        """p* where acting and not acting have equal expected cost."""
        gain_fn = self.false_negative - self.true_positive
        gain_fp = self.false_positive - self.true_negative
        denom = gain_fn + gain_fp
        return float(gain_fp / denom) if denom > 0 else 0.5


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    expected_cost: float
    cost_per_customer: float
    break_even_probability: float
    confusion: dict[str, int]
    baseline_cost_treat_none: float
    baseline_cost_treat_all: float
    baseline_cost_at_half: float
    sweep: dict[str, list[float]] = field(repr=False, default_factory=dict)

    @property
    def savings_vs_treat_none(self) -> float:
        return self.baseline_cost_treat_none - self.expected_cost

    @property
    def savings_vs_half(self) -> float:
        return self.baseline_cost_at_half - self.expected_cost


def total_cost(y_true: np.ndarray, y_pred: np.ndarray, costs: CostMatrix) -> float:
    """Total cost of a set of hard predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return (
        tp * costs.true_positive
        + tn * costs.true_negative
        + fp * costs.false_positive
        + fn * costs.false_negative
    )


def optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    costs: CostMatrix,
    grid: int = 501,
) -> ThresholdResult:
    """Sweep thresholds on held-out probabilities and return the cheapest.

    Ties are broken toward the higher threshold, i.e. toward fewer interventions,
    because a flat cost curve should not buy extra operational load for nothing.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")
    if y_true.size == 0:
        raise ValueError("cannot select a threshold from an empty array")

    thresholds = np.linspace(0.0, 1.0, grid)
    sweep = np.array([total_cost(y_true, y_prob >= t, costs) for t in thresholds])
    best_cost = float(sweep.min())
    best_idx = int(np.max(np.flatnonzero(sweep == best_cost)))
    best_t = float(thresholds[best_idx])

    y_best = (y_prob >= best_t).astype(int)
    confusion = {
        "tp": int(np.sum((y_true == 1) & (y_best == 1))),
        "fp": int(np.sum((y_true == 0) & (y_best == 1))),
        "fn": int(np.sum((y_true == 1) & (y_best == 0))),
        "tn": int(np.sum((y_true == 0) & (y_best == 0))),
    }
    return ThresholdResult(
        threshold=best_t,
        expected_cost=best_cost,
        cost_per_customer=best_cost / float(y_true.size),
        break_even_probability=costs.break_even_probability,
        confusion=confusion,
        baseline_cost_treat_none=total_cost(y_true, np.zeros_like(y_true), costs),
        baseline_cost_treat_all=total_cost(y_true, np.ones_like(y_true), costs),
        baseline_cost_at_half=total_cost(y_true, y_prob >= 0.5, costs),
        sweep={"threshold": thresholds.tolist(), "cost": sweep.tolist()},
    )

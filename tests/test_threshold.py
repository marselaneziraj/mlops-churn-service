"""Unit tests for the cost matrix and threshold search."""

from __future__ import annotations

import numpy as np
import pytest

from churn.models.threshold import CostMatrix, optimal_threshold, total_cost


def test_break_even_probability_matches_the_closed_form():
    costs = CostMatrix(false_negative=500.0, false_positive=50.0)
    assert costs.break_even_probability == pytest.approx(50 / 550)


def test_symmetric_costs_give_a_half_threshold():
    costs = CostMatrix(false_negative=10.0, false_positive=10.0)
    assert costs.break_even_probability == pytest.approx(0.5)


def test_total_cost_counts_each_cell():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])  # tp, fn, fp, tn
    costs = CostMatrix(false_negative=500.0, false_positive=50.0)
    assert total_cost(y_true, y_pred, costs) == pytest.approx(550.0)


def test_asymmetric_costs_push_the_threshold_below_half():
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.26, size=4000)
    # Calibrated-ish probabilities: positives drawn higher than negatives.
    y_prob = np.where(y_true == 1, rng.beta(3, 5, 4000), rng.beta(2, 9, 4000))
    expensive_fn = optimal_threshold(y_true, y_prob, CostMatrix(500.0, 50.0))
    symmetric = optimal_threshold(y_true, y_prob, CostMatrix(100.0, 100.0))
    assert expensive_fn.threshold < 0.5
    assert expensive_fn.threshold < symmetric.threshold


def test_chosen_threshold_is_the_cheapest_on_the_grid():
    rng = np.random.default_rng(1)
    y_true = rng.binomial(1, 0.3, size=1500)
    y_prob = np.clip(y_true * 0.4 + rng.normal(0.25, 0.18, 1500), 0, 1)
    costs = CostMatrix(400.0, 40.0)
    result = optimal_threshold(y_true, y_prob, costs)
    sweep = np.asarray(result.sweep["cost"])
    assert result.expected_cost == pytest.approx(sweep.min())
    assert result.expected_cost <= result.baseline_cost_at_half
    assert result.expected_cost <= result.baseline_cost_treat_none
    assert result.expected_cost <= result.baseline_cost_treat_all


def test_confusion_counts_sum_to_n():
    rng = np.random.default_rng(2)
    y_true = rng.binomial(1, 0.2, size=600)
    y_prob = rng.random(600)
    result = optimal_threshold(y_true, y_prob, CostMatrix(300.0, 30.0))
    assert sum(result.confusion.values()) == 600


def test_ties_break_toward_fewer_interventions():
    """A flat cost region should not buy extra retention offers for no gain."""
    y_true = np.zeros(100, dtype=int)
    y_prob = np.full(100, 0.01)
    result = optimal_threshold(y_true, y_prob, CostMatrix(500.0, 50.0))
    assert result.threshold > 0.01
    assert result.expected_cost == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        optimal_threshold(np.array([0, 1]), np.array([0.1, 0.2, 0.3]), CostMatrix(1.0, 1.0))


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        optimal_threshold(np.array([]), np.array([]), CostMatrix(1.0, 1.0))

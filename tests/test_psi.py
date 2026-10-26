"""Tests for the PSI implementation and its severity bands."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.monitoring.psi import ALERT, categorical_psi, feature_psi, numeric_psi, prediction_psi


def test_identical_numeric_distributions_give_near_zero_psi():
    rng = np.random.default_rng(0)
    values = pd.Series(rng.normal(65, 30, 5000))
    assert numeric_psi(values, values) == pytest.approx(0.0, abs=1e-9)


def test_resampling_the_same_distribution_stays_below_the_warn_band():
    """The false-positive check: sampling noise must not look like drift."""
    rng = np.random.default_rng(1)
    reference = pd.Series(rng.normal(65, 30, 20000))
    current = pd.Series(rng.normal(65, 30, 2000))
    assert numeric_psi(reference, current) < 0.1


def test_a_large_mean_shift_crosses_the_alert_threshold():
    rng = np.random.default_rng(2)
    reference = pd.Series(rng.normal(65, 10, 5000))
    current = pd.Series(rng.normal(95, 10, 5000))
    assert numeric_psi(reference, current) > ALERT


def test_psi_grows_monotonically_with_the_shift():
    rng = np.random.default_rng(3)
    reference = pd.Series(rng.normal(0, 1, 20000))
    psis = [
        numeric_psi(reference, pd.Series(rng.normal(shift, 1, 5000)))
        for shift in (0.1, 0.4, 0.8, 1.5)
    ]
    assert psis == sorted(psis)


def test_categorical_psi_detects_a_mix_change():
    reference = pd.Series(["Month-to-month"] * 500 + ["One year"] * 300 + ["Two year"] * 200)
    current = pd.Series(["Month-to-month"] * 900 + ["One year"] * 70 + ["Two year"] * 30)
    assert categorical_psi(reference, current) > ALERT
    assert categorical_psi(reference, reference) == pytest.approx(0.0, abs=1e-9)


def test_a_new_category_level_does_not_produce_infinity():
    reference = pd.Series(["a"] * 100 + ["b"] * 100)
    current = pd.Series(["a"] * 90 + ["b"] * 90 + ["c"] * 20)
    value = categorical_psi(reference, current)
    assert np.isfinite(value)
    assert value > 0


def test_empty_input_returns_zero_rather_than_raising():
    assert numeric_psi(pd.Series([], dtype=float), pd.Series([1.0, 2.0])) == 0.0


def test_feature_psi_is_sorted_worst_first_and_labels_severity():
    rng = np.random.default_rng(4)
    reference = pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, 4000),
            "MonthlyCharges": rng.normal(65, 20, 4000),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], 4000),
        }
    )
    current = reference.copy()
    current["MonthlyCharges"] = current["MonthlyCharges"] + 40
    results = feature_psi(
        reference, current, numeric=["tenure", "MonthlyCharges"], categorical=["Contract"]
    )
    assert [r.psi for r in results] == sorted((r.psi for r in results), reverse=True)
    assert results[0].feature == "MonthlyCharges"
    assert results[0].severity == "significant"
    assert results[0].is_alert
    assert results[-1].severity == "none"


def test_prediction_psi_returns_a_named_result():
    rng = np.random.default_rng(5)
    reference = pd.Series(rng.beta(2, 8, 5000))
    current = pd.Series(rng.beta(5, 5, 5000))
    result = prediction_psi(reference, current)
    assert result.feature == "prediction_proba"
    assert result.psi > ALERT

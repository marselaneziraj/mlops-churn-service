"""Data-contract test.

A data contract is the schema the pipeline promises to accept: which columns
exist, their types, their allowed values, and how much missingness is tolerable.
Enforcing it as a test means an upstream change (a renamed column, a new category
level, a column that silently became text) breaks CI instead of producing a model
that trains fine and predicts nonsense.

These assertions are deliberately about the *schema and ranges*, not about model
quality -- they should pass identically on the real Telco CSV and on the synthetic
stand-in, and they do.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churn.data.make_dataset import CATEGORY_LEVELS

MAX_MISSING_RATE = 0.02
EXPECTED_COLUMNS = 21


def test_all_configured_features_exist(raw_frame, cfg):
    missing = [c for c in cfg.feature_columns if c not in raw_frame.columns]
    assert missing == [], f"columns promised by the config are absent: {missing}"


def test_frame_has_the_expected_width(raw_frame):
    assert raw_frame.shape[1] == EXPECTED_COLUMNS


def test_target_is_binary_with_the_configured_labels(raw_frame, cfg):
    levels = set(raw_frame[cfg.target].unique())
    assert levels == {"Yes", "No"}
    assert cfg.positive_label in levels


def test_positive_rate_is_plausible(raw_frame, cfg):
    rate = (raw_frame[cfg.target] == cfg.positive_label).mean()
    assert 0.05 < rate < 0.60, f"base rate {rate:.3f} is outside the plausible band"


def test_numeric_columns_are_numeric_after_ingestion(raw_frame, cfg):
    for col in cfg.numeric_features:
        assert pd.api.types.is_numeric_dtype(raw_frame[col]), f"{col} is not numeric"


def test_numeric_ranges(raw_frame):
    assert raw_frame["tenure"].between(0, 120).all()
    assert raw_frame["MonthlyCharges"].between(0, 1000).all()
    assert raw_frame["TotalCharges"].dropna().ge(0).all()


def test_missingness_is_within_contract(raw_frame, cfg):
    """Only TotalCharges may be missing, and only for tenure-0 customers."""
    for col in cfg.feature_columns:
        rate = raw_frame[col].isna().mean()
        assert rate <= MAX_MISSING_RATE, (
            f"{col} is {rate:.2%} missing, contract allows {MAX_MISSING_RATE:.0%}"
        )
    blanks = raw_frame["TotalCharges"].isna()
    if blanks.any():
        assert (raw_frame.loc[blanks, "tenure"] == 0).all()


@pytest.mark.parametrize("column", sorted(CATEGORY_LEVELS))
def test_categorical_levels_are_from_the_known_vocabulary(raw_frame, column):
    observed = set(raw_frame[column].dropna().unique())
    allowed = set(CATEGORY_LEVELS[column])
    assert observed <= allowed, f"{column} has unexpected levels: {sorted(observed - allowed)}"


def test_identifier_is_unique(raw_frame, cfg):
    id_col = cfg.section("data")["id_column"]
    assert raw_frame[id_col].is_unique, "duplicate customer ids leak across the split"


def test_no_forbidden_column_is_configured_as_a_feature(cfg):
    assert not (set(cfg.feature_columns) & set(cfg.forbidden_columns))

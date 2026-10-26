"""Unit tests for the feature pipeline.

The load-bearing assertion is the no-leakage one: fitting the preprocessor on a
training fold must not let test-fold statistics in. We check it by fitting on a
subset and confirming the imputation value equals that subset's median, not the
full frame's.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from churn.features.build_features import (
    build_pipeline,
    build_preprocessor,
    feature_names,
    split_xy,
)
from churn.models.baselines import make_estimator


def test_split_xy_selects_only_configured_features(raw_frame, cfg):
    x, y = split_xy(raw_frame, cfg)
    assert list(x.columns) == cfg.feature_columns
    assert cfg.target not in x.columns
    assert "customerID" not in x.columns
    assert set(np.unique(y)) <= {0, 1}
    assert len(x) == len(raw_frame) == len(y)


def test_split_xy_rejects_missing_columns(raw_frame, cfg):
    with pytest.raises(KeyError, match="columns missing"):
        split_xy(raw_frame.drop(columns=["tenure"]), cfg)


def test_split_xy_binarises_target_on_the_positive_label(raw_frame, cfg):
    _, y = split_xy(raw_frame, cfg)
    expected = (raw_frame[cfg.target] == cfg.positive_label).sum()
    assert y.sum() == expected


def test_preprocessor_imputes_from_training_rows_only(raw_frame, cfg):
    """Median imputation must come from the fitted subset, never the whole frame."""
    x, _ = split_xy(raw_frame, cfg)
    assert x["TotalCharges"].isna().any(), "fixture should contain blanks to impute"
    head = x.iloc[:300]
    pre = build_preprocessor(cfg)
    pre.fit(head)
    imputer = pre.named_transformers_["num"].named_steps["impute"]
    idx = cfg.numeric_features.index("TotalCharges")
    assert imputer.statistics_[idx] == pytest.approx(head["TotalCharges"].median())
    assert imputer.statistics_[idx] != pytest.approx(x["TotalCharges"].median())


def test_unseen_category_does_not_raise(raw_frame, cfg):
    """handle_unknown='ignore' is what stops a new category level becoming a 500."""
    x, _ = split_xy(raw_frame, cfg)
    pre = build_preprocessor(cfg).fit(x)
    novel = x.iloc[[0]].copy()
    novel["Contract"] = "Three year lunar lease"
    transformed = pre.transform(novel)
    assert transformed.shape[1] == pre.transform(x.iloc[[0]]).shape[1]
    assert np.isfinite(transformed).all()


def test_no_nans_after_transform(raw_frame, cfg):
    x, _ = split_xy(raw_frame, cfg)
    matrix = build_preprocessor(cfg).fit_transform(x)
    assert not np.isnan(matrix).any()


def test_feature_names_match_matrix_width(raw_frame, cfg):
    x, _ = split_xy(raw_frame, cfg)
    pre = build_preprocessor(cfg).fit(x)
    assert len(feature_names(pre)) == pre.transform(x).shape[1]


def test_build_pipeline_is_a_two_step_pipeline(cfg):
    pipe = build_pipeline(cfg, make_estimator("logistic", cfg))
    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps) == ["preprocess", "model"]


def test_pipeline_is_deterministic_under_the_seed(raw_frame, cfg):
    x, y = split_xy(raw_frame, cfg)
    first = build_pipeline(cfg, make_estimator("logistic", cfg)).fit(x, y).predict_proba(x)[:, 1]
    second = build_pipeline(cfg, make_estimator("logistic", cfg)).fit(x, y).predict_proba(x)[:, 1]
    np.testing.assert_allclose(first, second)


def test_column_order_does_not_change_predictions(raw_frame, cfg):
    """The ColumnTransformer selects by name, so a shuffled frame must score identically."""
    x, y = split_xy(raw_frame, cfg)
    model = build_pipeline(cfg, make_estimator("logistic", cfg)).fit(x, y)
    shuffled = x.loc[:, list(reversed(x.columns))]
    np.testing.assert_allclose(
        model.predict_proba(x)[:, 1], model.predict_proba(shuffled)[:, 1], rtol=1e-12
    )

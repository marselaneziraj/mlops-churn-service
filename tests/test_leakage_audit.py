"""Tests for the leakage audit: it must pass clean data and catch a planted leak."""

from __future__ import annotations

import pandas as pd

from churn.config import Config
from churn.data.leakage_audit import (
    check_correlation,
    check_duplicates,
    check_forbidden,
    check_solo_auc,
    cramers_v,
)


def test_clean_frame_passes_every_check(raw_frame, cfg):
    assert check_forbidden(cfg).status == "PASS"
    assert check_correlation(raw_frame, cfg).status == "PASS"
    assert check_solo_auc(raw_frame, cfg).status == "PASS"
    assert check_duplicates(raw_frame, cfg).status in {"PASS", "WARN"}


def test_a_planted_leak_is_caught(raw_frame, cfg):
    """Add a feature that is the target in disguise and confirm both checks fire."""
    leaky = raw_frame.copy()
    leaky["retention_case_opened"] = leaky[cfg.target]
    leaky_cfg = Config(
        raw={
            **cfg.raw,
            "features": {
                **cfg.section("features"),
                "categorical": [*cfg.categorical_features, "retention_case_opened"],
            },
        },
        path=cfg.path,
    )
    assert check_correlation(leaky, leaky_cfg).status == "FAIL"
    assert check_solo_auc(leaky, leaky_cfg).status == "FAIL"


def test_forbidden_column_used_as_a_feature_fails(cfg):
    bad = Config(
        raw={
            **cfg.raw,
            "features": {
                **cfg.section("features"),
                "numeric": [*cfg.numeric_features, "customerID"],
            },
        },
        path=cfg.path,
    )
    finding = check_forbidden(bad)
    assert finding.status == "FAIL"
    assert "customerID" in finding.evidence["offenders"]


def test_duplicate_ids_fail(raw_frame, cfg):
    duped = pd.concat([raw_frame, raw_frame.iloc[:5]], ignore_index=True)
    assert check_duplicates(duped, cfg).status == "FAIL"


def test_cramers_v_bounds():
    identical = pd.Series(["a", "b", "a", "b"] * 50)
    assert cramers_v(identical, identical) > 0.95
    independent_x = pd.Series(["a", "b"] * 100)
    independent_y = pd.Series(["x", "x", "y", "y"] * 50)
    assert cramers_v(independent_x, independent_y) < 0.2

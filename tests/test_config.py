"""Tests for the config loader -- the thing every other module depends on."""

from __future__ import annotations

import pytest

from churn.config import load_config


def test_config_exposes_the_expected_sections(cfg):
    for section in (
        "data",
        "split",
        "features",
        "model",
        "calibration",
        "costs",
        "tracking",
        "serving",
        "monitoring",
        "paths",
    ):
        assert isinstance(cfg.section(section), dict)


def test_feature_columns_are_numeric_plus_categorical(cfg):
    assert cfg.feature_columns == cfg.numeric_features + cfg.categorical_features
    assert len(set(cfg.feature_columns)) == len(cfg.feature_columns), "duplicate feature listed"


def test_costs_are_asymmetric_in_the_shipped_config(cfg):
    costs = cfg.section("costs")
    assert costs["false_negative"] > costs["false_positive"], (
        "the shipped business framing assumes a missed churner costs more than a wasted offer"
    )


def test_missing_section_raises(cfg):
    with pytest.raises(KeyError):
        cfg.section("not_a_real_section")


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")

"""Shared fixtures.

The whole suite runs offline on a small synthetic frame: no network, no MLflow
server, no trained artefact on disk. A session-scoped trained pipeline keeps the
integration tests honest (they score through the real Pipeline) without paying for
a fit per test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from churn.config import Config, load_config
from churn.data.make_dataset import clean_raw, generate_synthetic

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config(ROOT / "configs" / "config.yaml")


@pytest.fixture(scope="session")
def raw_frame(cfg: Config) -> Any:
    """1200 synthetic rows with the Telco schema, including the TotalCharges blanks."""
    return clean_raw(generate_synthetic(1200, seed=7), cfg)


@pytest.fixture(scope="session")
def trained_bundle(cfg: Config, raw_frame: Any) -> dict[str, Any]:
    """A small calibrated pipeline plus a threshold, shaped like models/champion.joblib."""
    from sklearn.calibration import CalibratedClassifierCV

    from churn.features.build_features import build_pipeline, split_xy
    from churn.models.baselines import make_estimator
    from churn.models.threshold import CostMatrix, optimal_threshold

    x, y = split_xy(raw_frame, cfg)
    pipeline = build_pipeline(cfg, make_estimator("logistic", cfg))
    calibrated = CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=3, ensemble=False)
    calibrated.fit(x, y)
    prob = calibrated.predict_proba(x)[:, 1]
    result = optimal_threshold(y, prob, CostMatrix.from_config(cfg.section("costs")))
    return {
        "pipeline": calibrated,
        "threshold": result.threshold,
        "metadata": {
            "model_name": "logistic",
            "model_version": "test-0.0.1",
            "trained_at": "2026-01-01T00:00:00+00:00",
            "feature_columns": cfg.feature_columns,
            "numeric_features": cfg.numeric_features,
            "categorical_features": cfg.categorical_features,
        },
    }


@pytest.fixture
def valid_customer() -> dict[str, Any]:
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.7,
        "TotalCharges": 287.1,
    }

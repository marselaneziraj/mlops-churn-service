"""Baselines you must beat before anyone believes the gradient-boosted model.

* ``majority``  - always predict the majority class. Fixes what accuracy means:
  on Telco churn it scores ~73% accuracy while catching zero churners, which is
  the clearest possible argument for AUC/PR and a cost metric.
* ``logistic``  - regularised logistic regression on the same ColumnTransformer.
  A strong, calibrated-by-construction linear reference. If LightGBM does not
  clear it by a margin worth the operational cost, ship the linear model.
* ``lightgbm``  - the candidate.

All three go through :func:`churn.features.build_features.build_pipeline`, so the
comparison differs only in the estimator.
"""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.config import Config
from churn.features.build_features import build_pipeline


def make_estimator(name: str, cfg: Config) -> Any:
    """Instantiate an estimator by name, seeded from the config."""
    key = name.lower()
    if key == "majority":
        return DummyClassifier(strategy="prior", random_state=cfg.seed)
    if key == "logistic":
        return LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight=None,
            solver="lbfgs",
            random_state=cfg.seed,
        )
    if key == "lightgbm":
        from lightgbm import LGBMClassifier

        params = dict(cfg.section("model")["params"])
        params.setdefault("random_state", cfg.seed)
        return LGBMClassifier(**params)
    raise ValueError(f"unknown estimator '{name}' (expected majority | logistic | lightgbm)")


def make_candidates(cfg: Config) -> dict[str, Pipeline]:
    """The full comparison set, in reporting order."""
    return {
        name: build_pipeline(cfg, make_estimator(name, cfg))
        for name in ("majority", "logistic", "lightgbm")
    }

"""Stage 3: the feature pipeline, as a sklearn ColumnTransformer inside a Pipeline.

Why this shape and not manual transforms
----------------------------------------
If you impute or encode a full DataFrame and *then* split, the imputation median
and the category vocabulary have seen the test rows. The leak is small but real,
and it is invisible in the metrics. Wrapping every transform in a Pipeline means
``fit`` only ever touches the training fold -- including inside every
cross-validation fold and inside the calibration wrapper -- and ``predict`` on the
service applies exactly the transforms that were fitted. The artefact that ships
to production is the whole Pipeline, so there is no separate preprocessing script
to keep in sync.

Numeric: median impute -> standard scale. The scaling is redundant for LightGBM
but required for the logistic-regression baseline, and one shared preprocessor
keeps the comparison honest.
Categorical: most-frequent impute -> one-hot, ``handle_unknown="ignore"`` so an
unseen level at serving time becomes an all-zero block rather than a 500.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import Config


def build_preprocessor(cfg: Config) -> ColumnTransformer:
    """ColumnTransformer covering the configured numeric and categorical columns."""
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, cfg.numeric_features),
            ("cat", categorical, cfg.categorical_features),
        ],
        remainder="drop",  # anything not listed is dropped, by construction
        verbose_feature_names_out=True,
    )


def build_pipeline(cfg: Config, estimator: object) -> Pipeline:
    """Preprocessor + estimator as one fittable, picklable object."""
    return Pipeline(steps=[("preprocess", build_preprocessor(cfg)), ("model", estimator)])


def split_xy(frame: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, np.ndarray]:
    """Select the configured feature columns and binarise the target.

    Selecting by an explicit allow-list (rather than ``drop(target)``) is what makes
    the forbidden-column check in the leakage audit meaningful: a new column added
    to the CSV cannot silently become a feature.
    """
    missing = [c for c in cfg.feature_columns if c not in frame.columns]
    if missing:
        raise KeyError(f"columns missing from the input frame: {missing}")
    x = frame.loc[:, cfg.feature_columns].copy()
    for col in cfg.numeric_features:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    for col in cfg.categorical_features:
        x[col] = x[col].astype(str)
    y = (frame[cfg.target].astype(str).str.strip() == cfg.positive_label).astype(int).to_numpy()
    return x, y


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Post-transform column names, for SHAP and coefficient tables."""
    return [str(n) for n in preprocessor.get_feature_names_out()]

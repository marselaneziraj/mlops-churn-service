"""Loads the serving bundle once per process and answers predictions.

The bundle (``models/champion.joblib``) carries the fitted Pipeline, the decision
threshold chosen from the cost matrix, and the metadata that becomes the
``X-Model-Version`` response header. Threshold and model travel together on
purpose: a threshold is only valid for the probability distribution of the model
it was tuned on, so shipping them separately is how a service quietly starts
making the wrong decisions after a retrain.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from churn.logging_utils import get_logger

log = get_logger(__name__)

DEFAULT_MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/champion.joblib"))


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested before a model is available."""


class ChurnModel:
    """Wrapper around the serialised bundle."""

    def __init__(self, pipeline: Any, threshold: float, metadata: dict[str, Any]) -> None:
        self.pipeline = pipeline
        self.threshold = float(threshold)
        self.metadata = metadata

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    @property
    def feature_columns(self) -> list[str]:
        return list(self.metadata.get("feature_columns", []))

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> ChurnModel:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"model bundle not found at {p.resolve()}. Run: python -m churn.models.train"
            )
        bundle = joblib.load(p)
        model = cls(bundle["pipeline"], bundle["threshold"], bundle.get("metadata", {}))
        log.info(
            "model loaded",
            extra={"path": str(p), "version": model.version, "threshold": model.threshold},
        )
        return model

    def to_frame(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Build a DataFrame with the exact training column order and dtypes."""
        frame = pd.DataFrame.from_records(records)
        for col in self.feature_columns:
            if col not in frame.columns:
                frame[col] = None
        frame = frame.loc[:, self.feature_columns]
        for col in self.metadata.get("numeric_features", []):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        for col in self.metadata.get("categorical_features", []):
            frame[col] = frame[col].astype(str)
        return frame

    def predict_proba(self, records: list[dict[str, Any]]) -> list[float]:
        frame = self.to_frame(records)
        return [float(p) for p in self.pipeline.predict_proba(frame)[:, 1]]


class ModelHolder:
    """Process-level singleton with an explicit degraded state.

    /health reports ``degraded`` rather than crashing on startup when the bundle is
    missing, so an orchestrator can distinguish "container is up, model is not" from
    "container is dead" -- different alerts, different runbooks.
    """

    def __init__(self) -> None:
        self._model: ChurnModel | None = None
        self.load_error: str | None = None

    def load(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        try:
            self._model = ChurnModel.load(path)
            self.load_error = None
        except Exception as exc:
            self._model = None
            self.load_error = str(exc)
            log.error("model load failed", extra={"error": str(exc)})

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> ChurnModel:
        if self._model is None:
            raise ModelNotLoadedError(self.load_error or "no model loaded")
        return self._model

    def set(self, model: ChurnModel) -> None:
        self._model = model
        self.load_error = None


holder = ModelHolder()

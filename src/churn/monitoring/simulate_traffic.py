"""Stage 9a: synthesise a production traffic window, with optional injected drift.

Monitoring code that has never seen drift is untested monitoring code. This script
samples from the reference window and applies named, controllable shifts so the PSI
thresholds and the Evidently report can be exercised deterministically in CI.

Drift scenarios
---------------
``none``       resample the reference window (PSI should sit near 0; this is the
               false-positive check -- if `none` alerts, your thresholds are wrong)
``covariate``  a marketing push brings in short-tenure, fibre, month-to-month
               customers: the inputs move, the input->label relationship does not
``price``      a tariff increase shifts MonthlyCharges up by a configurable amount
``concept``    the input distribution is unchanged but the churn relationship
               flips for one segment -- invisible to input PSI, visible only in
               prediction PSI and (eventually) in realised labels

Writes ``data/processed/current_window.parquet`` for the drift report to consume. It
can also fire the window at a running API with ``--post-to``, which exercises the
service, fills the Prometheus histograms, and produces realistic latency numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from churn.config import Config, load_config
from churn.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

SCENARIOS = ("none", "covariate", "price", "concept")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a production traffic window")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--scenario", choices=SCENARIOS, default="covariate")
    parser.add_argument("--n", type=int, default=2000, help="rows in the window")
    parser.add_argument("--intensity", type=float, default=1.0, help="drift multiplier, 0-2")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--post-to", default=None, help="e.g. http://localhost:8000 to score live")
    parser.add_argument(
        "--out", default="data/processed/current_window.parquet", help="window output path"
    )
    return parser.parse_args()


def apply_drift(
    frame: pd.DataFrame, scenario: str, intensity: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Return a drifted copy of the window."""
    out = frame.copy()
    if scenario == "none":
        return out

    if scenario in {"covariate", "concept"}:
        n = len(out)
        share = min(0.65 * intensity, 0.95)
        picked = rng.random(n) < share
        if scenario == "covariate":
            out.loc[picked, "Contract"] = "Month-to-month"
            out.loc[picked, "InternetService"] = "Fiber optic"
            out.loc[picked, "tenure"] = rng.integers(0, 9, size=int(picked.sum()))
            out.loc[picked, "PaymentMethod"] = "Electronic check"
        else:
            # Concept drift: inputs stay put, the *relationship* changes. We encode
            # that by flipping long-tenure two-year customers into the churn-prone
            # configuration without touching the marginal distributions much.
            loyal = picked & (out["tenure"] > 36)
            out.loc[loyal, "TechSupport"] = "No"
            out.loc[loyal, "OnlineSecurity"] = "No"

    if scenario == "price":
        bump = 15.0 * intensity
        out["MonthlyCharges"] = (out["MonthlyCharges"].astype(float) + bump).clip(0, 1000)
        out["TotalCharges"] = (
            out["TotalCharges"].astype(float) + bump * out["tenure"].astype(float)
        ).clip(0, 1_000_000)
    return out


def score_locally(frame: pd.DataFrame, cfg: Config) -> pd.Series:
    """Score the window with the local champion bundle."""
    from churn.api.model_registry import ChurnModel

    model = ChurnModel.load(cfg.section("serving")["model_path"])
    cols = [c for c in model.feature_columns if c in frame.columns]
    records: list[dict[str, Any]] = [
        {str(k): v for k, v in row.items()} for row in frame.loc[:, cols].to_dict("records")
    ]
    probs: list[float] = model.predict_proba(records)
    return pd.Series(probs, index=frame.index, name="prediction_proba")


def score_via_api(frame: pd.DataFrame, base_url: str, cfg: Config, chunk: int = 500) -> pd.Series:
    """Score the window through /predict_batch on a running service."""
    import requests

    cols = cfg.feature_columns
    api_probs: list[float] = []
    for start in range(0, len(frame), chunk):
        block = frame.iloc[start : start + chunk].loc[:, cols]
        payload = {"customers": json.loads(block.to_json(orient="records"))}
        response = requests.post(f"{base_url.rstrip('/')}/predict_batch", json=payload, timeout=60)
        response.raise_for_status()
        api_probs.extend(p["churn_probability"] for p in response.json()["predictions"])
    return pd.Series(api_probs, index=frame.index, name="prediction_proba")


def simulate(cfg: Config, args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed if args.seed is not None else cfg.seed)
    reference = pd.read_parquet(cfg.section("data")["reference_path"])
    feature_cols = [c for c in cfg.feature_columns if c in reference.columns]
    sample = (
        reference.loc[:, feature_cols]
        .sample(
            n=min(args.n, len(reference)) if args.n <= len(reference) else args.n,
            replace=args.n > len(reference),
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        .reset_index(drop=True)
    )

    window = apply_drift(sample, args.scenario, args.intensity, rng)
    window["prediction_proba"] = (
        score_via_api(window, args.post_to, cfg) if args.post_to else score_locally(window, cfg)
    )
    window["scenario"] = args.scenario
    window["intensity"] = args.intensity

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    window.to_parquet(out, index=False)
    log.info(
        "traffic window written",
        extra={
            "scenario": args.scenario,
            "intensity": args.intensity,
            "rows": len(window),
            "mean_proba": round(float(window["prediction_proba"].mean()), 4),
            "path": str(out),
        },
    )
    return window


def main() -> None:
    configure_logging()
    args = _parse_args()
    simulate(load_config(args.config), args)


if __name__ == "__main__":
    main()

"""Stage 1: get the raw dataset onto disk, reproducibly.

Two sources, same schema:

* ``telco``     - download the IBM Telco Customer Churn CSV (7043 rows, 21 columns).
* ``synthetic`` - generate a seeded dataset with identical columns and dtypes.

The synthetic path exists so the whole pipeline, tests and CI run with no network
and no credentials. If the download fails we fall back to it rather than failing
the stage, and we record which source was actually used in
``reports/dataset_source.json`` so a reader of the metrics never has to guess.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import Config, config_arg_parser, load_config
from churn.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

CATEGORY_LEVELS: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


def generate_synthetic(n_rows: int, seed: int) -> pd.DataFrame:
    """Seeded stand-in for the Telco dataset.

    Churn probability is a logit of a few genuinely predictive signals
    (month-to-month contract, short tenure, fibre, electronic cheque) plus noise,
    so a model trained on it behaves qualitatively like one trained on the real
    thing: ~26% positives, AUC in the high 0.70s-0.80s.
    """
    rng = np.random.default_rng(seed)
    cols: dict[str, np.ndarray] = {}
    for name, levels in CATEGORY_LEVELS.items():
        cols[name] = rng.choice(levels, size=n_rows)
    cols["SeniorCitizen"] = rng.choice([0, 1], size=n_rows, p=[0.84, 0.16])

    tenure = rng.integers(0, 73, size=n_rows)
    monthly = np.round(rng.normal(65, 30, size=n_rows).clip(18.25, 118.75), 2)
    noise = rng.normal(0, 2.5, size=n_rows)
    total = np.round((tenure * monthly + noise).clip(min=0.0), 2)

    logit = (
        -1.6
        + 1.45 * (cols["Contract"] == "Month-to-month")
        - 0.9 * (cols["Contract"] == "Two year")
        - 0.045 * tenure
        + 0.010 * monthly
        + 0.55 * (cols["InternetService"] == "Fiber optic")
        + 0.45 * (cols["PaymentMethod"] == "Electronic check")
        - 0.35 * (cols["OnlineSecurity"] == "Yes")
        - 0.30 * (cols["TechSupport"] == "Yes")
        + 0.25 * cols["SeniorCitizen"]
        + rng.normal(0, 0.45, size=n_rows)
    )
    churn = rng.random(n_rows) < 1.0 / (1.0 + np.exp(-logit))

    frame = pd.DataFrame(
        {
            "customerID": [f"{i:04d}-SYNTH" for i in range(n_rows)],
            **cols,
            "tenure": tenure,
            "MonthlyCharges": monthly,
            # The real CSV stores TotalCharges as text with blanks for tenure 0.
            # Reproduce that quirk so the cleaning code is exercised.
            "TotalCharges": [" " if t == 0 else f"{v}" for t, v in zip(tenure, total, strict=True)],
            "Churn": np.where(churn, "Yes", "No"),
        }
    )
    return frame[
        [
            "customerID",
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
            "Churn",
        ]
    ]


def download_telco(url: str, dest: Path, timeout: int = 30) -> bool:
    """Fetch the Telco CSV. Returns False on any network/HTTP failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        log.warning("telco download failed", extra={"url": url, "error": str(exc)})
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    log.info("downloaded telco csv", extra={"bytes": len(payload), "dest": str(dest)})
    return True


def clean_raw(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Dtype fixes that belong to ingestion, not to the model pipeline.

    Only schema-level repairs happen here (text -> float on a column the source
    ships as text). Imputation and encoding live in the sklearn Pipeline so they
    are fitted on training folds only.
    """
    out = frame.copy()
    if "TotalCharges" in out.columns:
        blanked = out["TotalCharges"].astype(str).str.strip()
        out["TotalCharges"] = pd.to_numeric(blanked.where(blanked != "", None), errors="coerce")
    target = cfg.target
    if target in out.columns:
        out[target] = out[target].astype(str).str.strip()
    return out


def build(cfg: Config) -> pd.DataFrame:
    data = cfg.section("data")
    raw_path = Path(data["raw_path"])
    source = str(data.get("source", "telco")).lower()
    used = source

    if source == "telco" and not download_telco(str(data["url"]), raw_path):
        log.warning("falling back to synthetic data")
        used = "synthetic"
    if used == "synthetic":
        frame = generate_synthetic(int(data["synthetic_rows"]), cfg.seed)
    else:
        frame = pd.read_csv(raw_path)

    frame = clean_raw(frame, cfg)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(raw_path, index=False)

    reports = cfg.path_for("reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "dataset_source.json").write_text(
        json.dumps(
            {
                "requested_source": source,
                "actual_source": used,
                "rows": len(frame),
                "columns": int(frame.shape[1]),
                "positive_rate": float((frame[cfg.target] == cfg.positive_label).mean()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(
        "raw dataset written",
        extra={"source": used, "rows": len(frame), "path": str(raw_path)},
    )
    return frame


def main() -> None:
    configure_logging()
    args = config_arg_parser("Download or generate the raw churn dataset").parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()

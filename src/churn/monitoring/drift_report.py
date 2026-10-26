"""Stage 9b: drift report + alerting logic.

Produces three things from the reference window and the current traffic window:

1. ``reports/psi.json`` - PSI per input feature plus PSI on the predicted
   probability, with a severity per the configured thresholds.
2. ``reports/drift_report.html`` - an Evidently ``DataDriftPreset`` report, which
   adds per-feature statistical tests (KS for numeric, chi-square for categorical)
   and a share-of-drifted-columns summary. Skipped with a warning if Evidently is
   not installed; PSI alone is enough to gate CI.
3. ``docs/figures/drift_psi.png`` - the PSI bar chart with the warn/alert lines.

Exit code is 1 when any signal is ``significant``, so the same command works as a
cron job, a CI gate, and an on-call alert source. Prediction PSI is weighted more
heavily than any single input: an input can move without consequence, but the
output distribution moving means the decisions the business acts on have changed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from churn.config import Config, load_config
from churn.logging_utils import configure_logging, get_logger
from churn.monitoring.psi import feature_psi, prediction_psi

log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PSI + Evidently drift report")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--current", default="data/processed/current_window.parquet")
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        default=True,
        help="exit 1 when any signal reaches the alert threshold (default: on)",
    )
    parser.add_argument("--no-fail-on-alert", dest="fail_on_alert", action="store_false")
    return parser.parse_args()


def plot_psi(rows: list[dict[str, Any]], warn: float, alert: float, out_path: Path) -> Path:
    frame = pd.DataFrame(rows).sort_values("psi")
    colors = {"none": "#4C72B0", "moderate": "#DD8452", "significant": "#C44E52"}
    fig, ax = plt.subplots(figsize=(7.0, max(3.0, 0.32 * len(frame))))
    ax.barh(
        frame["feature"],
        frame["psi"],
        color=[colors[s] for s in frame["severity"]],
        height=0.7,
    )
    ax.axvline(warn, color="#DD8452", ls="--", lw=1.2, label=f"warn {warn}")
    ax.axvline(alert, color="#C44E52", ls="--", lw=1.4, label=f"alert {alert}")
    ax.set_xlabel("PSI (reference vs current window)")
    ax.set_title("Population Stability Index by feature")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, axis="x")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def evidently_report(
    reference: pd.DataFrame, current: pd.DataFrame, cols: list[str], out_path: Path
) -> Path | None:
    """Evidently DataDriftPreset. Returns None when Evidently is not installed."""
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError:  # pragma: no cover - optional extra
        log.warning(
            "evidently not installed; skipping the HTML report (pip install '.[monitoring]')"
        )
        return None
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference.loc[:, cols], current_data=current.loc[:, cols])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_path))
    return out_path


def run(cfg: Config, current_path: str) -> dict[str, Any]:
    mon = cfg.section("monitoring")
    warn = float(mon.get("psi_warn", 0.1))
    alert = float(mon.get("psi_alert", 0.25))
    bins = int(mon.get("psi_bins", 10))

    reference = pd.read_parquet(cfg.section("data")["reference_path"])
    current = pd.read_parquet(current_path)

    results = feature_psi(
        reference,
        current,
        numeric=[c for c in cfg.numeric_features if c in current.columns],
        categorical=[c for c in cfg.categorical_features if c in current.columns],
        bins=bins,
        warn=warn,
        alert=alert,
    )
    rows = [asdict(r) for r in results]
    pred = None
    if "prediction_proba" in reference.columns and "prediction_proba" in current.columns:
        pred = prediction_psi(
            reference["prediction_proba"],
            current["prediction_proba"],
            bins=bins,
            warn=warn,
            alert=alert,
        )
        rows.append(asdict(pred))

    alerts = [r for r in rows if r["severity"] == "significant"]
    warnings = [r for r in rows if r["severity"] == "moderate"]
    summary: dict[str, Any] = {
        "reference_rows": len(reference),
        "current_rows": len(current),
        "scenario": str(current["scenario"].iloc[0])
        if "scenario" in current.columns
        else "unknown",
        "thresholds": {"warn": warn, "alert": alert},
        "max_psi": max((r["psi"] for r in rows), default=0.0),
        "prediction_psi": pred.psi if pred else None,
        "n_alert": len(alerts),
        "n_warn": len(warnings),
        "status": "ALERT" if alerts else ("WARN" if warnings else "OK"),
        "alerting_features": [r["feature"] for r in alerts],
        "features": rows,
        "mean_prediction_reference": (
            float(reference["prediction_proba"].mean()) if "prediction_proba" in reference else None
        ),
        "mean_prediction_current": (
            float(current["prediction_proba"].mean()) if "prediction_proba" in current else None
        ),
    }

    reports_dir = cfg.path_for("reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "psi.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure = plot_psi(rows, warn, alert, cfg.path_for("figures_dir") / "drift_psi.png")
    html = evidently_report(
        reference,
        current,
        [c for c in cfg.feature_columns if c in current.columns],
        reports_dir / "drift_report.html",
    )
    log.info(
        "drift report complete",
        extra={
            "status": summary["status"],
            "max_psi": round(summary["max_psi"], 4),
            "alerting": summary["alerting_features"],
            "figure": str(figure),
            "html": str(html) if html else None,
        },
    )
    return summary


def main() -> None:
    configure_logging()
    args = _parse_args()
    summary = run(load_config(args.config), args.current)
    print(json.dumps({k: v for k, v in summary.items() if k != "features"}, indent=2))
    if args.fail_on_alert and summary["status"] == "ALERT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

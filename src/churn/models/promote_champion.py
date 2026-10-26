"""Stage 5: model registry promotion.

Why a registry
--------------
Experiment tracking answers "what did run 7f3a do?". A registry answers "what is
in production right now, and what was there last Tuesday?". It gives each version
of a model a number, an alias (``champion``, ``challenger``), and an audit trail of
who moved it. Without one, the deployed artefact is whatever file happened to be on
the box, and a rollback is an archaeology exercise.

This script compares the newest registered version against the current champion on
a single promotion metric and only moves the alias if the challenger wins by at
least ``--min-delta``. The gate is deliberately conservative: a 0.001 AUC gain is
noise, and every promotion costs a deployment.

Uses the alias API (MLflow >= 2.9). Stages (``Staging``/``Production``) are deprecated.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from churn.config import load_config
from churn.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote the best model version to champion")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--metric",
        default="test_calibrated_roc_auc",
        help="run metric used for the comparison (default: test_calibrated_roc_auc)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.002,
        help="challenger must beat the champion by at least this much (default: 0.002)",
    )
    parser.add_argument(
        "--max-brier",
        type=float,
        default=0.25,
        help="reject any candidate whose test Brier score exceeds this (default: 0.25)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report the decision, change nothing"
    )
    return parser.parse_args()


def _metric(client: Any, version: Any, name: str) -> float | None:
    run = client.get_run(version.run_id)
    value = run.data.metrics.get(name)
    return float(value) if value is not None else None


def promote(
    model_name: str,
    metric: str,
    min_delta: float,
    max_brier: float,
    tracking_uri: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    import mlflow
    from mlflow.tracking import MlflowClient

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise SystemExit(f"no registered versions for model '{model_name}'")
    latest = max(versions, key=lambda v: int(v.version))

    challenger_metric = _metric(client, latest, metric)
    challenger_brier = _metric(client, latest, "test_calibrated_brier")
    decision: dict[str, Any] = {
        "model": model_name,
        "candidate_version": latest.version,
        "metric": metric,
        "candidate_metric": challenger_metric,
        "candidate_brier": challenger_brier,
        "min_delta": min_delta,
    }

    if challenger_metric is None:
        decision.update(promoted=False, reason=f"candidate run has no metric '{metric}'")
        return decision
    if challenger_brier is not None and challenger_brier > max_brier:
        decision.update(
            promoted=False,
            reason=f"candidate Brier {challenger_brier:.4f} exceeds the {max_brier} ceiling",
        )
        return decision

    try:
        current = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except Exception:
        current = None

    if current is None:
        decision.update(
            champion_version=None,
            champion_metric=None,
            promoted=True,
            reason="no champion yet; promoting the first qualifying version",
        )
    else:
        champion_metric = _metric(client, current, metric)
        decision.update(champion_version=current.version, champion_metric=champion_metric)
        if champion_metric is not None and challenger_metric < champion_metric + min_delta:
            decision.update(
                promoted=False,
                reason=(
                    f"candidate {challenger_metric:.4f} does not beat champion "
                    f"{champion_metric:.4f} by {min_delta}"
                ),
            )
            return decision
        decision.update(promoted=True, reason="candidate clears the promotion gate")

    if not dry_run:
        if current is not None:
            client.set_registered_model_alias(model_name, CHALLENGER_ALIAS, current.version)
        client.set_registered_model_alias(model_name, CHAMPION_ALIAS, latest.version)
        client.set_model_version_tag(
            model_name, latest.version, "promoted_by", "promote_champion.py"
        )
        client.set_model_version_tag(model_name, latest.version, "promotion_metric", metric)
    decision["dry_run"] = dry_run
    return decision


def main() -> None:
    configure_logging()
    args = _parse_args()
    cfg = load_config(args.config)
    tracking = cfg.section("tracking")
    decision = promote(
        model_name=str(tracking.get("registered_model", "churn-classifier")),
        metric=args.metric,
        min_delta=args.min_delta,
        max_brier=args.max_brier,
        tracking_uri=tracking.get("tracking_uri"),
        dry_run=args.dry_run,
    )
    log.info("promotion decision", extra=decision)
    out = cfg.path_for("reports_dir") / "promotion.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

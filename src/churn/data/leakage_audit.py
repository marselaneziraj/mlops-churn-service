"""Stage 2: leakage audit.

Target leakage is a feature that encodes the answer. It inflates offline metrics
and vanishes in production, so it is the single most expensive failure mode in a
tabular project. This script runs four mechanical checks and writes a verdict to
``reports/leakage_audit.json``. It exits non-zero on a FAIL, which makes it a gate
in CI and a DVC stage that cannot silently pass.

Checks
------
1. Forbidden columns (ids, the target itself) are absent from the feature list.
2. No feature is near-perfectly correlated with the target (point-biserial |r| > 0.95
   for numeric, Cramer's V > 0.95 for categorical).
3. No single feature trained alone reaches AUC > 0.99.
4. No duplicated rows across the split boundary (the same customer twice).

Point 2 and 3 are deliberately permissive: a genuinely strong feature is not
leakage. The thresholds flag only features that are effectively the label.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from churn.config import Config, config_arg_parser, load_config
from churn.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

CORR_THRESHOLD = 0.95
SOLO_AUC_THRESHOLD = 0.99


@dataclass
class Finding:
    check: str
    status: str  # PASS | WARN | FAIL
    detail: str
    evidence: dict[str, Any]


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Association between two categorical variables, bias-corrected (Bergsma 2013)."""
    table = pd.crosstab(x, y)
    if table.size == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    n = observed.sum()
    row = observed.sum(axis=1, keepdims=True)
    col = observed.sum(axis=0, keepdims=True)
    expected = row @ col / n
    safe_expected = np.where(expected == 0, np.nan, expected)
    chi2 = float(np.nansum((observed - expected) ** 2 / safe_expected))
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def check_forbidden(cfg: Config) -> Finding:
    used = set(cfg.feature_columns)
    offenders = sorted(used & set(cfg.forbidden_columns))
    return Finding(
        check="forbidden_columns",
        status="FAIL" if offenders else "PASS",
        detail=(
            f"forbidden columns present in the feature list: {offenders}"
            if offenders
            else "no identifier or target column is used as a feature"
        ),
        evidence={"forbidden": cfg.forbidden_columns, "offenders": offenders},
    )


def check_correlation(frame: pd.DataFrame, cfg: Config) -> Finding:
    y = (frame[cfg.target] == cfg.positive_label).astype(int)
    scores: dict[str, float] = {}
    for col in cfg.numeric_features:
        if col not in frame:
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        mask = series.notna()
        if mask.sum() < 10 or series[mask].nunique() < 2:
            continue
        scores[col] = float(abs(np.corrcoef(series[mask], y[mask])[0, 1]))
    for col in cfg.categorical_features:
        if col not in frame:
            continue
        scores[col] = cramers_v(frame[col].astype(str), y.astype(str))
    offenders = {k: v for k, v in scores.items() if v > CORR_THRESHOLD}
    return Finding(
        check="target_association",
        status="FAIL" if offenders else "PASS",
        detail=(
            f"features almost identical to the target: {sorted(offenders)}"
            if offenders
            else f"max association {max(scores.values(), default=0.0):.3f} <= {CORR_THRESHOLD}"
        ),
        evidence={
            "scores": {k: round(v, 4) for k, v in sorted(scores.items(), key=lambda kv: -kv[1])}
        },
    )


def check_solo_auc(frame: pd.DataFrame, cfg: Config) -> Finding:
    """A single feature that separates the classes almost perfectly is the label in disguise."""
    y = (frame[cfg.target] == cfg.positive_label).astype(int).to_numpy()
    aucs: dict[str, float] = {}
    for col in cfg.feature_columns:
        if col not in frame:
            continue
        series = frame[col]
        if col in cfg.categorical_features:
            # Encode by in-sample positive rate: the most generous (leakiest)
            # single-feature encoding available, so this is a worst case.
            rate = y.mean()
            mapping = pd.Series(y, index=series.astype(str).to_numpy()).groupby(level=0).mean()
            values = series.astype(str).map(mapping).fillna(rate).to_numpy(dtype=float)
        else:
            values = (
                pd.to_numeric(series, errors="coerce")
                .fillna(pd.to_numeric(series, errors="coerce").median())
                .to_numpy(dtype=float)
            )
        if len(np.unique(values)) < 2:
            continue
        aucs[col] = float(max(roc_auc_score(y, values), 1 - roc_auc_score(y, values)))
    offenders = {k: v for k, v in aucs.items() if v > SOLO_AUC_THRESHOLD}
    return Finding(
        check="single_feature_auc",
        status="FAIL" if offenders else "PASS",
        detail=(
            f"single features with near-perfect AUC: {sorted(offenders)}"
            if offenders
            else (
                f"max single-feature AUC {max(aucs.values(), default=0.0):.3f} "
                f"<= {SOLO_AUC_THRESHOLD}"
            )
        ),
        evidence={
            "auc": {k: round(v, 4) for k, v in sorted(aucs.items(), key=lambda kv: -kv[1])[:10]}
        },
    )


def check_duplicates(frame: pd.DataFrame, cfg: Config) -> Finding:
    id_col = cfg.section("data").get("id_column")
    dup_ids = 0
    if id_col and id_col in frame:
        dup_ids = int(frame[id_col].duplicated().sum())
    dup_rows = int(frame.drop(columns=[c for c in [id_col] if c in frame]).duplicated().sum())
    status = "FAIL" if dup_ids else ("WARN" if dup_rows else "PASS")
    return Finding(
        check="duplicates",
        status=status,
        detail=(
            f"{dup_ids} duplicated entity ids - the same customer can land in both "
            "train and test, which leaks"
            if dup_ids
            else f"{dup_rows} duplicated feature rows (benign for a survey-style dataset)"
        ),
        evidence={"duplicate_ids": dup_ids, "duplicate_feature_rows": dup_rows},
    )


def run_audit(cfg: Config) -> dict[str, Any]:
    frame = pd.read_csv(cfg.section("data")["raw_path"])
    findings = [
        check_forbidden(cfg),
        check_correlation(frame, cfg),
        check_solo_auc(frame, cfg),
        check_duplicates(frame, cfg),
    ]
    verdict = (
        "FAIL"
        if any(f.status == "FAIL" for f in findings)
        else ("WARN" if any(f.status == "WARN" for f in findings) else "PASS")
    )
    report: dict[str, Any] = {
        "verdict": verdict,
        "rows": len(frame),
        "positive_rate": float((frame[cfg.target] == cfg.positive_label).mean()),
        "findings": [asdict(f) for f in findings],
    }
    out = cfg.path_for("reports_dir") / "leakage_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for f in findings:
        log.info("leakage check", extra={"check": f.check, "status": f.status, "detail": f.detail})
    log.info("leakage audit complete", extra={"verdict": verdict, "report": str(out)})
    return report


def main() -> None:
    configure_logging()
    args = config_arg_parser("Audit the raw dataset for target leakage").parse_args()
    report = run_audit(load_config(args.config))
    if report["verdict"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

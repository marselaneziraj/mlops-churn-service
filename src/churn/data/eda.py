"""Stage 2b: EDA figures.

Three figures, each answering a question the modelling decisions depend on:
target balance (does the metric need to be AUC/PR rather than accuracy?),
numeric distributions split by outcome (are the numeric features informative,
are there blanks?), and churn rate by category level (which categorical levels
carry signal?). Figures are written to docs/figures so the README can embed them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from churn.config import Config, config_arg_parser, load_config
from churn.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

PALETTE = {"stay": "#4C72B0", "churn": "#C44E52"}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def plot_target_balance(frame: pd.DataFrame, cfg: Config, out: Path) -> Path:
    counts = frame[cfg.target].value_counts()
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    colors = [
        PALETTE["churn"] if str(i) == cfg.positive_label else PALETTE["stay"] for i in counts.index
    ]
    bars = ax.bar(counts.index.astype(str), counts.to_numpy(), color=colors, width=0.6)
    total = int(counts.sum())
    for bar, value in zip(bars, counts.to_numpy(), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value),
            f"{value:,}\n({value / total:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("customers")
    ax.set_title(f"Target balance (n={total:,})")
    ax.set_ylim(0, counts.max() * 1.22)
    path = out / "eda_target_balance.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_numeric_distributions(frame: pd.DataFrame, cfg: Config, out: Path) -> Path:
    numeric = [c for c in cfg.numeric_features if c in frame.columns]
    fig, axes = plt.subplots(1, len(numeric), figsize=(4.2 * len(numeric), 3.4))
    axes = [axes] if len(numeric) == 1 else list(axes)
    is_pos = frame[cfg.target] == cfg.positive_label
    for ax, col in zip(axes, numeric, strict=True):
        series = pd.to_numeric(frame[col], errors="coerce")
        ax.hist(
            series[~is_pos].dropna(), bins=30, alpha=0.65, label="stayed", color=PALETTE["stay"]
        )
        ax.hist(
            series[is_pos].dropna(), bins=30, alpha=0.65, label="churned", color=PALETTE["churn"]
        )
        missing = int(series.isna().sum())
        ax.set_title(f"{col}" + (f"  ({missing} missing)" if missing else ""))
        ax.set_xlabel(col)
    axes[0].set_ylabel("customers")
    axes[0].legend(frameon=False)
    fig.suptitle("Numeric features by outcome", y=1.02)
    path = out / "eda_numeric_distributions.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_churn_by_category(frame: pd.DataFrame, cfg: Config, out: Path, top: int = 6) -> Path:
    base = float((frame[cfg.target] == cfg.positive_label).mean())
    cats = [c for c in cfg.categorical_features if c in frame.columns]
    spread = {
        c: float(
            frame.groupby(c)[cfg.target].apply(lambda s: (s == cfg.positive_label).mean()).max()
            - frame.groupby(c)[cfg.target].apply(lambda s: (s == cfg.positive_label).mean()).min()
        )
        for c in cats
    }
    chosen = [c for c, _ in sorted(spread.items(), key=lambda kv: -kv[1])[:top]]
    ncols = 3
    nrows = (len(chosen) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.1 * nrows))
    flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, col in zip(flat, chosen, strict=False):
        rates = (
            frame.groupby(col)[cfg.target]
            .apply(lambda s: (s == cfg.positive_label).mean())
            .sort_values()
        )
        ax.barh(rates.index.astype(str), rates.to_numpy(), color=PALETTE["churn"], alpha=0.85)
        ax.axvline(base, color="#333333", ls="--", lw=1)
        ax.set_title(col)
        ax.set_xlim(0, max(0.6, float(rates.max()) * 1.15))
        ax.tick_params(axis="y", labelsize=8)
    for ax in flat[len(chosen) :]:
        ax.axis("off")
    fig.suptitle(f"Churn rate by level (dashed line = base rate {base:.1%})", y=1.01)
    fig.tight_layout()
    path = out / "eda_churn_by_category.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def run_eda(cfg: Config) -> list[Path]:
    _style()
    frame = pd.read_csv(cfg.section("data")["raw_path"])
    out = cfg.path_for("figures_dir")
    out.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_target_balance(frame, cfg, out),
        plot_numeric_distributions(frame, cfg, out),
        plot_churn_by_category(frame, cfg, out),
    ]
    log.info("eda figures written", extra={"paths": [str(p) for p in paths]})
    return paths


def main() -> None:
    configure_logging()
    args = config_arg_parser("Generate EDA figures").parse_args()
    run_eda(load_config(args.config))


if __name__ == "__main__":
    main()

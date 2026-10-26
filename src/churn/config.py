"""Typed access to configs/config.yaml.

One YAML file drives ingestion, training, serving and monitoring. Scripts read
it through :func:`load_config` so that a parameter never exists in two places.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


@dataclass(frozen=True)
class Config:
    """Thin wrapper over the parsed YAML tree.

    Keeps dictionary access (so adding a key needs no code change) while giving
    a few typed helpers for the values touched everywhere.
    """

    raw: dict[str, Any] = field(repr=False)
    path: Path = DEFAULT_CONFIG_PATH

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise KeyError(f"config section '{name}' is missing or not a mapping")
        return value

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    @property
    def target(self) -> str:
        return str(self.section("data")["target"])

    @property
    def positive_label(self) -> str:
        return str(self.section("data")["positive_label"])

    @property
    def numeric_features(self) -> list[str]:
        return list(self.section("features")["numeric"])

    @property
    def categorical_features(self) -> list[str]:
        return list(self.section("features")["categorical"])

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric_features + self.categorical_features

    @property
    def forbidden_columns(self) -> list[str]:
        return list(self.section("features").get("forbidden", []))

    def path_for(self, key: str) -> Path:
        return Path(self.section("paths")[key])


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read a YAML config from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"config at {p} did not parse to a mapping")
    return Config(raw=raw, path=p)


def config_arg_parser(description: str) -> argparse.ArgumentParser:
    """Every entry point takes the same --config flag."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="path to the YAML config (default: configs/config.yaml)",
    )
    return parser

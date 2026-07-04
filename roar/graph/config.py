"""Loads experiments/configs/data.yaml — the single source of truth for every
parameter the data/graph pipeline uses (CLAUDE.md rule 3: no hard-coded
parameters inside pipeline code)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "experiments" / "configs" / "data.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)

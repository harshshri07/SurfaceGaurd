from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: str | Path, maybe_rel: str | Path) -> Path:
    base_p = Path(base)
    p = Path(maybe_rel)
    return p if p.is_absolute() else (base_p / p).resolve()


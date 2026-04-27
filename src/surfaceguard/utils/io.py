from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    np.savez_compressed(str(p), **arrays)


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def save_torch(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    torch.save(obj, str(p))


def load_torch(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    return torch.load(str(path), map_location=map_location)


from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any, Dict, Optional


def get_git_commit_short(cwd: Optional[str] = None) -> Optional[str]:
    """
    Best-effort git commit id for provenance.
    Returns None when git isn't available (e.g. packaged deployment).
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def stable_cfg_hash(cfg: Dict[str, Any]) -> str:
    """
    Hash a config dict in a stable way (sorted keys, JSON).
    """
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


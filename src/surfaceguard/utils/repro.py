from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, List

import torch

from surfaceguard.utils.io import ensure_dir


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return ""


def collect_env_info() -> Dict[str, Any]:
    return {
        "timestamp_utc": utc_now_iso(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        },
    }


def collect_git_info() -> Dict[str, str]:
    return {
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "status_short": _run(["git", "status", "--short"]),
    }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def snapshot_configs(paths: Iterable[str | Path]) -> List[Dict[str, Any]]:
    snaps: List[Dict[str, Any]] = []
    for p in paths:
        path = Path(p)
        item: Dict[str, Any] = {"path": str(path)}
        if path.exists():
            item.update(
                {
                    "sha256": _file_sha256(path),
                    "content": path.read_text(encoding="utf-8"),
                }
            )
        else:
            item["missing"] = True
        snaps.append(item)
    return snaps


def write_run_manifest(
    output_root: str | Path,
    run_name: str,
    args: Dict[str, Any],
    config_paths: Iterable[str | Path] = (),
    extra: Dict[str, Any] | None = None,
) -> str:
    out_dir = ensure_dir(Path(output_root) / "manifests")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{run_name}_{stamp}.json"
    payload: Dict[str, Any] = {
        "run_name": run_name,
        "args": args,
        "env": collect_env_info(),
        "git": collect_git_info(),
        "config_snapshots": snapshot_configs(config_paths),
    }
    if extra:
        payload["extra"] = extra
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)

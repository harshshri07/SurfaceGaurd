from __future__ import annotations

from pathlib import Path

import streamlit as st

from surfaceguard.utils.config import load_yaml
from surfaceguard.ui.inference_view import render_inference_page


def _auto_detect_outputs_dir(configured: str) -> str:
    """
    Pick an outputs directory that actually exists for this runtime context.
    Prioritize paths that contain PatchCore checkpoints.
    """
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(configured)

    candidates = []
    if cfg_path.is_absolute():
        candidates.append(cfg_path)
    else:
        candidates.append((Path.cwd() / cfg_path).resolve())
        candidates.append((repo_root / cfg_path).resolve())

    # Always try common defaults in case config/cwd differ.
    candidates.extend(
        [
            (Path.cwd() / "outputs").resolve(),
            (repo_root / "outputs").resolve(),
        ]
    )

    seen = set()
    unique_candidates = []
    for c in candidates:
        s = str(c)
        if s not in seen:
            seen.add(s)
            unique_candidates.append(c)

    def has_patchcore_ckpt(p: Path) -> bool:
        pc = p / "patchcore"
        return pc.exists() and any(d.is_dir() for d in pc.iterdir())

    for c in unique_candidates:
        if has_patchcore_ckpt(c):
            return str(c)
    for c in unique_candidates:
        if c.exists():
            return str(c)
    return str(unique_candidates[0]) if unique_candidates else configured


def main() -> None:
    st.set_page_config(page_title="SurfaceGuard", layout="wide")

    cfg_path = Path("configs/app.yaml")
    if cfg_path.exists():
        cfg = load_yaml(cfg_path)
    else:
        cfg = {"app": {"outputs_dir": "outputs"}}

    configured_outputs = str(cfg["app"].get("outputs_dir", "outputs"))
    st.session_state.setdefault("outputs_dir", _auto_detect_outputs_dir(configured_outputs))

    st.title("SurfaceGuard")
    st.caption("Industrial surface defect detection + localization (PatchCore + WinCLIP + Hybrid).")

    render_inference_page(title="Inference")


if __name__ == "__main__":
    main()


from __future__ import annotations

from pathlib import Path

import streamlit as st

from surfaceguard.utils.config import load_yaml
from surfaceguard.ui.inference_view import render_inference_page


def main() -> None:
    st.set_page_config(page_title="SurfaceGuard", layout="wide")

    cfg_path = Path("configs/app.yaml")
    if cfg_path.exists():
        cfg = load_yaml(cfg_path)
    else:
        cfg = {"app": {"outputs_dir": "outputs"}}

    st.session_state.setdefault("outputs_dir", cfg["app"].get("outputs_dir", "outputs"))

    st.title("SurfaceGuard")
    st.caption("Industrial surface defect detection + localization (PatchCore + WinCLIP).")

    render_inference_page(title="Inference")


if __name__ == "__main__":
    main()


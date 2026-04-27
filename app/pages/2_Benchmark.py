from __future__ import annotations

from pathlib import Path

import streamlit as st

from surfaceguard.eval.runner import eval_method_for_category
from surfaceguard.utils.config import load_yaml


def main() -> None:
    st.header("Benchmark (MVTec AD)")
    st.caption("Runs evaluation and writes a JSON report into outputs/.")

    method = st.selectbox("Method", ["patchcore", "winclip"], index=0)
    category = st.text_input("Category", value="bottle")
    config_path = st.text_input(
        "Config path",
        value="configs/patchcore_mvtec.yaml" if method == "patchcore" else "configs/winclip_mvtec.yaml",
    )

    if st.button("Run evaluation"):
        cfg = load_yaml(config_path)
        with st.spinner("Evaluating..."):
            report_path = eval_method_for_category(cfg, method=method, category=category)
        st.success(f"Done. Report written to: {report_path}")


if __name__ == "__main__":
    main()


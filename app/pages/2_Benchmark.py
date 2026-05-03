from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from surfaceguard.data.mvtec import MVTec_AD_15_CATEGORIES, resolve_mvtec_categories
from surfaceguard.eval.runner import eval_method_for_category
from surfaceguard.utils.config import load_yaml


def main() -> None:
    st.header("Benchmark (MVTec AD)")
    st.caption("Runs evaluation and writes per-category JSON reports into outputs/.")

    method = st.selectbox("Method", ["patchcore", "winclip"], index=0)
    categories = st.multiselect(
        "Categories",
        list(MVTec_AD_15_CATEGORIES),
        default=list(MVTec_AD_15_CATEGORIES),
    )
    config_path = st.text_input(
        "Config path",
        value="configs/patchcore_mvtec.yaml" if method == "patchcore" else "configs/winclip_mvtec.yaml",
    )

    if st.button("Run evaluation"):
        cfg = load_yaml(config_path)
        selected_categories = resolve_mvtec_categories(str(cfg["data"]["root"]), categories)
        if not selected_categories:
            st.error("No valid categories selected.")
            return
        reports = []
        with st.spinner("Evaluating..."):
            for cat in selected_categories:
                report_path = eval_method_for_category(cfg, method=method, category=cat)
                reports.append(json.loads(Path(report_path).read_text(encoding="utf-8")))
        st.success(f"Done. Generated {len(reports)} report(s).")
        st.dataframe(
            [
                {
                    "category": r["category"],
                    "image_auroc": r.get("image_auroc"),
                    "pixel_auroc": r.get("pixel_auroc"),
                    "au_pro": r.get("au_pro"),
                    "best_f1": r.get("best_f1", {}).get("f1"),
                }
                for r in reports
            ],
            width="stretch",
        )


if __name__ == "__main__":
    main()


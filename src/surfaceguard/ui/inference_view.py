from __future__ import annotations

import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from surfaceguard.eval.visualize import overlay_heatmap, threshold_mask
from surfaceguard.ui.model_cache import load_patchcore_model, load_winclip_model


def _read_image(uploaded) -> np.ndarray:
    data = uploaded.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")
    return bgr


def _trained_patchcore_categories(outputs_dir: str) -> list[str]:
    p = Path(outputs_dir) / "patchcore"
    if not p.exists():
        return []
    return sorted([d.name for d in p.iterdir() if d.is_dir()])


def render_inference_page(title: str = "Inference") -> None:
    st.header(title)

    with st.sidebar:
        st.subheader("Settings")
        method = st.selectbox("Method", ["patchcore", "winclip"], index=0)
        outputs_dir = st.session_state.get("outputs_dir", "outputs")
        category = "auto"
        with st.expander("Advanced", expanded=False):
            outputs_dir = st.text_input("Outputs dir", value=outputs_dir)
            if method == "patchcore":
                trained_for_advanced = _trained_patchcore_categories(outputs_dir)
                if trained_for_advanced:
                    category = st.selectbox("Force category", ["auto"] + trained_for_advanced, index=0)
            else:
                category = st.text_input("Prompt hint (optional)", value="auto")
        st.session_state["outputs_dir"] = outputs_dir
        trained = _trained_patchcore_categories(outputs_dir)
        if method == "patchcore":
            if trained:
                st.caption("PatchCore runs per-category checkpoints from outputs/patchcore/<category>/")
            else:
                st.warning("No PatchCore checkpoints found. Train first (example): python tools\\train_patchcore.py --category bottle")
        else:
            # Keep UX simple: category is optional for WinCLIP; default to category-agnostic prompts.
            st.caption("WinCLIP is training-free; category is optional (auto uses generic prompts).")
            quality = st.selectbox("WinCLIP localization quality", ["fast", "balanced", "best"], index=1)
            if quality == "fast":
                win_stride, blur_sigma = 112, 0.0
            elif quality == "balanced":
                win_stride, blur_sigma = 56, 3.0
            else:
                win_stride, blur_sigma = 28, 5.0

        st.subheader("Mask")
        mask_mode = st.selectbox("Mask threshold", ["p99", "p95", "otsu", "manual"], index=0)
        manual_thr = None
        if mask_mode == "manual":
            manual_thr = st.slider("Manual threshold (0–1)", min_value=0.0, max_value=1.0, value=0.85, step=0.01)

    def run_inference(image_bgr: np.ndarray) -> dict:
        chosen_category = category
        if method == "patchcore":
            trained_local = trained
            if chosen_category == "auto":
                if not trained_local:
                    raise FileNotFoundError("No PatchCore checkpoints found under outputs/patchcore/")
                best_score = None
                best_out = None
                best_cat = None
                for cat in trained_local:
                    m = load_patchcore_model(outputs_dir, cat)
                    o = m.predict(image_bgr)
                    s = float(o["score"])
                    if best_score is None or s < best_score:
                        best_score, best_out, best_cat = s, o, cat
                out = best_out
                chosen_category = str(best_cat)
            else:
                model = load_patchcore_model(outputs_dir, chosen_category)
                out = model.predict(image_bgr)
        else:
            # Denser stride + smoothing yields a significantly less blocky heatmap.
            model = load_winclip_model(
                outputs_dir,
                chosen_category,
                window_size=224,
                window_stride=win_stride,
                blur_sigma=blur_sigma,
            )
            out = model.predict(image_bgr)

        heatmap = out["heatmap"]
        h, w = image_bgr.shape[:2]
        if heatmap is not None and tuple(heatmap.shape[:2]) != (h, w):
            heatmap = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        overlay = overlay_heatmap(image_bgr, heatmap)
        mask = threshold_mask(heatmap, mode=mask_mode, manual_threshold=manual_thr) if heatmap is not None else None
        return {
            "label": out["label"],
            "score": float(out["score"]),
            "overlay_bgr": overlay,
            "mask_uint8": mask,
            "checkpoint_used": chosen_category if method == "patchcore" else None,
            "patchcore_checkpoints_available": trained if method == "patchcore" else None,
        }

    uploaded_files = st.file_uploader(
        "Upload image(s)",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload one or more images to run anomaly detection.")
        return

    if len(uploaded_files) == 1:
        uploaded = uploaded_files[0]
        colA, colB = st.columns([1, 1])

        try:
            bgr = _read_image(uploaded)
        except Exception as e:
            st.error(str(e))
            return

        with st.spinner("Running inference..."):
            res = run_inference(bgr)

        with colA:
            st.subheader("Input")
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        with colB:
            st.subheader("Result")
            payload = {"label": res["label"], "score": float(res["score"])}
            if method == "patchcore" and res.get("checkpoint_used"):
                payload["checkpoint_used"] = res["checkpoint_used"]
            st.write(payload)
            if method == "patchcore":
                avail = res.get("patchcore_checkpoints_available") or []
                if len(avail) <= 2:
                    st.warning(
                        "PatchCore Auto can only choose among trained checkpoints in outputs/patchcore/. "
                        f"Currently available: {', '.join(avail) if avail else '(none)'}.\n\n"
                        "If this image is a different object (e.g. capsule), train that category first."
                    )
                    st.code(
                        "\n".join(
                            [
                                r".\.venv\Scripts\Activate.ps1",
                                r"python tools\train_patchcore.py --config configs\patchcore_mvtec.yaml --category capsule",
                            ]
                        )
                    )
            st.image(cv2.cvtColor(res["overlay_bgr"], cv2.COLOR_BGR2RGB), use_container_width=True)

            if res.get("mask_uint8") is not None:
                st.subheader("Binary mask (optional)")
                st.image(res["mask_uint8"], use_container_width=True)

        return

    st.subheader(f"Batch inference ({len(uploaded_files)} images)")
    results_zip = io.BytesIO()
    with zipfile.ZipFile(results_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        prog = st.progress(0)
        for i, uf in enumerate(uploaded_files, start=1):
            try:
                bgr = _read_image(uf)
                res = run_inference(bgr)
                ok, buf = cv2.imencode(".png", res["overlay_bgr"])
                if ok:
                    zf.writestr(f"{Path(uf.name).stem}_overlay.png", buf.tobytes())
            except Exception as e:
                zf.writestr(f"{Path(uf.name).stem}_error.txt", str(e))
            prog.progress(int(i / len(uploaded_files) * 100))

    results_zip.seek(0)
    st.download_button(
        label="Download overlays (ZIP)",
        data=results_zip,
        file_name=f"surfaceguard_{method}_{category}_overlays.zip",
        mime="application/zip",
    )


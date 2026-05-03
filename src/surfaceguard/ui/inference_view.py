from __future__ import annotations

import io
import hashlib
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import streamlit as st

from surfaceguard.eval.visualize import overlay_heatmap, threshold_mask
from surfaceguard.ui.model_cache import load_patchcore_model, load_winclip_model


def _decode_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")
    return bgr


def _read_image(uploaded) -> np.ndarray:
    # getvalue() is stable across Streamlit reruns; read() can consume the buffer.
    data = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
    return _decode_image_bytes(data)


def _trained_patchcore_categories(outputs_dir: str) -> list[str]:
    p = Path(outputs_dir) / "patchcore"
    if not p.exists():
        return []
    return sorted([d.name for d in p.iterdir() if d.is_dir()])


def _minmax_norm(x: np.ndarray | None) -> np.ndarray | None:
    if x is None:
        return None
    x = x.astype(np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def render_inference_page(title: str = "Inference") -> None:
    st.header(title)

    with st.sidebar:
        st.subheader("Settings")
        is_hf_space = bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID"))
        default_methods = ["patchcore"] if is_hf_space else ["patchcore", "winclip"]
        selected_methods = st.multiselect(
            "Methods (side-by-side)",
            ["patchcore", "winclip", "hybrid"],
            default=default_methods,
        )
        if not selected_methods:
            st.warning("Select at least one method.")
            return

        outputs_dir = st.session_state.get("outputs_dir", "outputs")
        patchcore_category = "auto"
        winclip_prompt_hint = "auto"
        enable_int8 = st.checkbox("Enable INT8 inference (CPU)", value=False)
        with st.expander("Advanced", expanded=False):
            outputs_dir = st.text_input("Outputs dir", value=outputs_dir)
            trained_for_advanced = _trained_patchcore_categories(outputs_dir)
            if ("patchcore" in selected_methods or "hybrid" in selected_methods) and trained_for_advanced:
                patchcore_category = st.selectbox("PatchCore category", ["auto"] + trained_for_advanced, index=0)
            if "winclip" in selected_methods or "hybrid" in selected_methods:
                winclip_prompt_hint = st.text_input("WinCLIP prompt hint (optional)", value="auto")
        st.session_state["outputs_dir"] = outputs_dir

        trained = _trained_patchcore_categories(outputs_dir)
        if "patchcore" in selected_methods or "hybrid" in selected_methods:
            if trained:
                st.caption("PatchCore reads checkpoints from outputs/patchcore/<category>/.")
            else:
                st.warning(
                    "No PatchCore checkpoints found under outputs/patchcore/. Train first or configure Hugging Face Hub sync."
                )

        if "winclip" in selected_methods or "hybrid" in selected_methods:
            st.caption("WinCLIP is training-free; optional prompt hint defaults to generic object prompts.")
            quality = st.selectbox("WinCLIP localization quality", ["fast", "balanced", "best"], index=1)
            if quality == "fast":
                win_stride, blur_sigma = 112, 0.0
            elif quality == "balanced":
                win_stride, blur_sigma = 56, 3.0
            else:
                win_stride, blur_sigma = 28, 5.0
        else:
            win_stride, blur_sigma = 56, 3.0

        if "hybrid" in selected_methods:
            st.subheader("Hybrid")
            hybrid_alpha = st.slider("PatchCore weight", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
            hybrid_threshold = st.slider(
                "Hybrid defect threshold (0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01
            )
        else:
            hybrid_alpha = 0.6
            hybrid_threshold = 0.5

        st.subheader("Mask")
        mask_mode = st.selectbox("Mask threshold", ["p99", "p95", "otsu", "manual"], index=0)
        manual_thr = None
        if mask_mode == "manual":
            manual_thr = st.slider("Manual threshold (0-1)", min_value=0.0, max_value=1.0, value=0.85, step=0.01)

    patchcore_threshold_cache: Dict[str, float] = {}

    def _run_patchcore(image_bgr: np.ndarray) -> Dict[str, Any]:
        chosen_category = patchcore_category
        if chosen_category == "auto":
            if not trained:
                raise FileNotFoundError("No PatchCore checkpoints found under outputs/patchcore/")
            best_score = None
            best_out = None
            best_cat = None
            for cat in trained:
                m = load_patchcore_model(outputs_dir, cat, enable_int8=enable_int8)
                o = m.predict(image_bgr)
                s = float(o["score"])
                if best_score is None or s < best_score:
                    best_score, best_out, best_cat = s, o, cat
            if best_out is None or best_cat is None:
                raise RuntimeError("PatchCore auto category selection failed.")
            out = best_out
            chosen_category = str(best_cat)
        else:
            model = load_patchcore_model(outputs_dir, chosen_category, enable_int8=enable_int8)
            out = model.predict(image_bgr)

        if chosen_category not in patchcore_threshold_cache:
            model = load_patchcore_model(outputs_dir, chosen_category, enable_int8=enable_int8)
            patchcore_threshold_cache[chosen_category] = float(model.engine.state.threshold)
        threshold = patchcore_threshold_cache[chosen_category]

        heatmap = out["heatmap"]
        h, w = image_bgr.shape[:2]
        if heatmap is not None and tuple(heatmap.shape[:2]) != (h, w):
            heatmap = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        score = float(out["score"])
        return {
            "method": "patchcore",
            "label": "defective" if score >= threshold else "normal",
            "score": score,
            "heatmap": heatmap,
            "checkpoint_used": chosen_category,
        }

    def _run_winclip(image_bgr: np.ndarray) -> Dict[str, Any]:
        model = load_winclip_model(
            outputs_dir,
            winclip_prompt_hint,
            window_size=224,
            window_stride=win_stride,
            blur_sigma=blur_sigma,
            enable_int8=enable_int8,
        )
        out = model.predict(image_bgr)
        heatmap = out["heatmap"]
        h, w = image_bgr.shape[:2]
        if heatmap is not None and tuple(heatmap.shape[:2]) != (h, w):
            heatmap = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        return {
            "method": "winclip",
            "label": out["label"],
            "score": float(out["score"]),
            "heatmap": heatmap,
            "checkpoint_used": None,
        }

    def _run_hybrid(image_bgr: np.ndarray) -> Dict[str, Any]:
        pc = _run_patchcore(image_bgr)
        wc = _run_winclip(image_bgr)
        pc_h = _minmax_norm(pc["heatmap"])
        wc_h = _minmax_norm(wc["heatmap"])
        if pc_h is None and wc_h is None:
            raise RuntimeError("Hybrid requires base model heatmaps.")
        if pc_h is None:
            comb = wc_h
        elif wc_h is None:
            comb = pc_h
        else:
            comb = (hybrid_alpha * pc_h + (1.0 - hybrid_alpha) * wc_h).astype(np.float32)
        score = float(np.max(comb)) if comb is not None else 0.0
        return {
            "method": "hybrid",
            "label": "defective" if score >= hybrid_threshold else "normal",
            "score": score,
            "heatmap": comb,
            "checkpoint_used": pc.get("checkpoint_used"),
            "components": {"patchcore_score": pc["score"], "winclip_score": wc["score"]},
        }

    def _run_method(method_name: str, image_bgr: np.ndarray) -> Dict[str, Any]:
        if method_name == "patchcore":
            res = _run_patchcore(image_bgr)
        elif method_name == "winclip":
            res = _run_winclip(image_bgr)
        elif method_name == "hybrid":
            res = _run_hybrid(image_bgr)
        else:
            raise ValueError(f"Unsupported method {method_name}")
        overlay = overlay_heatmap(image_bgr, res["heatmap"])
        mask = threshold_mask(res["heatmap"], mode=mask_mode, manual_threshold=manual_thr) if res["heatmap"] is not None else None
        return {**res, "overlay_bgr": overlay, "mask_uint8": mask}

    uploaded_files_widget = st.file_uploader(
        "Upload image(s)",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        key="main_image_uploader",
    )

    uploaded_entries: List[Dict[str, Any]] = []
    if uploaded_files_widget:
        for uf in uploaded_files_widget:
            data = uf.getvalue() if hasattr(uf, "getvalue") else uf.read()
            if data:
                uploaded_entries.append({"name": uf.name, "bytes": data})
        st.session_state["uploaded_entries"] = uploaded_entries
    else:
        uploaded_entries = st.session_state.get("uploaded_entries", [])

    if not uploaded_entries:
        st.info("Upload one or more images to run anomaly detection.")
        return

    if len(uploaded_entries) == 1:
        uploaded = uploaded_entries[0]
        try:
            bgr = _decode_image_bytes(uploaded["bytes"])
        except Exception as e:
            st.error(str(e))
            return

        file_bytes = uploaded.get("bytes", b"")
        cache_key = (
            hashlib.md5(file_bytes).hexdigest() if file_bytes else uploaded.get("name", ""),
            tuple(sorted(selected_methods)),
            patchcore_category,
            winclip_prompt_hint,
            win_stride,
            blur_sigma,
            hybrid_alpha,
            hybrid_threshold,
            mask_mode,
            manual_thr,
            enable_int8,
        )
        run_single = st.button("Run inference", type="primary", key="run_single_inference")
        single_cache: Dict[Any, Any] = st.session_state.setdefault("single_infer_cache", {})
        cached_payload = single_cache.get(cache_key)
        if run_single:
            with st.spinner("Running inference..."):
                results = []
                method_errors = {}
                for method_name in selected_methods:
                    try:
                        results.append(_run_method(method_name, bgr))
                    except Exception as e:
                        method_errors[method_name] = str(e)
            single_cache[cache_key] = {"results": results, "method_errors": method_errors}
            cached_payload = single_cache.get(cache_key)

        if cached_payload is None:
            st.info("Click **Run inference** to generate results.")
            st.subheader("Input")
            input_cols = st.columns([1, 2, 1])
            with input_cols[1]:
                st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), width="stretch")
            return

        results = cached_payload["results"]
        method_errors = cached_payload["method_errors"]

        if method_errors:
            st.warning("Some selected methods failed. Showing available results.")
            for m, err in method_errors.items():
                st.caption(f"{m}: {err}")
        if not results:
            st.error("No method produced a result. Try selecting PatchCore only or configure HF_TOKEN.")
            return

        st.subheader("Input")
        input_cols = st.columns([1, 2, 1])
        with input_cols[1]:
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), width="stretch")

        st.subheader("Results (side-by-side)")
        method_cols = st.columns(len(results))
        for idx, res in enumerate(results):
            with method_cols[idx]:
                st.markdown(f"**{str(res['method']).title()}**")

                prediction = str(res["label"]).title()
                score_text = f"{float(res['score']):.4f}"
                pred_color = "#ef4444" if str(res["label"]).lower() == "defective" else "#22c55e"
                st.markdown(
                    f"Prediction: <span style='color:{pred_color}; font-weight:700;'>{prediction}</span> | "
                    f"Score: <b>{score_text}</b>",
                    unsafe_allow_html=True,
                )

                st.image(cv2.cvtColor(res["overlay_bgr"], cv2.COLOR_BGR2RGB), width="stretch")
                if res.get("mask_uint8") is not None:
                    st.caption("Binary mask")
                    st.image(res["mask_uint8"], width="stretch")
        return

    st.subheader(f"Batch inference ({len(uploaded_entries)} images)")
    decoded_images: List[np.ndarray | None] = []
    image_names: List[str] = []
    for uf in uploaded_entries:
        image_names.append(str(uf.get("name", "upload")))
        try:
            decoded_images.append(_decode_image_bytes(uf["bytes"]))
        except Exception:
            decoded_images.append(None)

    # Fast path for scalable batch inference: one PatchCore forward pass over many images.
    patchcore_batch_fast: Dict[int, Dict[str, Any]] = {}
    if "patchcore" in selected_methods and patchcore_category != "auto":
        valid_idx = [i for i, img in enumerate(decoded_images) if img is not None]
        if valid_idx:
            model = load_patchcore_model(outputs_dir, patchcore_category, enable_int8=enable_int8)
            batch_in = [decoded_images[i] for i in valid_idx if decoded_images[i] is not None]
            preds = model.predict_batch(batch_in)  # type: ignore[arg-type]
            for idx, pred in zip(valid_idx, preds):
                patchcore_batch_fast[idx] = {
                    "method": "patchcore",
                    "label": pred["label"],
                    "score": float(pred["score"]),
                    "heatmap": pred["heatmap"],
                    "checkpoint_used": patchcore_category,
                }

    batch_sig = hashlib.md5(
        b"||".join(
            [e.get("bytes", b"") for e in uploaded_entries]
            + [str(x).encode("utf-8") for x in sorted(selected_methods)]
            + [
                str(patchcore_category).encode("utf-8"),
                str(winclip_prompt_hint).encode("utf-8"),
                str(win_stride).encode("utf-8"),
                str(blur_sigma).encode("utf-8"),
                str(mask_mode).encode("utf-8"),
                str(manual_thr).encode("utf-8"),
                str(enable_int8).encode("utf-8"),
            ]
        )
    ).hexdigest()

    run_batch = st.button("Run batch inference", type="primary", key="run_batch_inference")
    batch_cache: Dict[str, bytes] = st.session_state.setdefault("batch_zip_cache", {})
    if run_batch:
        results_zip = io.BytesIO()
        with zipfile.ZipFile(results_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            prog = st.progress(0)
            for i, (img, img_name) in enumerate(zip(decoded_images, image_names), start=1):
                try:
                    if img is None:
                        raise ValueError("Could not decode image")
                    for method_name in selected_methods:
                        try:
                            if method_name == "patchcore" and (i - 1) in patchcore_batch_fast:
                                base = patchcore_batch_fast[i - 1]
                                overlay = overlay_heatmap(img, base["heatmap"])
                                res = {**base, "overlay_bgr": overlay}
                            else:
                                res = _run_method(method_name, img)
                            ok, buf = cv2.imencode(".png", res["overlay_bgr"])
                            if ok:
                                zf.writestr(f"{method_name}/{Path(img_name).stem}_overlay.png", buf.tobytes())
                        except Exception as e:
                            zf.writestr(f"{method_name}/{Path(img_name).stem}_error.txt", str(e))
                except Exception as e:
                    zf.writestr(f"{Path(img_name).stem}_error.txt", str(e))
                prog.progress(int(i / len(uploaded_entries) * 100))
        batch_cache[batch_sig] = results_zip.getvalue()

    if batch_sig in batch_cache:
        st.download_button(
            label="Download overlays (ZIP)",
            data=batch_cache[batch_sig],
            file_name="surfaceguard_compare_overlays.zip",
            mime="application/zip",
        )
    else:
        st.info("Click **Run batch inference** to generate overlays ZIP.")

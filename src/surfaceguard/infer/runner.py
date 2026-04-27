from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from surfaceguard.models.patchcore.api import PatchCoreModel
from surfaceguard.models.winclip.api import WinCLIPModel
from surfaceguard.eval.visualize import overlay_heatmap, threshold_mask


def _load_patchcore(outputs_dir: Path, category: str) -> PatchCoreModel:
    ckpt_dir = outputs_dir / "patchcore" / category
    return PatchCoreModel.load(ckpt_dir)


def _load_winclip(outputs_dir: Path, category: str) -> WinCLIPModel:
    return WinCLIPModel.load(outputs_dir / "winclip", category=category)


def infer_image_array(method: str, category: str, image_bgr: np.ndarray, outputs_dir: Path) -> Dict:
    if method == "patchcore":
        model = _load_patchcore(outputs_dir, category)
        out = model.predict(image_bgr)
    elif method == "winclip":
        model = _load_winclip(outputs_dir, category)
        out = model.predict(image_bgr)
    else:
        raise ValueError(f"Unknown method: {method}")

    heatmap = out["heatmap"]
    # Models may produce fixed-size heatmaps (e.g. 256x256). Resize to input image for UI/overlay.
    h, w = image_bgr.shape[:2]
    if heatmap is not None and tuple(heatmap.shape[:2]) != (h, w):
        heatmap = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    overlay = overlay_heatmap(image_bgr, heatmap)
    mask = None
    if out.get("heatmap") is not None:
        mask = threshold_mask(heatmap, mode="otsu")

    return {
        "label": out["label"],
        "score": float(out["score"]),
        "heatmap": heatmap,
        "overlay_bgr": overlay,
        "mask_uint8": mask,
    }


def infer_single_image(method: str, category: str, image_path: Path, outputs_dir: Path) -> Dict:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return infer_image_array(method=method, category=category, image_bgr=bgr, outputs_dir=outputs_dir)


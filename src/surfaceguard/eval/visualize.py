from __future__ import annotations

from typing import Literal, Optional

import cv2
import numpy as np


def normalize_heatmap(hm: np.ndarray) -> np.ndarray:
    hm = hm.astype(np.float32)
    mn = float(np.min(hm))
    mx = float(np.max(hm))
    if mx - mn < 1e-8:
        return np.zeros_like(hm, dtype=np.float32)
    return (hm - mn) / (mx - mn)


def overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    hm = normalize_heatmap(heatmap)
    hm_u8 = (hm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, colored, alpha, 0.0)
    return overlay


def threshold_mask(
    heatmap: np.ndarray,
    mode: Literal["otsu", "p95", "p99", "manual"] = "otsu",
    manual_threshold: Optional[float] = None,
) -> np.ndarray:
    hm = normalize_heatmap(heatmap)
    hm_u8 = (hm * 255).astype(np.uint8)
    if mode == "otsu":
        _, mask = cv2.threshold(hm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask
    if mode == "p95":
        thr = np.percentile(hm_u8, 95)
        return (hm_u8 >= thr).astype(np.uint8) * 255
    if mode == "p99":
        thr = np.percentile(hm_u8, 99)
        return (hm_u8 >= thr).astype(np.uint8) * 255
    if mode == "manual":
        t = 0.5 if manual_threshold is None else float(manual_threshold)
        thr = int(np.clip(round(t * 255.0), 0, 255))
        return (hm_u8 >= thr).astype(np.uint8) * 255
    raise ValueError(mode)


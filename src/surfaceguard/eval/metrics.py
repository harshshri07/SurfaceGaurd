from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import auc, f1_score, roc_auc_score, roc_curve
from skimage.measure import label as cc_label


def image_level_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def pixel_level_auroc(masks: List[np.ndarray], heatmaps: List[np.ndarray]) -> float:
    y_true = np.concatenate([m.reshape(-1) for m in masks]).astype(np.uint8)
    y_score = np.concatenate([h.reshape(-1) for h in heatmaps]).astype(np.float32)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def best_f1_at_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    best = {"f1": -1.0, "threshold": 0.0}
    for t in thr:
        y_pred = (y_score >= t).astype(np.uint8)
        f1 = float(f1_score(y_true, y_pred))
        if f1 > best["f1"]:
            best = {"f1": f1, "threshold": float(t)}
    return best


def _pro_for_threshold(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Per-region overlap (PRO): average IoU-like overlap over connected GT regions.
    Here: overlap = |pred ∩ region| / |region|
    """
    gt_lbl = cc_label(gt.astype(np.uint8), connectivity=1)
    region_ids = np.unique(gt_lbl)
    region_ids = region_ids[region_ids != 0]
    if region_ids.size == 0:
        return 0.0
    pros = []
    for rid in region_ids:
        region = (gt_lbl == rid)
        pros.append(float((pred & region).sum() / max(region.sum(), 1)))
    return float(np.mean(pros)) if pros else 0.0


def au_pro(
    masks: List[np.ndarray],
    heatmaps: List[np.ndarray],
    max_fpr: float = 0.3,
    num_thresholds: int = 200,
) -> float:
    """
    Approximate AU-PRO used in industrial anomaly localization.
    Computes PRO vs FPR curve by thresholding heatmaps.
    """
    assert len(masks) == len(heatmaps)
    # normalize heatmaps to [0,1] jointly
    all_scores = np.concatenate([h.reshape(-1) for h in heatmaps]).astype(np.float32)
    mn, mx = float(all_scores.min()), float(all_scores.max())
    if mx - mn < 1e-8:
        return 0.0
    heatmaps_n = [(h - mn) / (mx - mn) for h in heatmaps]

    thrs = np.linspace(0.0, 1.0, num_thresholds, dtype=np.float32)
    pro_vals = []
    fpr_vals = []

    for t in thrs:
        pros = []
        fp_area = 0.0
        neg_area = 0.0
        for gt, hm in zip(masks, heatmaps_n):
            gt = (gt > 0.5).astype(np.uint8)
            pred = (hm >= t).astype(np.uint8)
            pros.append(_pro_for_threshold(gt, pred))
            neg = (gt == 0)
            fp_area += float((pred.astype(bool) & neg).sum())
            neg_area += float(neg.sum())
        fpr = fp_area / max(neg_area, 1.0)
        pro_vals.append(float(np.mean(pros)))
        fpr_vals.append(float(fpr))

    # sort by fpr, integrate up to max_fpr
    fpr_arr = np.array(fpr_vals, dtype=np.float32)
    pro_arr = np.array(pro_vals, dtype=np.float32)
    order = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[order]
    pro_arr = pro_arr[order]

    # clip to max_fpr
    mask = fpr_arr <= max_fpr
    if mask.sum() < 2:
        return 0.0
    fpr_clip = fpr_arr[mask]
    pro_clip = pro_arr[mask]
    area = auc(fpr_clip, pro_clip) / max_fpr
    return float(area)


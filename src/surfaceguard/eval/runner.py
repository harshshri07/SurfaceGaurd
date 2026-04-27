from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from surfaceguard.data.mvtec import MVTecADDataset
from surfaceguard.data.transforms import build_image_transform, build_mask_transform
from surfaceguard.eval.metrics import au_pro, best_f1_at_threshold, image_level_auroc, pixel_level_auroc
from surfaceguard.models.patchcore.api import PatchCoreModel
from surfaceguard.models.winclip.api import WinCLIPModel
from surfaceguard.utils.io import ensure_dir


def _load_model(method: str, outputs_dir: Path, category: str):
    if method == "patchcore":
        return PatchCoreModel.load(outputs_dir / "patchcore" / category)
    if method == "winclip":
        return WinCLIPModel.load(outputs_dir / "winclip", category=category)
    raise ValueError(method)


def eval_method_for_category(cfg: Dict[str, Any], method: str, category: str) -> str:
    data_cfg = cfg["data"]
    outputs_dir = Path(cfg.get("outputs", {}).get("dir", "outputs"))
    out_dir = ensure_dir(outputs_dir / "reports")

    image_size = int(data_cfg.get("image_size", 256))
    t_img = build_image_transform(image_size, normalize=data_cfg.get("normalize", "imagenet"))
    t_mask = build_mask_transform(image_size)

    ds = MVTecADDataset(
        root=Path(data_cfg["root"]),
        category=category,
        split="test",
        image_transform=t_img,
        mask_transform=t_mask,
    )

    model = _load_model(method, outputs_dir=outputs_dir, category=category)

    y_true: List[int] = []
    y_score: List[float] = []
    heatmaps: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    for item in tqdm(ds, desc=f"eval {method}:{category}"):
        img_path = item["image_path"]
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        pred = model.predict(bgr)
        y_true.append(int(item["label"]))
        y_score.append(float(pred["score"]))

        if int(item["label"]) == 1 and bool(item.get("has_mask", True)):
            m = item["mask"].squeeze().numpy().astype(np.uint8)
            masks.append(m)
            hm = pred["heatmap"].astype(np.float32)
            if hm.shape != m.shape:
                hm = cv2.resize(hm, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            heatmaps.append(hm)

    y_true_arr = np.array(y_true, dtype=np.uint8)
    y_score_arr = np.array(y_score, dtype=np.float32)

    report: Dict[str, Any] = {
        "method": method,
        "category": category,
        "num_test": int(len(y_true)),
        "image_auroc": image_level_auroc(y_true_arr, y_score_arr),
        "best_f1": best_f1_at_threshold(y_true_arr, y_score_arr),
    }

    if masks:
        report.update(
            {
                "pixel_auroc": pixel_level_auroc(masks, heatmaps),
                "au_pro": au_pro(masks, heatmaps, max_fpr=0.3),
                "num_pixel_gt": int(len(masks)),
            }
        )

    path = out_dir / f"{method}_{category}_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(path)


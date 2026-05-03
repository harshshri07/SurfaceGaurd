from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import cv2
import torch

from surfaceguard.data.transforms import build_image_transform
from surfaceguard.models.patchcore.core import PatchCoreEngine, PatchCoreState
from surfaceguard.utils.device import get_device
from surfaceguard.utils.io import load_npz
from surfaceguard.utils.config import load_yaml


@dataclass
class PatchCoreModel:
    category: str
    engine: PatchCoreEngine

    @classmethod
    def load(cls, ckpt_dir: Path, enable_int8: bool = False) -> "PatchCoreModel":
        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"PatchCore checkpoint not found at {ckpt_dir}. "
                f"Train first: python tools\\train_patchcore.py --config configs\\patchcore_mvtec.yaml --category <category>"
            )
        meta = load_yaml(ckpt_dir / "meta.yaml")
        mem = load_npz(ckpt_dir / "memory.npz")["memory"].astype(np.float32)
        state = PatchCoreState(
            backbone=meta["backbone"],
            layer=meta["layer"],
            image_size=int(meta["image_size"]),
            nn_k=int(meta["nn_k"]),
            image_score=meta["image_score"],
            topk=int(meta["topk"]),
            threshold=float(meta["threshold"]),
            memory=mem,
        )
        device = get_device()
        engine = PatchCoreEngine(state, device=device, enable_int8=enable_int8)
        return cls(category=meta.get("category", ckpt_dir.name), engine=engine)

    def predict(self, image_bgr: np.ndarray) -> Dict:
        return self.predict_batch([image_bgr])[0]

    def predict_batch(self, images_bgr: List[np.ndarray]) -> List[Dict]:
        if not images_bgr:
            return []
        t = build_image_transform(self.engine.state.image_size, normalize="imagenet")
        import PIL.Image

        tensor_list = []
        for image_bgr in images_bgr:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tensor_list.append(t(PIL.Image.fromarray(rgb)))
        batch = torch.stack(tensor_list, dim=0)
        scores, heats = self.engine.batch_score(batch)
        out: List[Dict] = []
        thr = float(self.engine.state.threshold)
        for score, heat in zip(scores, heats):
            f_score = float(score)
            out.append(
                {
                    "label": "defective" if f_score >= thr else "normal",
                    "score": f_score,
                    "heatmap": heat,
                }
            )
        return out


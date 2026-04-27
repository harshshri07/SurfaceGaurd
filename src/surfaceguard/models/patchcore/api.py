from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import cv2

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
    def load(cls, ckpt_dir: Path) -> "PatchCoreModel":
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
        engine = PatchCoreEngine(state, device=device)
        return cls(category=meta.get("category", ckpt_dir.name), engine=engine)

    def predict(self, image_bgr: np.ndarray) -> Dict:
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        t = build_image_transform(self.engine.state.image_size, normalize="imagenet")
        import PIL.Image

        x = t(PIL.Image.fromarray(img)).unsqueeze(0)
        score, heat = self.engine.score(x)
        label = "defective" if score >= self.engine.state.threshold else "normal"
        return {"label": label, "score": float(score), "heatmap": heat}


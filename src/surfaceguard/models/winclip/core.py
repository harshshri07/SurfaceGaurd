from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch

import open_clip


def _l2(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


@dataclass
class WinCLIPState:
    model: str
    pretrained: str
    window_size: int
    window_stride: int
    normal_prompts: List[str]
    anomalous_prompts: List[str]
    image_size: int
    blur_sigma: float = 0.0


class WinCLIPEngine:
    def __init__(self, state: WinCLIPState, device: torch.device, category: str) -> None:
        self.state = state
        self.device = device
        self.category = category

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            state.model, pretrained=state.pretrained
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(state.model)

        self.text_normal = self._encode_text([p.format(category=category) for p in state.normal_prompts])
        self.text_anom = self._encode_text([p.format(category=category) for p in state.anomalous_prompts])

    @torch.no_grad()
    def _encode_text(self, prompts: List[str]) -> torch.Tensor:
        toks = self.tokenizer(prompts)
        toks = toks.to(self.device)
        t = self.model.encode_text(toks)
        return _l2(t)

    @torch.no_grad()
    def _encode_image(self, rgb_uint8: np.ndarray) -> torch.Tensor:
        import PIL.Image

        pil = PIL.Image.fromarray(rgb_uint8)
        x = self.preprocess(pil).unsqueeze(0).to(self.device)
        emb = self.model.encode_image(x)
        return _l2(emb)

    @torch.no_grad()
    def score(self, image_bgr: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Returns (image_score, heatmap_float32 [H,W]).

        We use a simple and stable scoring function:
        window_score = mean(sim_anomalous_prompts) - mean(sim_normal_prompts).
        This yields a heatmap with meaningful dynamic range, unlike a softmax that can collapse near 0.5.
        """
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        win = int(self.state.window_size)
        stride = int(self.state.window_stride)
        win = min(win, h, w)
        stride = max(1, min(stride, win))

        acc = np.zeros((h, w), dtype=np.float32)
        cnt = np.zeros((h, w), dtype=np.float32)

        def window_score(r0: int, c0: int, patch_rgb: np.ndarray) -> float:
            emb = self._encode_image(patch_rgb)  # [1,D]
            # similarity to prompt sets
            sim_n = (emb @ self.text_normal.T).mean(dim=1)  # [1]
            sim_a = (emb @ self.text_anom.T).mean(dim=1)    # [1]
            return float((sim_a - sim_n).item())

        # Ensure full coverage by also including the last row/col window positions.
        row_starts = list(range(0, h - win + 1, stride))
        col_starts = list(range(0, w - win + 1, stride))
        if not row_starts:
            row_starts = [0]
        if not col_starts:
            col_starts = [0]
        last_r = max(0, h - win)
        last_c = max(0, w - win)
        if row_starts[-1] != last_r:
            row_starts.append(last_r)
        if col_starts[-1] != last_c:
            col_starts.append(last_c)

        for r in row_starts:
            for c in col_starts:
                patch = rgb[r : r + win, c : c + win]
                if patch.shape[0] != win or patch.shape[1] != win:
                    continue
                s = window_score(r, c, patch)
                acc[r : r + win, c : c + win] += s
                cnt[r : r + win, c : c + win] += 1.0

        heat = acc / np.maximum(cnt, 1.0)
        if float(self.state.blur_sigma) > 0.0:
            heat = cv2.GaussianBlur(heat, ksize=(0, 0), sigmaX=float(self.state.blur_sigma))
        img_score = float(np.max(heat))
        return img_score, heat.astype(np.float32)


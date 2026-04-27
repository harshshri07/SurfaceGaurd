from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torchvision import models as tvm


def _build_backbone(name: str) -> torch.nn.Module:
    if not hasattr(tvm, name):
        raise ValueError(f"Unknown torchvision backbone: {name}")
    fn = getattr(tvm, name)
    try:
        model = fn(weights="DEFAULT")
    except TypeError:
        model = fn(pretrained=True)
    model.eval()
    return model


@torch.no_grad()
def extract_feature_map(model: torch.nn.Module, x: torch.Tensor, layer: str) -> torch.Tensor:
    feats: Dict[str, torch.Tensor] = {}

    def hook(_, __, out):
        feats["f"] = out

    if not hasattr(model, layer):
        raise ValueError(f"Backbone has no layer '{layer}'")
    h = getattr(model, layer).register_forward_hook(hook)
    try:
        _ = model(x)
    finally:
        h.remove()
    return feats["f"]


def feature_map_to_patches(fmap: torch.Tensor) -> torch.Tensor:
    # fmap: [B,C,H,W] -> [B, H*W, C]
    b, c, h, w = fmap.shape
    return fmap.permute(0, 2, 3, 1).reshape(b, h * w, c)


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


def kcenter_greedy_coreset(x: np.ndarray, fraction: float, seed: int = 42) -> np.ndarray:
    """
    k-center greedy coreset selection.
    x: [N,D] float32
    returns indices of selected points
    """
    n = x.shape[0]
    m = max(1, int(n * fraction))
    if m >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    selected = np.empty((m,), dtype=np.int64)
    selected[0] = first

    # initialize distances to first center
    d = np.linalg.norm(x - x[first], axis=1).astype(np.float32)
    for i in range(1, m):
        idx = int(np.argmax(d))
        selected[i] = idx
        d = np.minimum(d, np.linalg.norm(x - x[idx], axis=1).astype(np.float32))
    return selected


def random_coreset(x: np.ndarray, fraction: float, seed: int = 42) -> np.ndarray:
    n = x.shape[0]
    m = max(1, int(n * fraction))
    if m >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=m, replace=False).astype(np.int64)


@dataclass
class PatchCoreState:
    backbone: str
    layer: str
    image_size: int
    nn_k: int
    image_score: str
    topk: int
    threshold: float
    memory: np.ndarray  # [M,D]


class PatchCoreEngine:
    def __init__(self, state: PatchCoreState, device: torch.device) -> None:
        self.state = state
        self.device = device
        self.backbone = _build_backbone(state.backbone).to(device)
        self.nn = NearestNeighbors(n_neighbors=state.nn_k, algorithm="auto")
        self.nn.fit(state.memory)

    @torch.no_grad()
    def score(self, image_tensor: torch.Tensor) -> Tuple[float, np.ndarray]:
        """
        image_tensor: [1,3,H,W] normalized
        returns: (image_score, heatmap_float32 [H,W])
        """
        fmap = extract_feature_map(self.backbone, image_tensor.to(self.device), self.state.layer)
        patches = feature_map_to_patches(fmap)  # [1,P,C]
        patches = l2_normalize(patches, dim=-1)
        p = patches.squeeze(0).detach().cpu().numpy().astype(np.float32)  # [P,D]
        dist, _ = self.nn.kneighbors(p, return_distance=True)  # [P,k]
        patch_scores = dist[:, 0].astype(np.float32)

        # heatmap in feature-map resolution
        _, _, fh, fw = fmap.shape
        heat_feat = patch_scores.reshape(fh, fw)
        # upsample to image resolution
        heat = torch.from_numpy(heat_feat)[None, None, :, :]
        heat_up = F.interpolate(heat, size=(self.state.image_size, self.state.image_size), mode="bilinear", align_corners=False)
        heat_up = heat_up.squeeze().numpy().astype(np.float32)

        if self.state.image_score == "max":
            img_score = float(patch_scores.max())
        else:
            k = min(self.state.topk, patch_scores.shape[0])
            img_score = float(np.mean(np.sort(patch_scores)[-k:]))

        return img_score, heat_up


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torchvision import models as tvm

# One backbone instance per (name, device) when not using per-engine INT8 quantization.
# Avoids loading N copies of Wide ResNet when multiple PatchCore categories are used (e.g. Streamlit "auto").
_SHARED_BACKBONES: Dict[Tuple[str, str], torch.nn.Module] = {}


def _prepare_quant_backend() -> bool:
    supported = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    if not supported:
        return False
    if torch.backends.quantized.engine == "none":
        preferred = "qnnpack" if "qnnpack" in supported else supported[0]
        torch.backends.quantized.engine = preferred
    return torch.backends.quantized.engine in supported


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


def _get_shared_backbone(backbone_name: str, device: torch.device, enable_int8: bool) -> Tuple[torch.nn.Module, bool]:
    """
    INT8 path builds a dedicated backbone (quantization mutates weights; do not share).
    FP32 path shares one torchvision backbone across PatchCore engines on the same device.
    Returns (module, whether dynamic INT8 is active).
    """
    use_quant = bool(enable_int8 and device.type == "cpu")
    if use_quant:
        bb = _build_backbone(backbone_name).to(device)
        bb.eval()
        try:
            if _prepare_quant_backend():
                bb = torch.ao.quantization.quantize_dynamic(bb, {torch.nn.Linear}, dtype=torch.qint8)
                return bb, True
        except Exception:
            pass
        return bb, False
    key = (backbone_name, str(device))
    if key not in _SHARED_BACKBONES:
        m = _build_backbone(backbone_name).to(device)
        m.eval()
        _SHARED_BACKBONES[key] = m
    return _SHARED_BACKBONES[key], False


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
    def __init__(self, state: PatchCoreState, device: torch.device, enable_int8: bool = False) -> None:
        self.state = state
        self.device = device
        self.backbone, self.enable_int8 = _get_shared_backbone(state.backbone, device, enable_int8)
        self.nn = NearestNeighbors(n_neighbors=state.nn_k, algorithm="auto")
        self.nn.fit(state.memory)

    @torch.no_grad()
    def batch_score(self, image_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        image_tensor: [B,3,H,W] normalized
        returns: (image_scores[B], heatmaps[B,H,W])
        """
        fmap = extract_feature_map(self.backbone, image_tensor.to(self.device), self.state.layer)
        patches = feature_map_to_patches(fmap)  # [B,P,C]
        patches = l2_normalize(patches, dim=-1)
        p = patches.detach().cpu().numpy().astype(np.float32)  # [B,P,D]
        b, p_count, _ = p.shape
        flat = p.reshape(b * p_count, -1)
        dist, _ = self.nn.kneighbors(flat, return_distance=True)  # [B*P,k]
        patch_scores = dist[:, 0].astype(np.float32).reshape(b, p_count)

        # heatmap in feature-map resolution
        _, _, fh, fw = fmap.shape
        heat_feat = patch_scores.reshape(b, fh, fw)
        # upsample to image resolution
        heat = torch.from_numpy(heat_feat)[:, None, :, :]
        heat_up = F.interpolate(heat, size=(self.state.image_size, self.state.image_size), mode="bilinear", align_corners=False)
        heat_up = heat_up.squeeze(1).numpy().astype(np.float32)

        if self.state.image_score == "max":
            img_score = patch_scores.max(axis=1).astype(np.float32)
        else:
            k = min(self.state.topk, patch_scores.shape[1])
            topk_sorted = np.sort(patch_scores, axis=1)[:, -k:]
            img_score = topk_sorted.mean(axis=1).astype(np.float32)

        return img_score, heat_up

    @torch.no_grad()
    def score(self, image_tensor: torch.Tensor) -> Tuple[float, np.ndarray]:
        """
        image_tensor: [1,3,H,W] normalized
        returns: (image_score, heatmap_float32 [H,W])
        """
        img_scores, heatmaps = self.batch_score(image_tensor)
        return float(img_scores[0]), heatmaps[0]


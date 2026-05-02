from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from surfaceguard.data.mvtec import MVTecADDataset
from surfaceguard.data.transforms import build_image_transform
from surfaceguard.models.patchcore.core import (
    PatchCoreEngine,
    PatchCoreState,
    _build_backbone,
    extract_feature_map,
    feature_map_to_patches,
    kcenter_greedy_coreset,
    random_coreset,
    l2_normalize,
)
from surfaceguard.utils.device import get_device
from surfaceguard.utils.io import ensure_dir, save_npz
from surfaceguard.utils.provenance import get_git_commit_short, stable_cfg_hash
from surfaceguard.utils.seed import seed_everything


def _as_abs_outputs(cfg: Dict[str, Any]) -> Path:
    return Path(cfg.get("outputs", {}).get("patchcore_dir", "outputs/patchcore"))


def train_patchcore_for_category(cfg: Dict[str, Any], category: str) -> str:
    data_cfg = cfg["data"]
    pc_cfg = cfg["patchcore"]
    tr_cfg = cfg.get("train", {})

    root = Path(data_cfg["root"])
    image_size = int(data_cfg.get("image_size", 256))
    backbone = pc_cfg.get("backbone", "wide_resnet50_2")
    layer = pc_cfg.get("layer", "layer3")
    coreset_fraction = float(pc_cfg.get("coreset_fraction", 0.1))
    coreset_method = pc_cfg.get("coreset_method", "auto")  # auto|random|kcenter
    max_patch_samples = int(pc_cfg.get("max_patch_samples", 200000))
    patches_per_image = int(pc_cfg.get("patches_per_image", 2000))
    nn_k = int(pc_cfg.get("nn_k", 1))
    image_score = pc_cfg.get("image_score", "max")
    topk = int(pc_cfg.get("topk", 50))

    seed = int(tr_cfg.get("seed", 42))
    seed_everything(seed)

    device = get_device()
    backbone_model = _build_backbone(backbone).to(device)
    t_img = build_image_transform(image_size, normalize=data_cfg.get("normalize", "imagenet"))

    ds = MVTecADDataset(root=root, category=category, split="train", image_transform=t_img)
    dl = DataLoader(
        ds,
        batch_size=int(tr_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(tr_cfg.get("num_workers", 2)),
        pin_memory=(device.type == "cuda"),
    )

    # 1) Collect patch embeddings (subsample per image to keep training practical)
    patches_all: List[np.ndarray] = []
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(device)  # [B,3,H,W]
            fmap = extract_feature_map(backbone_model, x, layer)  # [B,C,h,w]
            patches = feature_map_to_patches(fmap)  # [B,P,C]
            patches = l2_normalize(patches, dim=-1)
            p = patches.detach().cpu().numpy().astype(np.float32)  # [B,P,D]
            for bi in range(p.shape[0]):
                pb = p[bi]
                if patches_per_image > 0 and patches_per_image < pb.shape[0]:
                    rng = np.random.default_rng(seed + bi)
                    idx = rng.choice(pb.shape[0], size=patches_per_image, replace=False)
                    pb = pb[idx]
                patches_all.append(pb)

    memory_full = np.concatenate(patches_all, axis=0)
    if max_patch_samples > 0 and memory_full.shape[0] > max_patch_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(memory_full.shape[0], size=max_patch_samples, replace=False)
        memory_full = memory_full[idx]

    # 2) Coreset sampling
    # k-center greedy is very slow for large N; default to fast random coreset unless small.
    if coreset_method == "kcenter":
        idx = kcenter_greedy_coreset(memory_full, fraction=coreset_fraction, seed=seed)
        coreset_used = "kcenter"
    elif coreset_method == "random":
        idx = random_coreset(memory_full, fraction=coreset_fraction, seed=seed)
        coreset_used = "random"
    else:
        # auto
        if memory_full.shape[0] <= 50000:
            idx = kcenter_greedy_coreset(memory_full, fraction=coreset_fraction, seed=seed)
            coreset_used = "kcenter"
        else:
            idx = random_coreset(memory_full, fraction=coreset_fraction, seed=seed)
            coreset_used = "random"
    memory = memory_full[idx]

    # 3) Fit engine and set threshold from train scores
    state = PatchCoreState(
        backbone=backbone,
        layer=layer,
        image_size=image_size,
        nn_k=nn_k,
        image_score=image_score,
        topk=topk,
        threshold=0.0,
        memory=memory,
    )
    engine = PatchCoreEngine(state, device=device)

    train_scores: List[float] = []
    with torch.no_grad():
        for batch in dl:
            x = batch["image"]  # keep on CPU; engine moves to device
            for i in range(x.shape[0]):
                s, _ = engine.score(x[i : i + 1])
                train_scores.append(float(s))

    mu = float(np.mean(train_scores)) if train_scores else 0.0
    sig = float(np.std(train_scores)) if train_scores else 1.0
    threshold = mu + 3.0 * sig
    engine.state.threshold = threshold

    # 4) Save checkpoint
    out_dir = ensure_dir(_as_abs_outputs(cfg) / category)
    save_npz(out_dir / "memory.npz", memory=memory.astype(np.float32))
    cfg_hash = stable_cfg_hash(cfg)
    git_commit = get_git_commit_short()
    meta = {
        "category": category,
        "backbone": backbone,
        "layer": layer,
        "image_size": image_size,
        "nn_k": nn_k,
        "image_score": image_score,
        "topk": topk,
        "coreset_fraction": coreset_fraction,
        "coreset_method": coreset_used,
        "max_patch_samples": max_patch_samples,
        "patches_per_image": patches_per_image,
        "threshold": threshold,
        "train_score_mean": mu,
        "train_score_std": sig,
        "num_train_images": len(ds),
        "num_memory_patches": int(memory.shape[0]),
        "patch_dim": int(memory.shape[1]),
        "config_hash": cfg_hash,
        "git_commit": git_commit,
    }

    import yaml

    (out_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    return str(out_dir)


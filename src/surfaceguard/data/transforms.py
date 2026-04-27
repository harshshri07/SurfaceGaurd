from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from torchvision import transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# OpenAI CLIP defaults (also used by many OpenCLIP checkpoints)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_image_transform(
    image_size: int,
    normalize: Literal["imagenet", "clip", "none"] = "imagenet",
) -> T.Compose:
    ops = [
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ]
    if normalize == "imagenet":
        ops.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    elif normalize == "clip":
        ops.append(T.Normalize(CLIP_MEAN, CLIP_STD))
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalize: {normalize}")
    return T.Compose(ops)


def build_mask_transform(image_size: int) -> T.Compose:
    # masks are loaded as single-channel PIL images (0/255), keep as 0/1 float tensor
    return T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ]
    )


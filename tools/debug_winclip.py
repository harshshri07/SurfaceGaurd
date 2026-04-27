from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from surfaceguard.models.winclip.api import WinCLIPModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, type=str)
    p.add_argument("--category", default="bottle", type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.image)
    m = WinCLIPModel.load(Path("outputs") / "winclip", category=args.category)
    out = m.predict(bgr)
    hm = out["heatmap"]
    print("label", out["label"])
    print("score", float(out["score"]))
    print("heatmap min/max/mean", float(hm.min()), float(hm.max()), float(hm.mean()))
    print("heatmap shape", hm.shape, "image shape", bgr.shape[:2])


if __name__ == "__main__":
    main()


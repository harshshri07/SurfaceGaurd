from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from surfaceguard.infer.runner import infer_single_image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["patchcore", "winclip"])
    p.add_argument("--category", required=True, type=str)
    p.add_argument("--image", required=True, type=str)
    p.add_argument("--outputs_dir", default="outputs", type=str)
    p.add_argument("--save_overlay", action="store_true")
    p.add_argument("--overlay_name", default="infer_overlay.png", type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    res = infer_single_image(
        method=args.method,
        category=args.category,
        image_path=Path(args.image),
        outputs_dir=Path(args.outputs_dir),
    )
    print(f"label={res['label']} score={res['score']:.6f}")
    if args.save_overlay:
        overlay = res["overlay_bgr"]
        out_path = Path(args.outputs_dir) / args.overlay_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), overlay)
        print(f"saved={out_path}")


if __name__ == "__main__":
    main()


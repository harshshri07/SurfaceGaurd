from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _ensure(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def make_toy_mvtec(root: Path, category: str, n_train: int = 20, n_test_good: int = 10, n_test_bad: int = 10) -> Path:
    cat = root / category
    train_good = cat / "train" / "good"
    test_good = cat / "test" / "good"
    test_bad = cat / "test" / "scratch"
    gt_bad = cat / "ground_truth" / "scratch"
    for p in [train_good, test_good, test_bad, gt_bad]:
        _ensure(p)

    rng = np.random.default_rng(0)

    def base_img() -> np.ndarray:
        img = (rng.normal(127, 12, size=(256, 256, 3)).clip(0, 255)).astype(np.uint8)
        # add faint texture
        for _ in range(8):
            x1, y1 = int(rng.integers(0, 256)), int(rng.integers(0, 256))
            x2, y2 = int(rng.integers(0, 256)), int(rng.integers(0, 256))
            cv2.line(img, (x1, y1), (x2, y2), (120, 120, 120), 1)
        return img

    for i in range(n_train):
        img = base_img()
        cv2.imwrite(str(train_good / f"{i:03d}.png"), img)

    for i in range(n_test_good):
        img = base_img()
        cv2.imwrite(str(test_good / f"{i:03d}.png"), img)

    for i in range(n_test_bad):
        img = base_img()
        mask = np.zeros((256, 256), dtype=np.uint8)
        # synthetic scratch (bright line)
        x1, y1 = int(rng.integers(10, 246)), int(rng.integers(10, 246))
        x2, y2 = int(rng.integers(10, 246)), int(rng.integers(10, 246))
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
        cv2.imwrite(str(test_bad / f"{i:03d}.png"), img)
        cv2.imwrite(str(gt_bad / f"{i:03d}_mask.png"), mask)

    return cat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/mvtec_ad", type=str)
    p.add_argument("--category", default="toy", type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = make_toy_mvtec(Path(args.root), args.category)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
from pathlib import Path

from surfaceguard.utils.config import load_yaml
from surfaceguard.utils.seed import seed_everything
from surfaceguard.train.patchcore_trainer import train_patchcore_for_category


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--category", required=True, type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("train", {}).get("seed", 42)))
    train_patchcore_for_category(cfg, category=args.category)


if __name__ == "__main__":
    main()


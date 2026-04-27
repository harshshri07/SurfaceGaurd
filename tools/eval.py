from __future__ import annotations

import argparse

from surfaceguard.utils.config import load_yaml
from surfaceguard.eval.runner import eval_method_for_category


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--method", required=True, choices=["patchcore", "winclip"])
    p.add_argument("--category", required=True, type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    eval_method_for_category(cfg, method=args.method, category=args.category)


if __name__ == "__main__":
    main()


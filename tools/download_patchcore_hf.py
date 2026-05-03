"""CLI: sync full outputs/patchcore tree from a Hugging Face Model repo."""

from __future__ import annotations

import argparse
from pathlib import Path

from surfaceguard.utils.hf_patchcore import sync_patchcore_tree_from_hub


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download patchcore/** from a Hub Model repo into outputs/ (same layout as local training)."
    )
    p.add_argument("--repo-id", required=True, help="Hub model repo id, e.g. username/surfaceguard-weights")
    p.add_argument(
        "--outputs-root",
        default="outputs",
        type=str,
        help="Directory that will contain patchcore/ (default: outputs)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.outputs_root)
    sync_patchcore_tree_from_hub(args.repo_id, root)
    print(f"OK: synced Hub patchcore/ tree under {root.resolve()}")


if __name__ == "__main__":
    main()

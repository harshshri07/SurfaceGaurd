from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from surfaceguard.eval.runner import eval_method_for_category
from surfaceguard.train.patchcore_trainer import train_patchcore_for_category
from surfaceguard.utils.config import load_yaml
from surfaceguard.utils.io import ensure_dir
from surfaceguard.utils.repro import write_run_manifest


DEFAULT_SWEEPS: Dict[str, List[str]] = {
    "backbone": ["wide_resnet50_2:layer3", "resnet18:layer3"],
    "coreset_fraction": ["0.02", "0.05", "0.1"],
    "image_size": ["224", "256", "320"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PatchCore ablation experiments for one category.")
    p.add_argument("--category", required=True, type=str)
    p.add_argument("--base_config", default="configs/patchcore_mvtec.yaml", type=str)
    p.add_argument("--sweep", choices=["backbone", "coreset_fraction", "image_size"], default="backbone")
    p.add_argument("--values", type=str, default="")
    p.add_argument("--out_dir", type=str, default="outputs/reports")
    return p.parse_args()


def _value_list(args: argparse.Namespace) -> List[str]:
    if args.values.strip():
        return [v.strip() for v in args.values.split(",") if v.strip()]
    return DEFAULT_SWEEPS[args.sweep]


def _apply_variant(cfg: Dict[str, Any], sweep: str, value: str, tag: str) -> Dict[str, Any]:
    c = deepcopy(cfg)
    c.setdefault("outputs", {})
    c["outputs"]["dir"] = f"outputs/ablations/{tag}"
    c["outputs"]["patchcore_dir"] = f"outputs/ablations/{tag}/patchcore"

    if sweep == "backbone":
        if ":" in value:
            backbone, layer = value.split(":", 1)
        else:
            backbone, layer = value, c["patchcore"].get("layer", "layer3")
        c["patchcore"]["backbone"] = backbone
        c["patchcore"]["layer"] = layer
    elif sweep == "coreset_fraction":
        c["patchcore"]["coreset_fraction"] = float(value)
    elif sweep == "image_size":
        c["data"]["image_size"] = int(value)
    else:
        raise ValueError(f"Unsupported sweep: {sweep}")
    return c


def _read_report(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.base_config)
    values = _value_list(args)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_prefix = f"{args.sweep}_{args.category}_{stamp}"

    rows: List[Dict[str, Any]] = []
    for i, v in enumerate(values):
        run_tag = f"{tag_prefix}_{i+1}"
        cfg = _apply_variant(base_cfg, args.sweep, v, run_tag)
        print(f"[run] sweep={args.sweep} value={v} category={args.category}")
        train_patchcore_for_category(cfg, category=args.category)
        report_path = eval_method_for_category(cfg, method="patchcore", category=args.category)
        rep = _read_report(report_path)
        rep["variant"] = {"sweep": args.sweep, "value": v, "tag": run_tag}
        rep["report_path"] = report_path
        rows.append(rep)

    out = {
        "kind": "ablation_summary",
        "sweep": args.sweep,
        "category": args.category,
        "values": values,
        "num_runs": len(rows),
        "runs": rows,
    }
    out_dir = ensure_dir(args.out_dir)
    out_path = out_dir / f"ablation_{args.sweep}_{args.category}_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    manifest_path = write_run_manifest(
        output_root=Path(args.out_dir).parent,
        run_name=f"ablation_{args.sweep}",
        args=vars(args),
        config_paths=[args.base_config],
        extra={"summary_path": str(out_path), "values": values},
    )
    print(f"summary={out_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()

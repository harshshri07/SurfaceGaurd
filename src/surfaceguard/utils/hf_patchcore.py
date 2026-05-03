"""Sync PatchCore checkpoints from Hugging Face Hub (full outputs/patchcore tree)."""

from __future__ import annotations

import os
from pathlib import Path


def patchcore_tree_has_checkpoint(outputs_root: Path) -> bool:
    """True if any category folder under outputs/patchcore contains memory.npz."""
    pc = outputs_root / "patchcore"
    if not pc.exists():
        return False
    for d in pc.iterdir():
        if d.is_dir() and (d / "memory.npz").is_file():
            return True
    return False


def sync_patchcore_tree_from_hub(repo_id: str, outputs_root: Path) -> None:
    """
    Download the entire ``patchcore/**`` subtree from a Hub *model* repo into ``outputs_root``.

    Mirror your local layout: ``outputs/patchcore/<category>/memory.npz`` + ``meta.yaml`` for each category.
    Use ``snapshot_download`` so **all** categories are available — same as shipping a full ``outputs/patchcore`` folder,
    which keeps **auto** category selection working after upload.
    """
    from huggingface_hub import snapshot_download

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    revision = os.getenv("HF_REVISION") or os.getenv("SURFACEGUARD_HF_REVISION") or None
    repo_type = os.getenv("SURFACEGUARD_HF_REPO_TYPE", "model")

    outputs_root.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(outputs_root),
        allow_patterns=["patchcore/**"],
        token=token,
        revision=revision,
    )


def ensure_patchcore_tree_from_hub(repo_id: str | None, outputs_root: Path) -> bool:
    """
    If ``repo_id`` is set and no local PatchCore checkpoints exist yet, sync ``patchcore/**`` from the Hub.
    Returns True if at least one checkpoint exists after the call (local or downloaded).
    """
    rid = (repo_id or "").strip()
    if not rid:
        return patchcore_tree_has_checkpoint(outputs_root)
    if patchcore_tree_has_checkpoint(outputs_root):
        return True
    sync_patchcore_tree_from_hub(rid, outputs_root)
    return patchcore_tree_has_checkpoint(outputs_root)


# Backwards-compatible names for imports that still expect the old API:
def ensure_patchcore_from_hub(repo_id: str | None, category: str, outputs_root: Path) -> bool:
    """Deprecated single-category helper; use ``ensure_patchcore_tree_from_hub`` (category ignored)."""
    return ensure_patchcore_tree_from_hub(repo_id, outputs_root)


def download_patchcore_from_hub(repo_id: str, category: str, outputs_root: Path) -> None:
    """Deprecated; sync the full tree instead."""
    sync_patchcore_tree_from_hub(repo_id, outputs_root)

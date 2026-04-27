from __future__ import annotations

from pathlib import Path

import streamlit as st

from surfaceguard.models.patchcore.api import PatchCoreModel
from surfaceguard.models.winclip.api import WinCLIPModel


@st.cache_resource(show_spinner=False)
def load_patchcore_model(outputs_dir: str, category: str) -> PatchCoreModel:
    ckpt = Path(outputs_dir) / "patchcore" / category
    return PatchCoreModel.load(ckpt)


@st.cache_resource(show_spinner=False)
def load_winclip_model(outputs_dir: str, category: str) -> WinCLIPModel:
    cache_dir = Path(outputs_dir) / "winclip"
    return WinCLIPModel.load(cache_dir, category=category)


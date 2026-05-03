#!/usr/bin/env bash
# One-command launcher for graders (bash). Creates venv, installs deps, starts Streamlit.
# Put Hugging Face repo id in configs/app.yaml (hf_patchcore_repo) or set SURFACEGUARD_HF_REPO_ID.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
exec streamlit run app/Home.py --server.address=127.0.0.1 --server.port=8501

---
title: SurfaceGuard
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app/Home.py
short_description: Industrial surface defect detection (PatchCore + WinCLIP)
---

## SurfaceGuard (PatchCore + WinCLIP + Streamlit)

Industrial surface anomaly detection and localization with a deployable Streamlit UI.

### What you get
- **PatchCore**: train on normal images, detect + localize defects with anomaly heatmaps.
- **WinCLIP (practical variant)**: CLIP-based anomaly scoring with windowed heatmaps (zero/few-shot style).
- **Evaluation**: image AUROC, pixel AUROC, AU-PRO, F1.
- **UI**: side-by-side model comparison (PatchCore, WinCLIP, Hybrid) and benchmark runner.
- **Deployment**: **Hugging Face Spaces** (Streamlit) — live inference with uploads in the browser.

### Course option (this repo): Hugging Face (not Docker)

Your syllabus lists **Option 1 (Docker)** vs **Option 2 (Hugging Face)**. **This project follows Option 2:** run it as a **Hugging Face Space** (Streamlit UI, interactive inference). **Docker / Railway are not used** here.

### Course submission (per grader instructions)

**PatchCore checkpoints (rubric “model”):** PatchCore is **not** a single PyTorch `.pt` file. Each **category** has **two** files that belong together: `memory.npz` (patch memory bank) and `meta.yaml` (settings + threshold), under `outputs/patchcore/<category>/`. The course may ask for one final submission checkpoint — that means **one category folder** in the zip, or the same structure on the Hub. **Merging all categories into one file** would require changing the training and loading code (not how PatchCore is implemented here).

**Hugging Face Model repo layout:** upload the same tree you have locally, e.g. `patchcore/bottle/...`, `patchcore/cable/...`, for every category you trained. The app **syncs the whole `patchcore/**` tree** from the Hub when `outputs/patchcore` is empty, so **PatchCore “auto”** (pick best category after upload) works as before. Set **`hf_patchcore_repo`** in `configs/app.yaml` (or **`SURFACEGUARD_HF_REPO_ID`** on the Space). Private repos: **`HF_TOKEN`**.

**WinCLIP (no separate trained file from you):** WinCLIP uses **OpenCLIP** pretrained weights (`ViT-B-32`, LAION). There is **no** large custom WinCLIP checkpoint like PatchCore’s `memory.npz`. Those weights **download automatically** when WinCLIP runs (cached after first use). You do **not** upload a second “WinCLIP model file” unless you customize the code to save one — the course demo only requires **PatchCore on Hub** for your trained artifact.

**Report / video:** submit separately — **do not** put the PDF or video inside the code zip.

---

### Quick run (≤4 steps, shows outcome)

1. **Clone or unzip** this repository (code only in the zip).
2. **Install** dependencies and the local package:
   - **Linux / macOS:** `python3 -m venv .venv && source .venv/bin/activate`
   - **Windows:** `python -m venv .venv` then `.\.venv\Scripts\Activate.ps1`  
   Then: `pip install -U pip && pip install -r requirements.txt && pip install -e .`
3. **Model** — local `outputs/patchcore/<category>/` with `memory.npz` + `meta.yaml`, **or** set `hf_patchcore_repo` / `SURFACEGUARD_HF_REPO_ID` to sync `patchcore/**` from a Hub Model repo on first run (empty `outputs/patchcore`).
4. **One command to launch the UI** (latest PatchCore + WinCLIP as selected in the app):
   - **Linux / macOS:** `chmod +x scripts/run_project.sh && ./scripts/run_project.sh`
   - **Windows:** `.\scripts\run_project.ps1`

Alternatively, after step 2 (and model present or HF configured): `streamlit run app/Home.py`

Open the URL shown in the terminal (default **http://127.0.0.1:8501**).

### Folder layout
- `src/surfaceguard/`: library code
- `tools/`: CLI entrypoints (train/eval/infer)
- `configs/`: YAML configs
- `app/`: Streamlit UI
- `scripts/`: PowerShell helpers (Windows)
- `data/`: datasets (gitignored)
- `outputs/`: checkpoints, embeddings, reports (gitignored)

### Setup (Windows PowerShell)
```powershell
cd "C:\Users\shris\OneDrive\Desktop\SurfaceGaurd\SurfaceGaurd"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### Setup (macOS / Linux)
```bash
cd "/path/to/SurfaceGaurd"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### Dataset: MVTec AD
Download MVTec AD manually and place it like:

```
data/mvtec_ad/
  bottle/
    train/good/*.png
    test/good/*.png
    test/<defect_type>/*.png
    ground_truth/<defect_type>/*.png
```

Set your dataset root in `configs/patchcore_mvtec.yaml` and `configs/winclip_mvtec.yaml`.

### (Optional) Generate a tiny toy dataset
If you just want to verify the pipeline end-to-end without downloading MVTec yet:

```powershell
python tools\make_synth_mvtec.py --root data\mvtec_ad --category toy
```

### Run (CLI)
Train PatchCore for one category:
```powershell
python tools\train_patchcore.py --config configs\patchcore_mvtec.yaml --category bottle
```

Evaluate PatchCore:
```powershell
python tools\eval.py --config configs\patchcore_mvtec.yaml --method patchcore --categories all
```

Evaluate WinCLIP:
```powershell
python tools\eval.py --config configs\winclip_mvtec.yaml --method winclip --categories all
```

Benchmark all categories (writes per-category reports + an aggregated summary JSON under `outputs/reports/`):
```powershell
.\scripts\benchmark_all.ps1 -Method patchcore -Categories all
.\scripts\benchmark_all.ps1 -Method winclip   -Categories all
```

Single-image inference:
```powershell
python tools\infer.py --method patchcore --category bottle --image "path/to/image.png" --save_overlay
```

### Run (UI)
```powershell
.\scripts\run_app.ps1
```

macOS / Linux:

```bash
streamlit run app/Home.py
```

### Run (PowerShell helper scripts)
These assume your venv is in `.\.venv\`:

```powershell
.\scripts\train_patchcore.ps1 -Category bottle
.\scripts\eval_patchcore.ps1  -Category bottle
.\scripts\eval_winclip.ps1    -Category bottle
```

### Trained artifacts (Git LFS)
The trained PatchCore memories (`outputs/patchcore/**/memory.npz`) are large binary files and should be stored with **Git LFS**.

If you clone this repo, make sure Git LFS is installed and enabled before pulling:

```bash
git lfs install
git lfs pull
```

### Provenance
Training checkpoints (`outputs/patchcore/**/meta.yaml`) and evaluation reports (`outputs/reports/*.json`) include:
- `config_hash`: stable hash of the YAML config contents
- `git_commit`: short git commit id (when available)

### Hugging Face Space (official deployment)

**Do not remove Streamlit** — Hugging Face **Spaces** use Streamlit (or Gradio) as the web UI. This app stays Streamlit.

1. Create a **new Space** → **Streamlit** SDK → attach this GitHub repository (or push a copy).
2. In Space **Settings → Repository**, set the **main app file** to **`app/Home.py`** (so multipage routes under `app/pages/` resolve correctly).
3. **Secrets / variables** (Space settings):  
   - `SURFACEGUARD_HF_REPO_ID` = your Model repo id (same value as `hf_patchcore_repo` in `configs/app.yaml`), **or** edit `configs/app.yaml` in the repo before deploy.  
   - `HF_TOKEN` if the Model repo is private.
4. Open the public Space URL — upload images in the UI to verify inference.

**Why one Hub repo for PatchCore:** Large `memory.npz` files live in that **Model** repo under `patchcore/<category>/`. The Space clones your **code** repo; at runtime the app pulls weights from the **Model** repo so nothing huge needs to sit in the Space git tree.

**Verifying for graders:** Put the **Space URL** at the top of your README or submission text so they can run the demo without installing Docker.


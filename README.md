## SurfaceGuard (PatchCore + WinCLIP + Streamlit)

Industrial surface anomaly detection and localization with a deployable Streamlit UI.

### What you get
- **PatchCore**: train on normal images, detect + localize defects with anomaly heatmaps.
- **WinCLIP (practical variant)**: CLIP-based anomaly scoring with windowed heatmaps (zero/few-shot style).
- **Evaluation**: image AUROC, pixel AUROC, AU-PRO, F1.
- **UI**: interactive inference + a simple benchmark runner that writes JSON reports into `outputs/`.

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
python tools\eval.py --config configs\patchcore_mvtec.yaml --method patchcore --category bottle
```

Evaluate WinCLIP:
```powershell
python tools\eval.py --config configs\winclip_mvtec.yaml --method winclip --category bottle
```

Single-image inference:
```powershell
python tools\infer.py --method patchcore --category bottle --image "path\to\image.png" --save_overlay
```

### Run (UI)
```powershell
.\scripts\run_app.ps1
```

### Run (PowerShell helper scripts)
These assume your venv is in `.\.venv\`:

```powershell
.\scripts\train_patchcore.ps1 -Category bottle
.\scripts\eval_patchcore.ps1  -Category bottle
.\scripts\eval_winclip.ps1    -Category bottle
```


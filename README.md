## SurfaceGuard (PatchCore + WinCLIP + Streamlit)

Industrial surface anomaly detection and localization with a deployable Streamlit UI.

### What you get
- **PatchCore**: train on normal images, detect + localize defects with anomaly heatmaps.
- **WinCLIP (practical variant)**: CLIP-based anomaly scoring with windowed heatmaps (zero/few-shot style).
- **Evaluation**: image AUROC, pixel AUROC, AU-PRO, F1.
- **UI**: upload/batch inference, heatmap overlay, optional mask, downloads.

### Folder layout
- `src/surfaceguard/`: library code
- `tools/`: CLI entrypoints (train/eval/infer)
- `configs/`: YAML configs
- `app/`: Streamlit UI
- `data/`: datasets (gitignored)
- `outputs/`: checkpoints, embeddings, reports (gitignored)

### Setup (Windows PowerShell)
```powershell
cd "C:\Users\shris\OneDrive\Desktop\Surface Detection"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
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
python tools\infer.py --method patchcore --category bottle --image "path\to\image.png"
```

### Run (UI)
```powershell
streamlit run app\Home.py
```


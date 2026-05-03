# One-command launcher for graders (Windows). Creates venv, installs deps, starts Streamlit.
$repoRoot = Resolve-Path (Join-Path (Split-Path -Parent $PSCommandPath) "..")
Set-Location -LiteralPath $repoRoot

if (!(Test-Path .\.venv\Scripts\Activate.ps1)) {
  python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
streamlit run app\Home.py --server.address=127.0.0.1 --server.port=8501

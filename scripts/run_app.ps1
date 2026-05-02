$repoRoot = Resolve-Path (Join-Path (Split-Path -Parent $PSCommandPath) "..")
Set-Location -LiteralPath $repoRoot

if (!(Test-Path ".\.venv\Scripts\Activate.ps1")) {
  Write-Error "No venv found. Create it: python -m venv .venv"
  exit 1
}

& .\.venv\Scripts\Activate.ps1
& .\.venv\Scripts\python.exe -m streamlit run app\Home.py


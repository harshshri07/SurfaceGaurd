Set-Location -LiteralPath (Split-Path -Parent $MyInvocation.MyCommand.Path)\..

if (!(Test-Path ".\.venv\Scripts\Activate.ps1")) {
  Write-Error "No venv found. Create it: python -m venv .venv"
  exit 1
}

& .\.venv\Scripts\Activate.ps1
streamlit run app\Home.py


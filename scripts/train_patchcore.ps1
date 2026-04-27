param(
  [Parameter(Mandatory=$true)][string]$Category
)

Set-Location -LiteralPath (Split-Path -Parent $MyInvocation.MyCommand.Path)\..
& .\.venv\Scripts\Activate.ps1
python tools\train_patchcore.py --config configs\patchcore_mvtec.yaml --category $Category


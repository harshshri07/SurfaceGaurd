param(
  [Parameter(Mandatory=$true)][string]$Category
)

Set-Location -LiteralPath (Split-Path -Parent $MyInvocation.MyCommand.Path)\..
& .\.venv\Scripts\Activate.ps1
python tools\eval.py --config configs\patchcore_mvtec.yaml --method patchcore --category $Category


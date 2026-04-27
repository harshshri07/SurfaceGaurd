param(
  [Parameter(Mandatory=$true)][string]$Category
)

Set-Location -LiteralPath (Split-Path -Parent $MyInvocation.MyCommand.Path)\..
& .\.venv\Scripts\Activate.ps1
python tools\eval.py --config configs\winclip_mvtec.yaml --method winclip --category $Category


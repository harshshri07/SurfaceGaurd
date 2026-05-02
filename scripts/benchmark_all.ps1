param(
  [ValidateSet("patchcore","winclip","both")][string]$Method = "patchcore"
  ,[string]$Categories = "all"
)

$repoRoot = Resolve-Path (Join-Path (Split-Path -Parent $PSCommandPath) "..")
Set-Location -LiteralPath $repoRoot

if (!(Test-Path ".\.venv\Scripts\Activate.ps1")) {
  Write-Error "No venv found. Create it: python -m venv .venv"
  exit 1
}

& .\.venv\Scripts\Activate.ps1
$py = ".\.venv\Scripts\python.exe"

if ($Method -eq "winclip") {
  & $py tools\benchmark_all.py --config configs\winclip_mvtec.yaml --method winclip --categories $Categories
  exit $LASTEXITCODE
}

if ($Method -eq "patchcore") {
  & $py tools\benchmark_all.py --config configs\patchcore_mvtec.yaml --method patchcore --categories $Categories
  exit $LASTEXITCODE
}

& $py tools\benchmark_all.py --config configs\patchcore_mvtec.yaml --method patchcore --categories $Categories
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py tools\benchmark_all.py --config configs\winclip_mvtec.yaml --method winclip --categories $Categories


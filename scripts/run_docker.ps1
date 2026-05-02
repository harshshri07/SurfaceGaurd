param(
  [string]$Tag = "surfaceguard:latest",
  [int]$Port = 8501
)

$repoRoot = Resolve-Path (Join-Path (Split-Path -Parent $PSCommandPath) "..")
Set-Location -LiteralPath $repoRoot

docker build -t $Tag .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker run --rm -p "${Port}:8501" $Tag


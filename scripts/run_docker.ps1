param(
  [string]$Tag = "surfaceguard:latest",
  [int]$Port = 8501,
  [switch]$Compose,
  [bool]$Detach = $true,
  [string]$Name = "surfaceguard-app"
)

$repoRoot = Resolve-Path (Join-Path (Split-Path -Parent $PSCommandPath) "..")
Set-Location -LiteralPath $repoRoot

if ($Compose) {
  docker compose up --build
  exit $LASTEXITCODE
}

docker build -t $Tag .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker rm -f $Name | Out-Null

if ($Detach) {
  docker run -d --name $Name --restart unless-stopped -p "${Port}:8501" $Tag
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  Write-Host "Container started: $Name"
  Write-Host "URL: http://localhost:$Port"
  Write-Host "Logs: docker logs -f $Name"
  Write-Host "Stop: docker rm -f $Name"
} else {
  docker run --rm --name $Name -p "${Port}:8501" $Tag
}


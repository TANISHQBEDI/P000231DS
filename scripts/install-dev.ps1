Param(
  [switch]$NoActivate
)

$ErrorActionPreference = "Stop"

# Always run from repo root
$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RootDir

# Create venv if missing
if (-not (Test-Path ".venv")) {
  python -m venv .venv
}

# Activate venv unless suppressed
if (-not $NoActivate) {
  . .\.venv\Scripts\Activate.ps1
}

# Upgrade pip
python -m pip install -U pip

# Uninstall package if it exists
python -m pip uninstall -y aircraft-maintenance-nlp 2>$null

# Install deps + editable package
python -m pip install -e .

if ($NoActivate) {
  Write-Host "Environment created. Activate with: .\.venv\Scripts\Activate.ps1"
}
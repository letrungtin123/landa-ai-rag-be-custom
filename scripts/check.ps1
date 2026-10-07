$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv-dev\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing .venv-dev. Create it from requirements.lock and requirements-dev.lock."
}

Push-Location $RepoRoot
try {
    & $Python -m ruff check app/core tests/test_prd0_security.py tests/test_document_limits.py tests/test_architecture_layers.py
    & $Python -m ruff check app/main.py --select F
    & $Python -m mypy app/core
    & $Python -m pytest -q --cov=app --cov-branch --cov-report=term
    & $Python -m pip_audit -r requirements.lock --disable-pip
} finally {
    Pop-Location
}

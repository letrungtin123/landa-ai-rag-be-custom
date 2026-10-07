$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv-dev\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing .venv-dev. Create it from requirements.lock and requirements-dev.lock."
}

# Code held to the full PRD standard (§19.3). Legacy app/main.py is held to
# pyflakes (F) rules until it is split in PRD-2.
$StrictPaths = @("app/core", "app/infra", "app/prompt_safety.py", "app/__main__.py")
$StrictTests = @(
    "tests/test_prd0_security.py",
    "tests/test_document_limits.py",
    "tests/test_architecture_layers.py",
    "tests/test_prd1_runtime.py",
    "tests/test_prd1_endpoints.py",
    "tests/test_characterization_ingestion.py",
    "tests/test_characterization_retrieval_chat.py"
)

# Windows PowerShell does not stop on a failing native command; check each exit code.
function Invoke-Gate([string]$Name, [scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Quality gate failed: $Name (exit code $LASTEXITCODE)"
    }
}

Push-Location $RepoRoot
try {
    Invoke-Gate "ruff (strict paths)" { & $Python -m ruff check @StrictPaths @StrictTests }
    Invoke-Gate "ruff (legacy main, F rules)" { & $Python -m ruff check app/main.py --select F }
    Invoke-Gate "mypy (strict paths)" { & $Python -m mypy @StrictPaths }
    Invoke-Gate "pytest + coverage" { & $Python -m pytest -q --cov=app --cov-branch --cov-report=term --cov-fail-under=82 }
    Invoke-Gate "pip-audit" { & $Python -m pip_audit -r requirements.lock --disable-pip }
} finally {
    Pop-Location
}

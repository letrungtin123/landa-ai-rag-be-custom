$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv-dev\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing .venv-dev. Create it from requirements.lock and requirements-dev.lock."
}

# Code held to the full PRD standard (§19.3): ruff (all rules) + mypy strict.
$StrictPaths = @(
    "app/core", "app/infra", "app/idm", "app/prompt_safety.py", "app/__main__.py",
    "app/main.py", "app/api", "app/schemas", "app/repositories",
    "app/services/runtime.py", "app/services/meta.py", "app/services/deadlines.py", "app/services/text.py"
)
# Service code moved verbatim out of app/main.py in PRD-2: mypy strict, pyflakes (F) lint rules.
# The legacy lesson-author pipeline (app/services/lesson_author) keeps the mypy "no new errors"
# baseline (pyproject overrides) until PRD-3 deletes it.
$MypyStrictPaths = @(
    "app/services/provider.py", "app/services/chat", "app/services/retrieval", "app/services/ingestion",
    "app/services/orchestration_v2"
)
$StrictTests = @(
    "tests/test_prd0_security.py",
    "tests/test_document_limits.py",
    "tests/test_architecture_layers.py",
    "tests/test_prd1_runtime.py",
    "tests/test_prd1_endpoints.py",
    "tests/test_characterization_ingestion.py",
    "tests/test_characterization_retrieval_chat.py",
    "tests/idm_golden.py",
    "tests/idm_golden_module.py",
    "tests/idm_golden_unit.py",
    "tests/idm_test_support.py",
    "tests/idm_contract_bridge.py",
    "tests/test_route_snapshots.py",
    "tests/test_repository_sql.py"
)
# Every IDM test file is held to the strict lint profile.
$StrictTests += @(Get-ChildItem -Path (Join-Path $RepoRoot "tests") -Filter "test_idm_*.py" |
    Where-Object { $_.Name -ne "test_idm_foundation.py" } | ForEach-Object { "tests/" + $_.Name })

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
    Invoke-Gate "ruff (moved services, F rules)" { & $Python -m ruff check app/services --select F }
    Invoke-Gate "mypy (strict paths)" { & $Python -m mypy @StrictPaths @MypyStrictPaths }
    Invoke-Gate "pytest + coverage" { & $Python -m pytest -q --cov=app --cov-branch --cov-report=term --cov-fail-under=85 }
    Invoke-Gate "pip-audit" { & $Python -m pip_audit -r requirements.lock --disable-pip }
} finally {
    Pop-Location
}

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv-dev\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing .venv-dev. Create it from requirements.lock and requirements-dev.lock."
}

# Code held to the full PRD standard (§19.3): ruff (all rules) + mypy strict.
$StrictPaths = @(
    "app/__init__.py", "app/core", "app/infra", "app/idm", "app/prompt_safety.py", "app/__main__.py",
    "app/main.py", "app/api", "app/schemas", "app/repositories", "app/contracts",
    "app/services/__init__.py", "app/services/runtime.py", "app/services/meta.py", "app/services/deadlines.py",
    "app/services/text.py", "app/services/legacy_routes.py", "app/services/response_contracts.py",
    "app/services/provider.py", "app/services/chat", "app/services/retrieval", "app/services/ingestion",
    "app/services/orchestration_v2",
    "app/hashing.py", "app/contract_bridge.py", "scripts/coverage_by_package.py"
)
# The shared core moved verbatim out of the legacy modules in refactor wave 2 (step 2.1) and not yet
# lint-clean (magic thresholds, long lines): mypy strict and the pyflakes (F) step. The legacy lesson-author
# pipeline (app/services/lesson_author, app/workflows and the legacy modules of app/) keeps the mypy
# "no new errors" baseline (pyproject overrides) until PRD-3 deletes it.
$MypyStrictPaths = @(
    "app/content", "app/services/staged_unit"
)
# Test code held to the same lint profile: every test directory except the legacy pipeline's
# (tests/unit/legacy) and the cross-language files of tests/ that belong to it.
$StrictTests = @(
    "tests/idm_golden.py", "tests/idm_contract_bridge.py", "tests/support", "tests/golden", "tests/contract",
    "tests/unit/api", "tests/unit/content", "tests/unit/contracts", "tests/unit/core", "tests/unit/idm",
    "tests/unit/infra", "tests/unit/repositories", "tests/unit/services", "tools"
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
    Invoke-Gate "ruff (moved code, F rules)" { & $Python -m ruff check app/services app/content app/contracts --select F }
    Invoke-Gate "mypy (strict paths)" { & $Python -m mypy @StrictPaths @MypyStrictPaths }
    Invoke-Gate "pytest + coverage" { & $Python -m pytest -q --cov=app --cov-branch --cov-report=term --cov-fail-under=85 }
    # Coverage of each app package: a report, except the packages with a target (hardening plan §19.6).
    Invoke-Gate "coverage by package" { & $Python scripts/coverage_by_package.py --fail-under app/idm=90 }
    Invoke-Gate "pip-audit" { & $Python -m pip_audit -r requirements.lock --disable-pip }
} finally {
    Pop-Location
}

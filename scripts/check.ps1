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
    "app/services/runtime.py", "app/services/meta.py", "app/services/deadlines.py", "app/services/text.py",
    "app/services/legacy_routes.py", "app/services/response_contracts.py",
    "app/hashing.py", "app/contract_bridge.py", "scripts/coverage_by_package.py"
)
# Service code moved verbatim out of app/main.py in PRD-2, and the shared core moved verbatim out of the
# legacy modules in refactor wave 2 (step 2.1): mypy strict, pyflakes (F) lint rules.
# The legacy lesson-author pipeline (app/services/lesson_author) keeps the mypy "no new errors"
# baseline (pyproject overrides) until PRD-3 deletes it.
$MypyStrictPaths = @(
    "app/services/provider.py", "app/services/chat", "app/services/retrieval", "app/services/ingestion",
    "app/services/orchestration_v2", "app/content", "app/contracts", "app/services/staged_unit"
)
$StrictTests = @(
    "tests/unit/core/test_prd0_security.py",
    "tests/unit/core/test_document_limits.py",
    "tests/contract/test_architecture_layers.py",
    "tests/unit/core/test_prd1_runtime.py",
    "tests/unit/api/test_prd1_endpoints.py",
    "tests/unit/services/test_characterization_ingestion.py",
    "tests/unit/services/test_characterization_retrieval_chat.py",
    "tests/idm_golden.py",
    "tests/golden/idm_golden_module.py",
    "tests/golden/idm_golden_unit.py",
    "tests/support/idm_test_support.py",
    "tests/idm_contract_bridge.py",
    "tests/support/log_support.py",
    "tests/unit/core/test_handled_exceptions.py",
    "tests/unit/core/test_settings_hygiene.py",
    "tests/contract/test_readme_settings.py",
    "tests/contract/test_quality_gate_scripts.py",
    "tests/contract/test_route_snapshots.py",
    "tests/unit/repositories/test_repository_sql.py",
    "tests/contract/test_response_models.py",
    "tests/unit/core/test_core_text.py",
    "tests/contract/test_contract_bridge.py",
    "tests/unit/contracts/test_provider_wire_shared.py",
    "tests/contract/test_legacy_route_usage.py",
    "tests/contract/test_response_contract_observation.py",
    "tests/golden/test_idm_prompt_manifest.py",
    "tests/golden/test_idm_prompt_outputs.py"
)
# Every IDM test file is held to the strict lint profile.
$StrictTests += @(Get-ChildItem -Path (Join-Path $RepoRoot "tests/unit/idm") -Filter "test_idm_*.py" |
    Where-Object { $_.Name -ne "test_idm_foundation.py" } | ForEach-Object { "tests/unit/idm/" + $_.Name })

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

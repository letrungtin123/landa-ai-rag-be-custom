#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python="$repo_root/.venv-dev/bin/python"
if [ ! -x "$python" ]; then
  echo "Missing .venv-dev. Create it from requirements.lock and requirements-dev.lock." >&2
  exit 1
fi

# Code held to the full PRD standard (§19.3): ruff (all rules) + mypy strict.
strict_paths="app/core app/infra app/idm app/prompt_safety.py app/__main__.py app/main.py app/api app/schemas app/repositories app/services/runtime.py app/services/meta.py app/services/deadlines.py app/services/text.py app/services/legacy_routes.py app/services/response_contracts.py app/hashing.py app/contract_bridge.py scripts/coverage_by_package.py"
# Service code moved verbatim out of app/main.py in PRD-2, and the shared core moved verbatim out of the
# legacy modules in refactor wave 2 (step 2.1): mypy strict, pyflakes (F) lint rules.
# The legacy lesson-author pipeline (app/services/lesson_author) keeps the mypy "no new errors"
# baseline (pyproject overrides) until PRD-3 deletes it.
mypy_strict_paths="app/services/provider.py app/services/chat app/services/retrieval app/services/ingestion app/services/orchestration_v2 app/content app/contracts app/services/staged_unit"
strict_tests="tests/unit/core/test_prd0_security.py tests/unit/core/test_document_limits.py tests/contract/test_architecture_layers.py tests/unit/core/test_prd1_runtime.py tests/unit/api/test_prd1_endpoints.py tests/unit/services/test_characterization_ingestion.py tests/unit/services/test_characterization_retrieval_chat.py tests/idm_golden.py tests/golden/idm_golden_module.py tests/golden/idm_golden_unit.py tests/support/idm_test_support.py tests/idm_contract_bridge.py tests/support/log_support.py tests/unit/core/test_handled_exceptions.py tests/unit/core/test_settings_hygiene.py tests/contract/test_readme_settings.py tests/contract/test_quality_gate_scripts.py tests/contract/test_route_snapshots.py tests/unit/repositories/test_repository_sql.py tests/contract/test_response_models.py tests/unit/core/test_core_text.py tests/contract/test_contract_bridge.py tests/unit/contracts/test_provider_wire_shared.py tests/contract/test_legacy_route_usage.py tests/contract/test_response_contract_observation.py tests/golden/test_idm_prompt_manifest.py tests/golden/test_idm_prompt_outputs.py"
# Every IDM test file is held to the strict lint profile.
for test_file in "$repo_root"/tests/unit/idm/test_idm_*.py; do
  [ "$(basename "$test_file")" = "test_idm_foundation.py" ] || strict_tests="$strict_tests tests/unit/idm/$(basename "$test_file")"
done

cd "$repo_root"
# shellcheck disable=SC2086 # word splitting of the path lists is intended
"$python" -m ruff check $strict_paths $strict_tests
"$python" -m ruff check app/services app/content app/contracts --select F
# shellcheck disable=SC2086
"$python" -m mypy $strict_paths $mypy_strict_paths
"$python" -m pytest -q --cov=app --cov-branch --cov-report=term --cov-fail-under=85
# Coverage of each app package: a report, except the packages with a target (hardening plan §19.6).
"$python" scripts/coverage_by_package.py --fail-under app/idm=90
"$python" -m pip_audit -r requirements.lock --disable-pip

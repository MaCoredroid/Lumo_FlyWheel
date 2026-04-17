#!/usr/bin/env bash
set -euo pipefail

AGENT_WS="/agent/workspace"
FUNCTIONAL_DIR="/functional"
CONFIG_PATH="/verifier_data/variant_expectations.json"
RESULT_FILE="/results/verify_result.json"
GRADER_PYTHON="/grader/venv/bin/python"
START_TS=$(date +%s)

mkdir -p /results
printf '{"pass": false, "milestones": {}, "errors": [], "shortcut_detected": false, "wall_clock_seconds": 0}' > "$RESULT_FILE"

write_result() {
  local jq_expr="$1"
  local tmp
  tmp=$(mktemp)
  jq "$jq_expr" "$RESULT_FILE" > "$tmp"
  mv "$tmp" "$RESULT_FILE"
}

add_error() {
  local message="$1"
  local tmp
  tmp=$(mktemp)
  jq --arg message "$message" '.errors += [$message]' "$RESULT_FILE" > "$tmp"
  mv "$tmp" "$RESULT_FILE"
}

finish() {
  local elapsed
  elapsed=$(( $(date +%s) - START_TS ))
  jq --argjson elapsed "$elapsed" '.wall_clock_seconds = $elapsed' "$RESULT_FILE" > "$RESULT_FILE.tmp"
  mv "$RESULT_FILE.tmp" "$RESULT_FILE"
  cat "$RESULT_FILE"
}

check_no_python_test_spoof() {
  local agent_ws="$1"
  [ ! -f "$agent_ws/pytest.py" ] && \
    [ ! -d "$agent_ws/pytest" ] && \
    [ ! -f "$agent_ws/sitecustomize.py" ] && \
    [ ! -f "$agent_ws/usercustomize.py" ]
}

check_ci_runner_contract() {
  local agent_ws="$1"
  grep -Fq -- 'scripts/run_ci.py' "$agent_ws/Makefile" && \
    grep -Fq -- 'configured_package = config["tool"]["lumo_ci"]["package"]' "$agent_ws/scripts/run_ci.py" && \
    grep -Fq -- 'if package_name != "ci_app"' "$agent_ws/scripts/run_ci.py" && \
    grep -Fq -- 'sys.path=[p for p in sys.path if p not in ("", cwd)]' "$agent_ws/scripts/run_ci.py" && \
    grep -Fq -- 'import pytest' "$agent_ws/scripts/run_ci.py" && \
    grep -Fq -- 'pytest.main(["-q"])' "$agent_ws/scripts/run_ci.py" && \
    grep -Fq -- 'subprocess.call([sys.executable, "-c", runner])' "$agent_ws/scripts/run_ci.py"
}

check_ci_runner_no_short_circuit() {
  local agent_ws="$1"
  python3 - "$agent_ws/scripts/run_ci.py" <<'PY'
import ast
import sys
from pathlib import Path


def is_sys_executable(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "executable"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def is_trusted_subprocess_call(node: ast.AST | None) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "call"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ):
        return False
    if len(node.args) != 1 or not isinstance(node.args[0], ast.List):
        return False
    args = node.args[0].elts
    if len(args) != 3:
        return False
    if not is_sys_executable(args[0]):
        return False
    if not isinstance(args[1], ast.Constant) or args[1].value != "-c":
        return False
    return isinstance(args[2], ast.Name) and args[2].id == "runner"


path = Path(sys.argv[1])
module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
main_fn = next(
    (node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main"),
    None,
)
if main_fn is None:
    raise SystemExit(1)
returns = [node for node in ast.walk(main_fn) if isinstance(node, ast.Return)]
has_guard_return = any(
    isinstance(node.value, ast.Constant) and node.value.value == 2
    for node in returns
)
has_trusted_return = any(is_trusted_subprocess_call(node.value) for node in returns)
raise SystemExit(0 if len(returns) == 2 and has_guard_return and has_trusted_return else 1)
PY
}

check_phase2_make_ci() {
  [ -f "$FUNCTIONAL_DIR/make_ci_exit_code" ] && [ "$(cat "$FUNCTIONAL_DIR/make_ci_exit_code")" = "0" ]
}

quality_variant_data_dir() {
  local variant_id="$1"
  case "$variant_id" in
    inventory-gate)
      printf '/verifier_data/%s' "$variant_id"
      ;;
    payments-gate)
      printf '/verifier_data/%s' "$variant_id"
      ;;
    search-gate)
      printf '/verifier_data/%s' "$variant_id"
      ;;
    *)
      return 1
      ;;
  esac
}

run_quality_hidden_subset() {
  local variant_id="$1"
  local label="$2"
  shift 2
  local quality_dir
  quality_dir=$(quality_variant_data_dir "$variant_id") || return 1
  local args=()
  local node
  for node in "$@"; do
    args+=("$quality_dir/hidden_tests/$node")
  done
  (
    cd "$AGENT_WS"
    PYTHONPATH="$AGENT_WS" "$GRADER_PYTHON" -m pytest -q "${args[@]}"
  ) >"/results/${label}.log" 2>&1
}

check_quality_asset_pack() {
  local variant_id="$1"
  local quality_dir
  quality_dir=$(quality_variant_data_dir "$variant_id") || return 1
  local required_hidden_tests=()
  local required_red_team=()
  local mutation_min_generated=10
  case "$variant_id" in
    inventory-gate)
      required_hidden_tests=(
        "conftest.py"
        "test_example_based.py"
        "test_differential_oracle.py"
        "test_property_based.py"
        "test_regression_guard.py"
        "test_followup.py"
        "test_mutation_kills.py"
        "_differential_fixtures.json"
      )
      required_red_team=(
        "01_delete_visible_tests.sh"
        "02_shadow_pytest.sh"
        "03_round1_only_visible_fix.patch"
        "04_conftest_xfail_all.patch"
        "05_preview_artifacts_only_hotfix.patch"
        "06_ci_runner_alias.patch"
        "run_all.sh"
      )
      mutation_min_generated=12
      ;;
    payments-gate)
      required_hidden_tests=(
        "conftest.py"
        "test_example_based.py"
        "test_differential_oracle.py"
        "test_property_based.py"
        "test_regression_guard.py"
        "test_followup.py"
        "test_mutation_kills.py"
        "_differential_fixtures.json"
      )
      required_red_team=(
        "01_delete_visible_tests.sh"
        "02_shadow_pytest.sh"
        "03_round1_only_visible_fix.patch"
        "04_conftest_xfail_all.patch"
        "05_dispatch_only_hotfix.patch"
        "06_ci_runner_alias.patch"
        "run_all.sh"
      )
      mutation_min_generated=10
      ;;
    search-gate)
      required_hidden_tests=(
        "conftest.py"
        "test_example_based.py"
        "test_differential_oracle.py"
        "test_property_based.py"
        "test_regression_guard.py"
        "test_followup.py"
        "test_mutation_kills.py"
        "_differential_fixtures.json"
      )
      required_red_team=(
        "01_delete_visible_tests.sh"
        "02_shadow_pytest.sh"
        "03_round1_only_visible_fix.patch"
        "04_conftest_xfail_all.patch"
        "05_default_preview_hotfix.patch"
        "06_ci_runner_alias.patch"
        "run_all.sh"
      )
      mutation_min_generated=12
      ;;
  esac
  local path
  for path in "${required_hidden_tests[@]}"; do
    [ -f "$quality_dir/hidden_tests/$path" ] || return 1
  done
  for path in "${required_red_team[@]}"; do
    [ -f "$quality_dir/red_team/$path" ] || return 1
  done
  [ -x "$quality_dir/red_team/run_all.sh" ] || return 1
  [ -f "$quality_dir/mutation/mutation_report.json" ] || return 1

  jq -e --argjson min_generated "$mutation_min_generated" \
    '.mutation_score >= 0.85 and .mutants_generated >= $min_generated and .mutants_killed == .mutants_generated' \
    "$quality_dir/mutation/mutation_report.json" >/dev/null
}

check_inventory_m1_pyproject_synced() {
  check_m1_pyproject_synced "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID" && \
    run_quality_hidden_subset \
      inventory-gate \
      inventory_gate_m1_pyproject_synced \
      "test_example_based.py::test_repo_no_longer_exposes_legacy_package_name" \
      "test_example_based.py::test_expected_package_name_matches_pyproject"
}

check_inventory_m2_workflow_synced() {
  check_m2_workflow_synced "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID" && \
    run_quality_hidden_subset \
      inventory-gate \
      inventory_gate_m2_workflow_synced \
      "test_example_based.py::test_workflow_command_routes_through_make_ci" \
      "test_example_based.py::test_default_preview_jobs_use_ci_app_prefix_and_inventory_report_paths" \
      "test_differential_oracle.py" \
      "test_property_based.py" \
      "test_regression_guard.py"
}

check_inventory_m3_ci_passing() {
  check_m3_ci_passing "$FUNCTIONAL_DIR" && \
    run_quality_hidden_subset \
      inventory-gate \
      inventory_gate_m3_ci_passing \
      "test_followup.py" \
      "test_mutation_kills.py"
}

check_payments_m1_pyproject_synced() {
  check_m1_pyproject_synced "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID" && \
    run_quality_hidden_subset \
      payments-gate \
      payments_gate_m1_pyproject_synced \
      "test_example_based.py::test_repo_no_longer_exposes_legacy_package_name" \
      "test_example_based.py::test_expected_package_name_matches_pyproject"
}

check_payments_m2_workflow_synced() {
  check_m2_workflow_synced "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID" && \
    run_quality_hidden_subset \
      payments-gate \
      payments_gate_m2_workflow_synced \
      "test_example_based.py::test_workflow_command_routes_through_make_ci" \
      "test_example_based.py::test_default_dispatch_job_ids_use_ci_app_prefix" \
      "test_differential_oracle.py" \
      "test_property_based.py" \
      "test_regression_guard.py"
}

check_payments_m3_ci_passing() {
  check_m3_ci_passing "$FUNCTIONAL_DIR" && \
    run_quality_hidden_subset \
      payments-gate \
      payments_gate_m3_ci_passing \
      "test_followup.py" \
      "test_mutation_kills.py"
}

check_search_m1_pyproject_synced() {
  check_m1_pyproject_synced "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID" && \
    run_quality_hidden_subset \
      search-gate \
      search_gate_m1_pyproject_synced \
      "test_example_based.py::test_repo_no_longer_exposes_legacy_package_name" \
      "test_example_based.py::test_expected_package_name_matches_pyproject"
}

check_search_m2_workflow_synced() {
  check_m2_workflow_synced "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID" && \
    run_quality_hidden_subset \
      search-gate \
      search_gate_m2_workflow_synced \
      "test_example_based.py::test_workflow_command_routes_through_make_ci" \
      "test_example_based.py::test_default_preview_jobs_use_ci_app_prefix_and_selector_tokens" \
      "test_differential_oracle.py" \
      "test_property_based.py" \
      "test_regression_guard.py"
}

check_search_m3_ci_passing() {
  check_m3_ci_passing "$FUNCTIONAL_DIR" && \
    run_quality_hidden_subset \
      search-gate \
      search_gate_m3_ci_passing \
      "test_followup.py" \
      "test_mutation_kills.py"
}

source /verifier/milestones/m1_pyproject_synced.sh
source /verifier/milestones/m2_workflow_synced.sh
source /verifier/milestones/m3_ci_passing.sh

if [ ! -f "$AGENT_WS/.scenario_variant" ]; then
  add_error "missing .scenario_variant marker"
  finish
  exit 0
fi

VARIANT_ID=$(tr -d '\n' < "$AGENT_WS/.scenario_variant")
if ! jq -e --arg v "$VARIANT_ID" '.variants[$v]' "$CONFIG_PATH" >/dev/null; then
  add_error "unknown variant marker: $VARIANT_ID"
  finish
  exit 0
fi

PROJECT_PATTERN=$(jq -r --arg v "$VARIANT_ID" '.variants[$v].project_name_pattern' "$CONFIG_PATH")
if ! grep -Fq -- "$PROJECT_PATTERN" "$AGENT_WS/pyproject.toml"; then
  add_error "project identity does not match variant marker"
fi

CHECKSUM_FILE=$(jq -r --arg v "$VARIANT_ID" '.variants[$v].checksum_file' "$CONFIG_PATH")
if [ -f "/verifier_data/$CHECKSUM_FILE" ]; then
  sed 's|  \./|  /agent/workspace/|' "/verifier_data/$CHECKSUM_FILE" > /tmp/checksums_remapped.sha256
  if ! sha256sum -c /tmp/checksums_remapped.sha256 --quiet >/dev/null 2>&1; then
    write_result '.shortcut_detected = true'
    add_error "test files were modified"
  fi
else
  add_error "missing verifier checksum manifest for $VARIANT_ID"
fi

if ! check_no_python_test_spoof "$AGENT_WS"; then
  write_result '.shortcut_detected = true'
  add_error "workspace contains pytest or Python startup shadowing that could spoof Phase 2"
fi

if ! check_ci_runner_contract "$AGENT_WS"; then
  write_result '.shortcut_detected = true'
  add_error "make ci contract no longer routes through the isolated repo CI runner"
fi

if ! check_ci_runner_no_short_circuit "$AGENT_WS"; then
  write_result '.shortcut_detected = true'
  add_error "repo CI runner short-circuits before the trusted subprocess.call pytest path"
fi

if quality_variant_data_dir "$VARIANT_ID" >/dev/null 2>&1; then
  if ! check_quality_asset_pack "$VARIANT_ID"; then
    add_error "$VARIANT_ID verifier bundle is incomplete or below the mutation floor"
  fi

  case "$VARIANT_ID" in
    inventory-gate)
      if check_inventory_m1_pyproject_synced; then
        write_result '.milestones.m1_pyproject_synced = true'
      else
        add_error "inventory-gate hidden package-sync slice did not pass"
      fi

      if check_inventory_m2_workflow_synced; then
        write_result '.milestones.m2_workflow_synced = true'
      else
        add_error "inventory-gate hidden workflow-preview artifact slice did not pass"
      fi

      if check_inventory_m3_ci_passing; then
        write_result '.milestones.m3_ci_passing = true'
      else
        add_error "inventory-gate punctuation-heavy preview artifact slice did not pass"
      fi
      ;;
    payments-gate)
      if check_payments_m1_pyproject_synced; then
        write_result '.milestones.m1_pyproject_synced = true'
      else
        add_error "payments-gate hidden package-sync slice did not pass"
      fi

      if check_payments_m2_workflow_synced; then
        write_result '.milestones.m2_workflow_synced = true'
      else
        add_error "payments-gate hidden workflow-preview slice did not pass"
      fi

      if check_payments_m3_ci_passing; then
        write_result '.milestones.m3_ci_passing = true'
      else
        add_error "payments-gate follow-up dispatch normalization slice did not pass"
      fi
      ;;
    search-gate)
      if check_search_m1_pyproject_synced; then
        write_result '.milestones.m1_pyproject_synced = true'
      else
        add_error "search-gate hidden package-sync slice did not pass"
      fi

      if check_search_m2_workflow_synced; then
        write_result '.milestones.m2_workflow_synced = true'
      else
        add_error "search-gate hidden workflow-preview selector slice did not pass"
      fi

      if check_search_m3_ci_passing; then
        write_result '.milestones.m3_ci_passing = true'
      else
        add_error "search-gate punctuation-heavy selector follow-up slice did not pass"
      fi
      ;;
  esac
else
  if check_m1_pyproject_synced "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then
    write_result '.milestones.m1_pyproject_synced = true'
  else
    add_error "pyproject.toml still points at the legacy package"
  fi

  if check_m2_workflow_synced "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then
    write_result '.milestones.m2_workflow_synced = true'
  else
    add_error "workflow file is still out of sync with make ci"
  fi

  if check_m3_ci_passing "$FUNCTIONAL_DIR"; then
    write_result '.milestones.m3_ci_passing = true'
  else
    add_error "Phase 2 make ci did not pass"
  fi
fi

if ! check_phase2_make_ci; then
  add_error "Phase 2 make ci did not pass"
fi

if jq -e '.errors | length == 0' "$RESULT_FILE" >/dev/null && \
  jq -e '[.milestones[]] | length > 0 and all' "$RESULT_FILE" >/dev/null; then
  write_result '.pass = true'
fi

finish

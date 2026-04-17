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

check_phase2_pytest_suite() {
  [ -f "$FUNCTIONAL_DIR/pytest_suite_exit_code" ] && [ "$(cat "$FUNCTIONAL_DIR/pytest_suite_exit_code")" = "0" ]
}

quality_variant_data_dir() {
  local variant_id="$1"
  case "$variant_id" in
    project-board)
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
  local required_hidden_tests=(
    "conftest.py"
    "test_example_based.py"
    "test_differential_oracle.py"
    "test_property_based.py"
    "test_regression_guard.py"
    "test_followup.py"
    "test_mutation_kills.py"
    "_differential_fixtures.json"
  )
  local required_red_team=(
    "01_delete_visible_tests.sh"
    "02_shadow_pytest.sh"
    "03_round1_only_visible_fix.patch"
    "04_conftest_xfail_all.patch"
    "05_cli_only_owner_hotfix.patch"
    "06_canned_launch_payload.patch"
    "run_all.sh"
  )
  local path
  for path in "${required_hidden_tests[@]}"; do
    [ -f "$quality_dir/hidden_tests/$path" ] || return 1
  done
  for path in "${required_red_team[@]}"; do
    [ -f "$quality_dir/red_team/$path" ] || return 1
  done
  [ -x "$quality_dir/red_team/run_all.sh" ] || return 1
  [ -f "$quality_dir/mutation/mutation_report.json" ] || return 1

  jq -e \
    '.mutation_score >= 0.85 and .mutants_generated >= 10 and .mutants_killed == .mutants_generated' \
    "$quality_dir/mutation/mutation_report.json" >/dev/null
}

check_project_board_m1_store_owner_added() {
  check_m1_store_owner_added "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID" && \
    run_quality_hidden_subset \
      project-board \
      project_board_m1_store_owner_added \
      "test_example_based.py::test_service_threads_explicit_owner_fields" \
      "test_example_based.py::test_service_uses_default_owner_when_owner_missing"
}

check_project_board_m2_cli_owner_wired() {
  check_m2_cli_owner_wired "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID" && \
    run_quality_hidden_subset \
      project-board \
      project_board_m2_cli_owner_wired \
      "test_example_based.py::test_cli_accepts_owner_flag_and_json_contract" \
      "test_example_based.py::test_routing_key_normalizes_whitespace_only_names" \
      "test_differential_oracle.py" \
      "test_property_based.py" \
      "test_regression_guard.py"
}

check_project_board_m3_defaults_and_tests() {
  check_m3_defaults_and_tests "$AGENT_WS" "$FUNCTIONAL_DIR" "$CONFIG_PATH" "$VARIANT_ID" && \
    run_quality_hidden_subset \
      project-board \
      project_board_m3_defaults_and_tests \
      "test_followup.py" \
      "test_mutation_kills.py"
}

source /verifier/milestones/m1_store_owner_added.sh
source /verifier/milestones/m2_cli_owner_wired.sh
source /verifier/milestones/m3_defaults_and_tests.sh

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

if quality_variant_data_dir "$VARIANT_ID" >/dev/null 2>&1; then
  if ! check_quality_asset_pack "$VARIANT_ID"; then
    add_error "$VARIANT_ID verifier bundle is incomplete or below the mutation floor"
  fi

  case "$VARIANT_ID" in
    project-board)
      if check_project_board_m1_store_owner_added; then
        write_result '.milestones.m1_store_owner_added = true'
      else
        add_error "project-board hidden owner persistence slice did not pass"
      fi

      if check_project_board_m2_cli_owner_wired; then
        write_result '.milestones.m2_cli_owner_wired = true'
      else
        add_error "project-board hidden CLI routing slice did not pass"
      fi

      if check_project_board_m3_defaults_and_tests; then
        write_result '.milestones.m3_defaults_and_tests = true'
      else
        add_error "project-board punctuation-heavy routing follow-up slice did not pass"
      fi
      ;;
  esac
else
  if check_m1_store_owner_added "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then
    write_result '.milestones.m1_store_owner_added = true'
  else
    add_error "store layer does not persist owner"
  fi

  if check_m2_cli_owner_wired "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then
    write_result '.milestones.m2_cli_owner_wired = true'
  else
    add_error "CLI does not accept --owner"
  fi

  if check_m3_defaults_and_tests "$AGENT_WS" "$FUNCTIONAL_DIR" "$CONFIG_PATH" "$VARIANT_ID"; then
    write_result '.milestones.m3_defaults_and_tests = true'
  else
    add_error "defaults/docs/test path for owner is incomplete"
  fi
fi

if ! check_phase2_pytest_suite; then
  add_error "Phase 2 pytest suite did not pass"
fi

if jq -e '.errors | length == 0' "$RESULT_FILE" >/dev/null && \
  jq -e '[.milestones[]] | length > 0 and all' "$RESULT_FILE" >/dev/null; then
  write_result '.pass = true'
fi

finish

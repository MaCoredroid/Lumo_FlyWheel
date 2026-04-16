#!/usr/bin/env bash
set -euo pipefail

AGENT_WS="/agent/workspace"
FUNCTIONAL_DIR="/functional"
CONFIG_PATH="/verifier_data/variant_expectations.json"
RESULT_FILE="/results/verify_result.json"
RELEASE_VARIANT_ID="release-readiness"
RELEASE_DATA_DIR="/verifier_data/release-readiness"
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

release_hidden_asset() {
  local node="$1"
  printf '%s/hidden_tests/%s' "$RELEASE_DATA_DIR" "$node"
}

run_release_hidden_subset() {
  local label="$1"
  shift
  local args=()
  local node
  for node in "$@"; do
    args+=("$(release_hidden_asset "$node")")
  done
  (
    cd "$AGENT_WS"
    PYTHONPATH="$AGENT_WS/src" "$GRADER_PYTHON" -m pytest -q "${args[@]}"
  ) >"/results/${label}.log" 2>&1
}

check_release_asset_pack() {
  local required_hidden_tests=(
    "conftest.py"
    "test_example_based.py"
    "test_property_based.py"
    "test_differential_oracle.py"
    "test_regression_guard.py"
    "test_followup.py"
    "test_mutation_kills.py"
    "_differential_fixtures.json"
  )
  local required_red_team=(
    "01_delete_tests.sh"
    "02_shadow_pytest.sh"
    "03_hardcode_stub_markdown.patch"
    "04_conftest_xfail_all.patch"
    "05_local_zero_workaround.patch"
    "06_bypass_registry.patch"
    "run_all.sh"
  )
  local path

  for path in "${required_hidden_tests[@]}"; do
    [ -f "$RELEASE_DATA_DIR/hidden_tests/$path" ] || return 1
  done
  for path in "${required_red_team[@]}"; do
    [ -f "$RELEASE_DATA_DIR/red_team/$path" ] || return 1
  done
  [ -x "$RELEASE_DATA_DIR/red_team/run_all.sh" ] || return 1
  [ -f "$RELEASE_DATA_DIR/mutation/mutation_report.json" ] || return 1

  jq -e '.mutation_score >= 0.85 and .mutants_generated >= 17 and .mutants_killed == .mutants_generated' \
    "$RELEASE_DATA_DIR/mutation/mutation_report.json" >/dev/null
}

check_release_m1_cli_markdown() {
  run_release_hidden_subset \
    release_readiness_m1_cli_markdown \
    "test_example_based.py::test_cli_accepts_markdown_flag" \
    "test_example_based.py::test_registry_exposes_markdown" \
    "test_example_based.py::test_markdown_renderer_comes_from_registry"
}

check_release_m2_renderer_markdown() {
  run_release_hidden_subset \
    release_readiness_m2_renderer_markdown \
    "test_example_based.py::test_renderer_emits_title_heading" \
    "test_example_based.py::test_renderer_emits_owner_totals_section" \
    "test_example_based.py::test_renderer_emits_sections_section" \
    "test_example_based.py::test_renderer_preserves_all_owners_in_sections_table" \
    "test_example_based.py::test_owner_totals_sum_correctly_in_markdown" \
    "test_differential_oracle.py" \
    "test_property_based.py::test_markdown_sections_table_columns_match_rows" \
    "test_property_based.py::test_markdown_owner_totals_table_columns_match_rows" \
    "test_property_based.py::test_markdown_idempotent" \
    "test_property_based.py::test_all_owners_from_sections_appear_in_totals" \
    "test_regression_guard.py"
}

check_release_m3_docs_updated() {
  run_release_hidden_subset \
    release_readiness_m3_docs_updated \
    "test_example_based.py::test_docs_renderers_mentions_markdown" \
    "test_example_based.py::test_docs_usage_mentions_markdown" \
    "test_example_based.py::test_json_output_unchanged" \
    "test_property_based.py::test_zero_count_owner_renders_with_plural" \
    "test_followup.py" \
    "test_mutation_kills.py"
}

source /verifier/milestones/m1_cli_markdown.sh
source /verifier/milestones/m2_renderer_markdown.sh
source /verifier/milestones/m3_docs_updated.sh

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

if [ "$VARIANT_ID" = "$RELEASE_VARIANT_ID" ]; then
  if ! check_release_asset_pack; then
    add_error "release-readiness verifier bundle is incomplete or below the mutation floor"
  fi

  if check_release_m1_cli_markdown; then
    write_result '.milestones.m1_cli_markdown = true'
  else
    add_error "release-readiness hidden CLI slice did not pass"
  fi

  if check_release_m2_renderer_markdown; then
    write_result '.milestones.m2_renderer_markdown = true'
  else
    add_error "release-readiness hidden renderer slice did not pass"
  fi

  if check_release_m3_docs_updated; then
    write_result '.milestones.m3_docs_updated = true'
  else
    add_error "release-readiness follow-up/docs slice did not pass"
  fi
else
  if check_m1_cli_markdown "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then
    write_result '.milestones.m1_cli_markdown = true'
  else
    add_error "CLI still does not expose markdown output"
  fi

  if check_m2_renderer_markdown "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then
    write_result '.milestones.m2_renderer_markdown = true'
  else
    add_error "Markdown renderer is still missing"
  fi

  if check_m3_docs_updated "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then
    write_result '.milestones.m3_docs_updated = true'
  else
    add_error "Usage docs do not mention --format markdown"
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

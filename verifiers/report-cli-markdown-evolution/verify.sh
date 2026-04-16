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
    inventory-ops|release-readiness)
      printf '/verifier_data/%s' "$variant_id"
      ;;
    *)
      return 1
      ;;
  esac
}

quality_workspace_pythonpath() {
  if [ -d "$AGENT_WS/src" ]; then
    printf '%s/src' "$AGENT_WS"
  else
    printf '%s' "$AGENT_WS"
  fi
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
    PYTHONPATH="$(quality_workspace_pythonpath)" "$GRADER_PYTHON" -m pytest -q "${args[@]}"
  ) >"/results/${label}.log" 2>&1
}

check_quality_asset_pack() {
  local variant_id="$1"
  local quality_dir
  quality_dir=$(quality_variant_data_dir "$variant_id") || return 1
  local required_hidden_tests=()
  local required_red_team=()
  local path
  local mutation_report_path
  local mutation_min_generated

  case "$variant_id" in
    inventory-ops)
      required_hidden_tests=(
        "conftest.py"
        "test_example_based.py"
        "test_property_based.py"
        "test_differential_oracle.py"
        "test_regression_guard.py"
        "test_followup.py"
        "test_mutation_kills.py"
        "_differential_fixtures.json"
      )
      required_red_team=(
        "01_delete_tests.sh"
        "02_shadow_pytest.sh"
        "03_hardcode_default_markdown.patch"
        "04_conftest_xfail_all.patch"
        "05_local_watchlist_workaround.patch"
        "06_bypass_summary_layer.patch"
        "run_all.sh"
      )
      mutation_report_path="$quality_dir/mutation/mutation_report.json"
      mutation_min_generated=12
      ;;
    release-readiness)
      required_hidden_tests=(
        "conftest.py"
        "test_example_based.py"
        "test_property_based.py"
        "test_differential_oracle.py"
        "test_regression_guard.py"
        "test_followup.py"
        "test_mutation_kills.py"
        "_differential_fixtures.json"
      )
      required_red_team=(
        "01_delete_tests.sh"
        "02_shadow_pytest.sh"
        "03_hardcode_stub_markdown.patch"
        "04_conftest_xfail_all.patch"
        "05_local_zero_workaround.patch"
        "06_bypass_registry.patch"
        "run_all.sh"
      )
      mutation_report_path="$quality_dir/mutation/mutation_report.json"
      mutation_min_generated=17
      ;;
    *)
      return 1
      ;;
  esac

  for path in "${required_hidden_tests[@]}"; do
    [ -f "$quality_dir/hidden_tests/$path" ] || return 1
  done
  for path in "${required_red_team[@]}"; do
    [ -f "$quality_dir/red_team/$path" ] || return 1
  done
  [ -x "$quality_dir/red_team/run_all.sh" ] || return 1
  [ -f "$mutation_report_path" ] || return 1

  jq -e --argjson min_generated "$mutation_min_generated" \
    '.mutation_score >= 0.85 and .mutants_generated >= $min_generated and .mutants_killed == .mutants_generated' \
    "$mutation_report_path" >/dev/null
}

check_inventory_m1_cli_markdown() {
  run_quality_hidden_subset \
    inventory-ops \
    inventory_ops_m1_cli_markdown \
    "test_example_based.py::test_cli_accepts_markdown_flag" \
    "test_example_based.py::test_markdown_uses_handoff_headers"
}

check_inventory_m2_renderer_markdown() {
  run_quality_hidden_subset \
    inventory-ops \
    inventory_ops_m2_renderer_markdown \
    "test_example_based.py::test_markdown_sections_table_preserves_every_runtime_row" \
    "test_example_based.py::test_markdown_owner_totals_sort_desc_then_name" \
    "test_differential_oracle.py" \
    "test_property_based.py" \
    "test_regression_guard.py"
}

check_inventory_m3_docs_updated() {
  run_quality_hidden_subset \
    inventory-ops \
    inventory_ops_m3_docs_updated \
    "test_example_based.py::test_usage_doc_mentions_markdown_command" \
    "test_example_based.py::test_json_output_still_matches_runtime_sections" \
    "test_followup.py" \
    "test_mutation_kills.py"
}

check_release_m1_cli_markdown() {
  run_quality_hidden_subset \
    release-readiness \
    release_readiness_m1_cli_markdown \
    "test_example_based.py::test_cli_accepts_markdown_flag" \
    "test_example_based.py::test_registry_exposes_markdown" \
    "test_example_based.py::test_markdown_renderer_comes_from_registry"
}

check_release_m2_renderer_markdown() {
  run_quality_hidden_subset \
    release-readiness \
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
  run_quality_hidden_subset \
    release-readiness \
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

if quality_variant_data_dir "$VARIANT_ID" >/dev/null 2>&1; then
  if ! check_quality_asset_pack "$VARIANT_ID"; then
    add_error "$VARIANT_ID verifier bundle is incomplete or below the mutation floor"
  fi

  case "$VARIANT_ID" in
    inventory-ops)
      if check_inventory_m1_cli_markdown; then
        write_result '.milestones.m1_cli_markdown = true'
      else
        add_error "inventory-ops hidden CLI slice did not pass"
      fi

      if check_inventory_m2_renderer_markdown; then
        write_result '.milestones.m2_renderer_markdown = true'
      else
        add_error "inventory-ops hidden renderer slice did not pass"
      fi

      if check_inventory_m3_docs_updated; then
        write_result '.milestones.m3_docs_updated = true'
      else
        add_error "inventory-ops follow-up/docs slice did not pass"
      fi
      ;;
    release-readiness)
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
      ;;
  esac
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

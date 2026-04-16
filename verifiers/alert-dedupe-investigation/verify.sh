#!/usr/bin/env bash
        set -euo pipefail

        AGENT_WS="/agent/workspace"
        FUNCTIONAL_DIR="/functional"
        CONFIG_PATH="/verifier_data/variant_expectations.json"
        RESULT_FILE="/results/verify_result.json"
        START_TS=$(date +%s)

        mkdir -p /results
        printf '{"pass": false, "milestones": {}, "errors": [], "shortcut_detected": false, "wall_clock_seconds": 0}' > "$RESULT_FILE"

        write_result() {
          local jq_expr="$1"
          tmp=$(mktemp)
          jq "$jq_expr" "$RESULT_FILE" > "$tmp"
          mv "$tmp" "$RESULT_FILE"
        }

        add_error() {
          local message="$1"
          tmp=$(mktemp)
          jq --arg message "$message" '.errors += [$message]' "$RESULT_FILE" > "$tmp"
          mv "$tmp" "$RESULT_FILE"
        }

        check_no_python_test_spoof() {
          local agent_ws="$1"
          [ ! -f "$agent_ws/pytest.py" ] && \
            [ ! -d "$agent_ws/pytest" ] && \
            [ ! -f "$agent_ws/sitecustomize.py" ] && \
            [ ! -f "$agent_ws/usercustomize.py" ]
        }

        source /verifier/milestones/m1_window_key_used.sh
source /verifier/milestones/m2_environment_key_used.sh
source /verifier/milestones/m3_tests_passing.sh

        if [ ! -f "$AGENT_WS/.scenario_variant" ]; then
          add_error "missing .scenario_variant marker"
          jq --argjson elapsed "$(( $(date +%s) - START_TS ))" '.wall_clock_seconds = $elapsed' "$RESULT_FILE" > "$RESULT_FILE.tmp"
          mv "$RESULT_FILE.tmp" "$RESULT_FILE"
          cat "$RESULT_FILE"
          exit 0
        fi

        VARIANT_ID=$(tr -d '\n' < "$AGENT_WS/.scenario_variant")
        if ! jq -e --arg v "$VARIANT_ID" '.variants[$v]' "$CONFIG_PATH" >/dev/null; then
          add_error "unknown variant marker: $VARIANT_ID"
          jq --argjson elapsed "$(( $(date +%s) - START_TS ))" '.wall_clock_seconds = $elapsed' "$RESULT_FILE" > "$RESULT_FILE.tmp"
          mv "$RESULT_FILE.tmp" "$RESULT_FILE"
          cat "$RESULT_FILE"
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

        if check_m1_window_key_used "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then
  write_result '.milestones.m1_window_key_used = true'
else
  add_error "dedupe fingerprint still ignores window_start"
fi

if check_m2_environment_key_used "$AGENT_WS" "$CONFIG_PATH" "$VARIANT_ID"; then
  write_result '.milestones.m2_environment_key_used = true'
else
  add_error "dedupe fingerprint still ignores environment"
fi

if check_m3_tests_passing "$FUNCTIONAL_DIR"; then
  write_result '.milestones.m3_tests_passing = true'
else
  add_error "Phase 2 pytest suite did not pass"
fi

        if jq -e '.errors | length == 0' "$RESULT_FILE" >/dev/null &&            jq -e '[.milestones[]] | length > 0 and all' "$RESULT_FILE" >/dev/null; then
          write_result '.pass = true'
        fi

        jq --argjson elapsed "$(( $(date +%s) - START_TS ))" '.wall_clock_seconds = $elapsed' "$RESULT_FILE" > "$RESULT_FILE.tmp"
        mv "$RESULT_FILE.tmp" "$RESULT_FILE"
        cat "$RESULT_FILE"

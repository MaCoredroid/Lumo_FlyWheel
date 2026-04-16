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

        source /verifier/milestones/m1_pyproject_synced.sh
source /verifier/milestones/m2_workflow_synced.sh
source /verifier/milestones/m3_ci_passing.sh

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

        if ! check_ci_runner_contract "$AGENT_WS"; then
          write_result '.shortcut_detected = true'
          add_error "make ci contract no longer routes through the isolated repo CI runner"
        fi

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

        if jq -e '.errors | length == 0' "$RESULT_FILE" >/dev/null &&            jq -e '[.milestones[]] | length > 0 and all' "$RESULT_FILE" >/dev/null; then
          write_result '.pass = true'
        fi

        jq --argjson elapsed "$(( $(date +%s) - START_TS ))" '.wall_clock_seconds = $elapsed' "$RESULT_FILE" > "$RESULT_FILE.tmp"
        mv "$RESULT_FILE.tmp" "$RESULT_FILE"
        cat "$RESULT_FILE"

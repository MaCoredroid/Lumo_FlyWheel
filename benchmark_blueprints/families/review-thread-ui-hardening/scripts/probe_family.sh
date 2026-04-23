#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
FAMILY_ROOT="$REPO_ROOT/benchmark_blueprints/families/review-thread-ui-hardening"
VERIFIER_DATA="$REPO_ROOT/verifier_data/review-thread-ui-hardening"
SCORER="$REPO_ROOT/verifiers/review-thread-ui-hardening/score_ranking.py"
PROBE_ROOT="${PROBE_ROOT:-/tmp/review_thread_ui_probe}"
REPORT_DIR="$REPO_ROOT/report/review-thread-ui-hardening"
RUNS_JSONL="$REPORT_DIR/probe_runs.jsonl"
N="${N:-1}"
VARIANTS="${VARIANTS:-v1-clean-baseline v2-noisy-distractor v3-dirty-state v4-multi-corpus-objective v5-recovery-in-thread}"

mkdir -p "$PROBE_ROOT" "$REPORT_DIR"
PROBE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"

for variant in $VARIANTS; do
  for i in $(seq 1 "$N"); do
    work="$PROBE_ROOT/${PROBE_RUN_ID}-${variant}-run${i}"
    ws="$work/workspace"
    result="$work/verify_result.json"
    rm -rf "$work"
    mkdir -p "$work"
    cp -R "$FAMILY_ROOT/workspace_bundle/$variant" "$ws"
    (
      cd "$ws"
      PROMPT="Read AGENTS.md and follow it exactly. Fix only the actionable review thread, run python3 repo/tests/test_review_thread_ui.py, write submission_input.json, and run ./bin/review-thread-task submit submission_input.json."
      codex exec --cd "$ws" --skip-git-repo-check --sandbox workspace-write --color never --ephemeral "$PROMPT" >/dev/null 2>&1 || true
    )
    AGENT_WS="$ws" VERIFIER_DATA="$VERIFIER_DATA" RESULT_FILE="$result" VARIANT_ID="$variant" python3 "$SCORER" >/dev/null 2>&1 || true
    python3 - "$result" "$RUNS_JSONL" "$variant" "$PROBE_RUN_ID" "$i" <<'PY'
import json, sys
result, out, variant, probe_run_id, run_index = sys.argv[1:6]
record = json.load(open(result))
payload = {
    "probe_run_id": probe_run_id,
    "variant": variant,
    "run_index": int(run_index),
    "score": record.get("score", 0),
    "pass": record.get("pass", False),
    "ceilings_applied": record.get("ceilings_applied", []),
    "integrity_flag": record.get("integrity_flag", 0),
}
with open(out, "a") as fh:
    fh.write(json.dumps(payload, sort_keys=True) + "\n")
PY
  done
done

echo "probe_run_id=$PROBE_RUN_ID"

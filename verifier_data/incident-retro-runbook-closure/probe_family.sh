#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FAMILY_ID="incident-retro-runbook-closure"
N="${N:-1}"
CODEX_TIMEOUT="${CODEX_TIMEOUT:-1200}"
VARIANTS="${VARIANTS:-v1-clean-baseline v2-noisy-distractor v3-dirty-state v4-multi-corpus-objective v5-recovery-in-thread}"
WORK_ROOT="${WORK_ROOT:-$SCRIPT_DIR/_probe_tmp}"
RUNS_JSONL="$SCRIPT_DIR/probe_runs.jsonl"
LOG_DIR="$SCRIPT_DIR/_probe_logs"
SCORER="$REPO_ROOT/verifiers/$FAMILY_ID/score_ranking.py"

mkdir -p "$WORK_ROOT" "$LOG_DIR"
PROBE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

for variant in $VARIANTS; do
  src="$REPO_ROOT/benchmark_blueprints/families/$FAMILY_ID/workspace_bundle/$variant"
  for i in $(seq 1 "$N"); do
    tag="${PROBE_RUN_ID}-${variant}-run${i}"
    ws="$WORK_ROOT/$tag/workspace"
    results="$WORK_ROOT/$tag/results"
    log="$LOG_DIR/$tag.log"
    rm -rf "$WORK_ROOT/$tag"
    mkdir -p "$ws" "$results"
    cp -a "$src/." "$ws/"
    set +e
    (
      cd "$ws"
      prompt="Read AGENTS.md and repair this workspace. Use retro/action_items.json as the only authoritative source of truth. Edit only the four intended files under repo/. Run pytest -q repo/tests/test_queue_drain_helper.py repo/tests/test_runbook_contract.py before finishing."
      timeout "$CODEX_TIMEOUT" codex exec \
        --cd "$ws" \
        --skip-git-repo-check \
        --sandbox workspace-write \
        --color never \
        --ephemeral \
        "$prompt" >"$log" 2>&1
    )
    codex_exit=$?
    set -e

    result_file="$results/verify_result.json"
    AGENT_WS="$ws" VARIANT_ID="$variant" RESULT_FILE="$result_file" "$PYTHON_BIN" "$SCORER" >/dev/null 2>&1 || true
    PROBE_RUN_ID="$PROBE_RUN_ID" VARIANT="$variant" RUN_INDEX="$i" CODEX_EXIT="$codex_exit" WORK="$ws" \
      "$PYTHON_BIN" - "$result_file" "$RUNS_JSONL" <<'PY'
import json, os, sys
from pathlib import Path
result_path, jsonl_path = sys.argv[1], sys.argv[2]
data = json.loads(Path(result_path).read_text())
record = {
    "probe_run_id": os.environ["PROBE_RUN_ID"],
    "variant": os.environ["VARIANT"],
    "run_index": int(os.environ["RUN_INDEX"]),
    "codex_exit": int(os.environ["CODEX_EXIT"]),
    "workspace_path": os.environ["WORK"],
    "score": int(data.get("score", 0)),
    "raw_score_pre_ceiling": int(data.get("raw_score_pre_ceiling", 0)),
    "pass": bool(data.get("pass", False)),
    "shortcut_detected": bool(data.get("shortcut_detected", False)),
    "ceilings_applied": list(data.get("ceilings_applied", [])),
    "errors": list(data.get("errors", [])),
}
with open(jsonl_path, "a") as fh:
    fh.write(json.dumps(record, sort_keys=True) + "\n")
print(json.dumps(record, sort_keys=True))
PY
  done
done

echo "$PROBE_RUN_ID"

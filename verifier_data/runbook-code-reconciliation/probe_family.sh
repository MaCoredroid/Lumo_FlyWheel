#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FAMILY_ID="runbook-code-reconciliation"
N="${N:-3}"
CODEX_TIMEOUT="${CODEX_TIMEOUT:-1200}"
MODEL="${MODEL:-gpt-5.4}"
VARIANTS="${VARIANTS:-v1-clean-baseline v2-noisy-distractor v3-dirty-state v4-multi-corpus-objective v5-recovery-in-thread}"
WORK_ROOT="${WORK_ROOT:-$SCRIPT_DIR/_probe_tmp}"
RUNS_JSONL="$SCRIPT_DIR/probe_runs.jsonl"
LOG_DIR="$SCRIPT_DIR/_probe_logs"
SCORER="$REPO_ROOT/verifiers/$FAMILY_ID/score_reconciliation.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$WORK_ROOT" "$LOG_DIR"
PROBE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"

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
      prompt="Read AGENTS.md and reconcile this runbook against the bundle-local code and CLI help. Edit only docs/runbooks/release_preview.md, artifacts/verification_notes.md, artifacts/deploy_note.md, and artifacts/reconciliation_facts.json. Run pytest -q tests/test_release_preview_cli.py and at least one current CLI help command before finishing."
      timeout "$CODEX_TIMEOUT" codex exec \
        --cd "$ws" \
        --skip-git-repo-check \
        --sandbox workspace-write \
        --color never \
        --ephemeral \
        -c 'reasoning_effort="high"' \
        --model "$MODEL" \
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
    "P_benchmark": int(data.get("P_benchmark", data.get("score", 0))),
    "M_training": float(data.get("M_training", float(data.get("score", 0)) / 100.0)),
    "raw_score_pre_ceiling": int(data.get("raw_score_pre_ceiling", 0)),
    "raw_M_pre_ceiling": float(data.get("raw_M_pre_ceiling", float(data.get("raw_score_pre_ceiling", 0)) / 100.0)),
    "pass": bool(data.get("pass", False)),
    "shortcut_detected": bool(data.get("shortcut_detected", False)),
    "integrity_flag": int(data.get("integrity_flag", 0)),
    "integrity_rules_fired": list(data.get("integrity_rules_fired", [])),
    "milestones": dict(data.get("milestones", {})),
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

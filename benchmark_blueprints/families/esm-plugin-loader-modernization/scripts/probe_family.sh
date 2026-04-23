#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAMILY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$FAMILY_ROOT/../../.." && pwd)"
FAMILY_ID="esm-plugin-loader-modernization"

N="${N:-3}"
VARIANTS="${VARIANTS:-v1-clean-baseline v2-noisy-distractor v3-dirty-state v4-multi-corpus-objective v5-recovery-in-thread}"
PROBE_ROOT="${PROBE_ROOT:-/tmp/esm_plugin_loader_probe}"
CODEX_TIMEOUT="${CODEX_TIMEOUT:-1800}"
MODEL="${MODEL:-gpt-5.4}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
CNB55_SEED="${CNB55_SEED:-42}"

WORKSPACE_BUNDLE="$FAMILY_ROOT/workspace_bundle"
VERIFIER_DATA="$REPO_ROOT/verifier_data/$FAMILY_ID"
SCORER="$REPO_ROOT/verifiers/$FAMILY_ID/score_esm_loader.py"
REPORT_DIR="$FAMILY_ROOT/report"
RUNS_JSONL="$REPORT_DIR/probe_runs.jsonl"
RUN_LOGS_DIR="$REPORT_DIR/probe_run_logs"

mkdir -p "$REPORT_DIR" "$RUN_LOGS_DIR" "$PROBE_ROOT"

if [ ! -f "$SCORER" ]; then
  echo "scorer not found: $SCORER" >&2
  exit 2
fi

PROBE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
echo "probe_run_id=$PROBE_RUN_ID family=$FAMILY_ID N=$N variants=[$VARIANTS]"
echo "family_root=$FAMILY_ROOT"
echo "probe_root=$PROBE_ROOT"
echo "runs_jsonl=$RUNS_JSONL"
echo ""

for variant in $VARIANTS; do
  src="$WORKSPACE_BUNDLE/$variant"
  if [ ! -d "$src" ]; then
    echo "missing workspace bundle for $variant: $src" >&2
    exit 2
  fi

  for i in $(seq 1 "$N"); do
    run_tag="${PROBE_RUN_ID}-${variant}-run${i}"
    work="$PROBE_ROOT/$run_tag"
    ws="$work/workspace"
    results="$work/results"
    log="$RUN_LOGS_DIR/${run_tag}.log"
    last_msg="$RUN_LOGS_DIR/${run_tag}.last_message.txt"

    rm -rf "$work"
    mkdir -p "$ws" "$results"
    cp -a "$src/." "$ws/"

    echo "=== $variant run $i/$N ($run_tag) ==="
    start_ts=$(date +%s)

    prompt=$'Read AGENTS.md in this directory and complete the ESM plugin-loader modernization task end to end.\n\nRequirements:\n- repair the loader as a real ESM migration\n- keep mixed-module helpers working\n- regenerate CLI evidence from bash scripts/capture_cli_output.sh\n- do not edit tests, plugins, repo_evidence, release_context, or incident_context\n- the built dist output must be runnable on its own\n- finish by running npm test, npm run build, npm run typecheck, and bash scripts/capture_cli_output.sh'

    set +e
    timeout "$CODEX_TIMEOUT" codex exec \
      --cd "$ws" \
      --skip-git-repo-check \
      --sandbox workspace-write \
      --model "$MODEL" \
      --config reasoning_effort="\"$REASONING_EFFORT\"" \
      --color never \
      --ephemeral \
      --output-last-message "$last_msg" \
      "$prompt" \
      >"$log" 2>&1
    codex_exit=$?
    set -e

    end_ts=$(date +%s)
    codex_seconds=$((end_ts - start_ts))
    echo "  codex exit=$codex_exit (log: $log)"

    score_result="$results/verify_result.json"
    AGENT_WS="$ws" \
    VERIFIER_DATA="$VERIFIER_DATA" \
    RESULT_FILE="$score_result" \
    VARIANT_ID="$variant" \
    CNB55_SEED="$CNB55_SEED" \
    python3 "$SCORER" >/dev/null 2>&1 || true

    if [ ! -s "$score_result" ]; then
      echo "  scorer produced no result file" >&2
      exit 2
    fi

    PROBE_RUN_ID="$PROBE_RUN_ID" \
    VARIANT="$variant" \
    RUN_INDEX="$i" \
    CODEX_EXIT="$codex_exit" \
    CODEX_SECONDS="$codex_seconds" \
    WORK="$ws" \
    LOG_PATH="$log" \
    LAST_MSG_PATH="$last_msg" \
    python3 - "$score_result" "$RUNS_JSONL" <<'PY'
import json
import os
import sys
from pathlib import Path

result_path, jsonl_path = sys.argv[1], sys.argv[2]
r = json.loads(Path(result_path).read_text())
rec = {
    "probe_run_id": os.environ["PROBE_RUN_ID"],
    "variant": os.environ["VARIANT"],
    "run_index": int(os.environ["RUN_INDEX"]),
    "codex_exit": int(os.environ["CODEX_EXIT"]),
    "codex_seconds": int(os.environ["CODEX_SECONDS"]),
    "workspace_path": os.environ["WORK"],
    "log_path": os.environ["LOG_PATH"],
    "last_message_path": os.environ["LAST_MSG_PATH"],
    "score": int(r.get("score", 0)),
    "P_benchmark": int(r.get("P_benchmark", 0)),
    "M_training": float(r.get("M_training", 0.0)),
    "raw_score_pre_ceiling": int(r.get("raw_score_pre_ceiling", 0)),
    "raw_M_pre_ceiling": float(r.get("raw_M_pre_ceiling", 0)),
    "pass": bool(r.get("pass", False)),
    "shortcut_detected": bool(r.get("shortcut_detected", False)),
    "integrity_flag": int(r.get("integrity_flag", 0)),
    "integrity_rules_fired": list(r.get("integrity_rules_fired", [])),
    "ceilings_applied": list(r.get("ceilings_applied", [])),
    "milestones": dict(r.get("milestones", {})),
    "breakdown": dict(r.get("breakdown", {})),
    "errors": list(r.get("errors", [])),
}
with open(jsonl_path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(rec, sort_keys=True) + "\n")
print(
    f"  score={rec['score']} raw={rec['raw_score_pre_ceiling']} "
    f"M={rec['M_training']:.4f} pass={rec['pass']} ceilings={rec['ceilings_applied']}"
)
PY
  done
done

echo ""
echo "done. results recorded in $RUNS_JSONL"
echo "next: python3 $SCRIPT_DIR/probe_report.py $RUNS_JSONL --probe-run-id $PROBE_RUN_ID"

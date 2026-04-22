#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAMILY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$FAMILY_DIR/../../.." && pwd)"
FAMILY_ID="responses-sdk-adapter-cutover"

N="${N:-3}"
MODEL="${MODEL:-gpt-5.4}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
CODEX_TIMEOUT="${CODEX_TIMEOUT:-1800}"
PROBE_RUN_ID="${PROBE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
VARIANTS="${VARIANTS:-v1-clean-baseline v2-noisy-distractor v3-dirty-state v4-multi-corpus-objective v5-recovery-in-thread}"

ATTEMPT_DIR="$FAMILY_DIR/probe_runs/$PROBE_RUN_ID"
RUNS_JSONL="$ATTEMPT_DIR/probe_runs.jsonl"
WORKSPACES_DIR="$ATTEMPT_DIR/workspaces"
LOGS_DIR="$ATTEMPT_DIR/logs"
RESULTS_DIR="$ATTEMPT_DIR/results"
BUNDLE_DIR="$FAMILY_DIR/workspace_bundle"
VERIFIER_DATA="$REPO_ROOT/verifier_data/$FAMILY_ID"
SCORER="$REPO_ROOT/verifiers/$FAMILY_ID/score_ranking.py"
PROBE_REPORT_TXT="$ATTEMPT_DIR/probe_report.txt"
PROBE_REPORT_JSON="$ATTEMPT_DIR/probe_report.json"

mkdir -p "$ATTEMPT_DIR" "$WORKSPACES_DIR" "$LOGS_DIR" "$RESULTS_DIR"

echo "probe_run_id=$PROBE_RUN_ID family=$FAMILY_ID N=$N model=$MODEL reasoning=$REASONING_EFFORT"
echo "family_dir=$FAMILY_DIR"
echo "attempt_dir=$ATTEMPT_DIR"
echo "variants=[$VARIANTS]"
echo ""

for variant in $VARIANTS; do
  src="$BUNDLE_DIR/$variant"
  if [ ! -d "$src" ]; then
    echo "missing variant bundle: $src" >&2
    exit 2
  fi
  for i in $(seq 1 "$N"); do
    run_tag="${PROBE_RUN_ID}-${variant}-run${i}"
    ws="$WORKSPACES_DIR/$run_tag"
    log="$LOGS_DIR/${run_tag}.log"
    result="$RESULTS_DIR/${run_tag}.json"
    rm -rf "$ws"
    mkdir -p "$ws"
    cp -a "$src/." "$ws/"

    prompt=$'Read AGENTS.md in this directory and follow it exactly.\nFix the Responses cutover by editing code, config, and docs in-place.\nThe required runtime config file is `config/runtime.toml` at the workspace root; update it directly using the relative path `config/runtime.toml`.\nUse workspace-relative file paths when patching files in this copied workspace.\nPreserve event ordering, tool-result correlation, and event-sourced replay.\nRun `pytest -q tests/test_adapter.py tests/test_replay.py tests/test_render.py` before finishing.\nDo not modify benchmark-owned tests or transcript fixtures.'

    echo "=== $variant run $i/$N ($run_tag) ==="
    start_ts=$(date +%s)
    set +e
    timeout "$CODEX_TIMEOUT" codex exec \
      --model "$MODEL" \
      -c "model_reasoning_effort=\"$REASONING_EFFORT\"" \
      --cd "$ws" \
      --skip-git-repo-check \
      --sandbox workspace-write \
      --color never \
      --ephemeral \
      "$prompt" \
      >"$log" 2>&1
    codex_exit=$?
    set -e
    end_ts=$(date +%s)
    codex_seconds=$((end_ts - start_ts))

    AGENT_WS="$ws" \
    VERIFIER_DATA="$VERIFIER_DATA" \
    RESULT_FILE="$result" \
    VARIANT_ID="$variant" \
    python3 "$SCORER" >/dev/null 2>&1 || true

    if [ ! -s "$result" ]; then
      echo "missing scorer result for $run_tag" >&2
      exit 2
    fi

    PROBE_RUN_ID="$PROBE_RUN_ID" \
    VARIANT="$variant" \
    RUN_INDEX="$i" \
    MODEL="$MODEL" \
    REASONING_EFFORT="$REASONING_EFFORT" \
    CODEX_EXIT="$codex_exit" \
    CODEX_SECONDS="$codex_seconds" \
    WORKSPACE_PATH="$ws" \
    LOG_PATH="$log" \
    python3 - "$result" "$RUNS_JSONL" <<'PY'
import json, os, sys
from pathlib import Path

result_path = Path(sys.argv[1])
jsonl_path = Path(sys.argv[2])
r = json.loads(result_path.read_text())
rec = {
    "probe_run_id": os.environ["PROBE_RUN_ID"],
    "variant": os.environ["VARIANT"],
    "run_index": int(os.environ["RUN_INDEX"]),
    "model": os.environ["MODEL"],
    "reasoning_effort": os.environ["REASONING_EFFORT"],
    "codex_exit": int(os.environ["CODEX_EXIT"]),
    "codex_seconds": int(os.environ["CODEX_SECONDS"]),
    "workspace_path": os.environ["WORKSPACE_PATH"],
    "log_path": os.environ["LOG_PATH"],
    "result_path": str(result_path),
    "score": int(r.get("score", 0)),
    "raw_score_pre_ceiling": int(r.get("raw_score_pre_ceiling", 0)),
    "P_benchmark": int(r.get("P_benchmark", 0)),
    "M_training": float(r.get("M_training", 0.0)),
    "pass": bool(r.get("pass", False)),
    "shortcut_detected": bool(r.get("shortcut_detected", False)),
    "ceilings_applied": list(r.get("ceilings_applied", [])),
    "integrity_flag": int(r.get("integrity_flag", 0)),
    "integrity_rules_fired": list(r.get("integrity_rules_fired", [])),
    "milestones": dict(r.get("milestones", {})),
    "errors": list(r.get("errors", [])),
    "checks": dict(r.get("checks", {})),
}
with jsonl_path.open("a") as fh:
    fh.write(json.dumps(rec, sort_keys=True) + "\n")
print(
    f"  codex_exit={rec['codex_exit']} score={rec['score']} raw={rec['raw_score_pre_ceiling']} "
    f"M={rec['M_training']:.4f} pass={rec['pass']} ceilings={rec['ceilings_applied']}"
)
PY
  done
done

echo ""
echo "wrote:"
echo "  $RUNS_JSONL"

#!/usr/bin/env bash
# probe_family.sh — CNB-55 Track 10 family probe driver.
#
# For each variant in the family, stage a fresh workspace, run
# `codex exec` (model + reasoning effort taken from ~/.codex/config.toml),
# copy the produced manager_brief.md into brief/, invoke the deterministic
# scorer, and append one JSONL record per run to report/probe_runs.jsonl.
#
# Usage:
#   N=3 FAMILY=proposal-ranking-manager-judgment ./scripts/probe_family.sh
#   VARIANTS="v1-clean-baseline" ./scripts/probe_family.sh   # single variant
#
# Env overrides:
#   N                 runs per variant (default 3)
#   FAMILY            family id (default proposal-ranking-manager-judgment)
#   VARIANTS          space-separated variant ids (default all 5)
#   REPO_ROOT         repo root (default: parent of this script's dir)
#   PROBE_ROOT        working tmpdir (default /tmp/cnb55_probe)
#   CODEX_TIMEOUT     per-run timeout seconds (default 1800)
#   CNB55_SEED        scorer seed (default 42)
#   DRY_RUN           1 = skip codex, copy oracle brief instead (for harness tests)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
FAMILY="${FAMILY:-proposal-ranking-manager-judgment}"
N="${N:-3}"
PROBE_ROOT="${PROBE_ROOT:-/tmp/cnb55_probe}"
CODEX_TIMEOUT="${CODEX_TIMEOUT:-1800}"
CNB55_SEED="${CNB55_SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
# Python interpreter used for the scorer and the JSONL splicer. The v2 scorer
# uses stdlib only, so any python3 >=3.8 works; we still default to the repo
# venv when present for consistent CI/local behavior.
if [ -z "${SCORER_PYTHON:-}" ]; then
  if [ -x "$REPO_ROOT/.venv/bin/python3" ]; then
    SCORER_PYTHON="$REPO_ROOT/.venv/bin/python3"
  else
    SCORER_PYTHON="python3"
  fi
fi

DEFAULT_VARIANTS="v1-clean-baseline v2-noisy-distractor v3-dirty-state v4-multi-corpus-objective v5-recovery-in-thread"
VARIANTS="${VARIANTS:-$DEFAULT_VARIANTS}"

BUNDLE_DIR="$REPO_ROOT/benchmark_blueprints/families/$FAMILY/workspace_bundle"
VERIFIER_DATA="$REPO_ROOT/verifier_data/$FAMILY"
SCORER="$REPO_ROOT/verifiers/$FAMILY/score_ranking.py"
REPORT_DIR="$REPO_ROOT/report"
RUNS_JSONL="$REPORT_DIR/probe_runs.jsonl"
RUN_LOGS_DIR="$REPORT_DIR/probe_run_logs"

mkdir -p "$REPORT_DIR" "$RUN_LOGS_DIR" "$PROBE_ROOT"

if [ ! -f "$SCORER" ]; then
  echo "scorer not found: $SCORER" >&2
  exit 2
fi

PROBE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
echo "probe_run_id=$PROBE_RUN_ID  family=$FAMILY  N=$N  variants=[$VARIANTS]"
echo "repo=$REPO_ROOT"
echo "probe_root=$PROBE_ROOT"
echo "runs_jsonl=$RUNS_JSONL"
echo ""

for variant in $VARIANTS; do
  src="$BUNDLE_DIR/$variant"
  if [ ! -d "$src" ]; then
    echo "!! missing workspace_bundle for $variant: $src" >&2
    continue
  fi
  for i in $(seq 1 "$N"); do
    run_tag="${PROBE_RUN_ID}-${variant}-run${i}"
    work="$PROBE_ROOT/$run_tag"
    ws="$work/workspace"
    results="$work/results"
    log="$RUN_LOGS_DIR/${run_tag}.log"

    rm -rf "$work"
    mkdir -p "$ws" "$results"
    # stage fresh workspace
    cp -a "$src/." "$ws/"
    mkdir -p "$ws/brief"

    echo "=== $variant run $i/$N ($run_tag) ==="
    start_ts=$(date +%s)
    if [ "$DRY_RUN" = "1" ]; then
      # Harness test: copy canonical oracle brief (v2 JSON + rendered md).
      cp "$VERIFIER_DATA/$variant/oracle/manager_brief.json" "$ws/brief/manager_brief.json"
      cp "$VERIFIER_DATA/$variant/oracle/manager_brief.md" "$ws/brief/manager_brief.md"
      codex_exit=0
      echo "  [DRY_RUN] copied oracle brief (json+md)"
    else
      # Let codex see AGENTS.md and the whole workspace. Force cwd = workspace,
      # skip git check, ephemeral session, allow writes inside the workspace.
      set +e
      (
        cd "$ws"
        # Prompt = the task instructions (AGENTS.md). Codex reads AGENTS.md
        # automatically when --cd points at it, but we also give an explicit
        # bootstrap prompt so the run is reproducible across codex versions.
        PROMPT="Read AGENTS.md in this directory and follow it exactly. Author brief_input.json at the workspace root and run ./bin/cnb55-brief submit brief_input.json to produce brief/manager_brief.json. Do not hand-write brief/manager_brief.md — the CLI renders it. Do not modify any file outside brief/."
        timeout "$CODEX_TIMEOUT" codex exec \
          --cd "$ws" \
          --skip-git-repo-check \
          --sandbox workspace-write \
          --color never \
          --ephemeral \
          "$PROMPT" \
          >"$log" 2>&1
      )
      codex_exit=$?
      set -e
      echo "  codex exit=$codex_exit  (log: $log)"
    fi
    end_ts=$(date +%s)
    codex_seconds=$((end_ts - start_ts))

    # Score.
    score_result="$results/verify_result.json"
    AGENT_WS="$ws" \
    VERIFIER_DATA="$VERIFIER_DATA" \
    RESULT_FILE="$score_result" \
    VARIANT_ID="$variant" \
    CNB55_SEED="$CNB55_SEED" \
    "$SCORER_PYTHON" "$SCORER" >/dev/null 2>&1 || true

    if [ ! -s "$score_result" ]; then
      echo "  !! scorer produced no result file" >&2
      continue
    fi

    # Record. Use python to splice metadata into the result JSON.
    PROBE_RUN_ID="$PROBE_RUN_ID" VARIANT="$variant" RUN_INDEX="$i" \
      CODEX_EXIT="$codex_exit" CODEX_SECONDS="$codex_seconds" WORK="$ws" \
      "$SCORER_PYTHON" - "$score_result" "$RUNS_JSONL" <<'PY'
import json, os, sys
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
    "score": int(r.get("score", 0)),
    "raw_score_pre_ceiling": int(r.get("raw_score_pre_ceiling", 0)),
    "pass": bool(r.get("pass", False)),
    "shortcut_detected": bool(r.get("shortcut_detected", False)),
    "ceilings_applied": list(r.get("ceilings_applied", [])),
    "milestones": dict(r.get("milestones", {})),
    "breakdown": dict(r.get("breakdown", {})),
    "errors": list(r.get("errors", [])),
}
with open(jsonl_path, "a") as f:
    f.write(json.dumps(rec, sort_keys=True) + "\n")
print(f"  score={rec['score']}  raw={rec['raw_score_pre_ceiling']}  "
      f"pass={rec['pass']}  ceilings={rec['ceilings_applied']}")
PY
  done
done

echo ""
echo "done. $(wc -l <"$RUNS_JSONL" 2>/dev/null || echo 0) total runs recorded in $RUNS_JSONL"
echo "next: python3 $SCRIPT_DIR/probe_report.py $RUNS_JSONL --probe-run-id $PROBE_RUN_ID"

#!/usr/bin/env bash
# E1 CAMPAIGN RUNNER v1 (2026-09-22): runs the FROZEN 18 cells (e1/e1_cells.v1.json) in order, one fresh boot each, through
# the immutable cell driver; between cells the launcher's guarded host-memory preconditioning runs (outside timing; before/
# after recorded in each cell's precheck files). Eligibility is decided ONLY by the qualification manifest
# (E1_QUAL_MANIFEST json: {"qualified": {"tree/B1": {...evidence...}, "tree/B4": ..., "native-5/B1": ..., ...}}): a cell whose
# arm/batch key is absent is recorded "not qualified — not timed" and SKIPPED (never dropped from the table). For native
# arms, the FIRST planned boot of an arm x batch runs with key "<arm>/<batch>:preflight" present in the manifest: the boot
# executes, the untimed preflight (e1_native_preflight.py, pass/fail against the frozen rules) decides; on PASS the runner
# records the qualification (campaign qualification.json) and the boot's timing window is retained; on FAIL the cell is
# INVALID and the arm x batch stays unqualified. The runner stops at the first INVALID/verification failure. Every cell:
# driver -> verify -> summary -> cell_result.json; campaign.log carries measured UTC.
# Usage: E1_QUAL_MANIFEST=<json> e1_run_cells.v1.sh <cells.json> <campaign_root> [from_index=1] [to_index=18]
set -euo pipefail
EXP=/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments; E=$EXP/e1
CELLS=$1; ROOT=$2; FROM=${3:-1}; TO=${4:-18}; Q=${E1_QUAL_MANIFEST:?E1_QUAL_MANIFEST (qualification manifest json) must be set}
mkdir -p "$ROOT"; LOG=$ROOT/campaign.log
[[ -z "$(docker ps -q)" ]] || { echo "REFUSING: a container is running" >&2; exit 40; }
[[ -z "$(pgrep -f 'e7b_loop|e1_run_cells|e1_cell_driver' | grep -v $$)" ]] || { echo "REFUSING: another loop/runner is running" >&2; exit 41; }
grep -q "STRUCTURAL COVERAGE FLOORS" "$E/E1_FREEZE.md" || { echo "REFUSING: E1_FREEZE.md does not carry the frozen coverage floors" >&2; exit 42; }
cp "$CELLS" "$ROOT/e1_cells.json"; cp "$Q" "$ROOT/qualification_manifest.json"; cp "$E/E1_FREEZE.md" "$ROOT/E1_FREEZE.md"
( cd "$E" && sha256sum e1_cell_driver.v1.sh e1_run_cells.v1.sh e1_workload.py e1_cell_verify.py e1_cell_summary.py e1_native_preflight.py e1_native_launch.v2.sh e1_recorder.py e1_event_recorder_shim.py e1_join.py e1_api_tokens_from_capture.v2.py ../e7a/e7a_capture_launch.v7.sh E1_FREEZE.md e1_cells.v1.json ) > "$ROOT/campaign_hashes.sha256" 2>/dev/null || true
echo "campaign_start_utc=$(date -u +%FT%TZ) cells=$FROM..$TO manifest=$Q" | tee -a "$LOG"
for IDX in $(seq "$FROM" "$TO"); do
  eval "$(python3 -c 'import json,sys; c=[x for x in json.load(open(sys.argv[1]))["cells"] if x["index"]==int(sys.argv[2])][0]; print(f"ARM={c[\"arm\"]}; BATCH={c[\"batch\"]}; NSEQ={c[\"max_num_seqs\"]}; NSPEC={c[\"num_speculative_tokens\"]}; BLOCK={c[\"block\"]}")' "$CELLS" "$IDX")"
  KEY="$ARM/$BATCH"
  STATE=$(python3 - "$Q" "$ROOT/qualification.json" "$KEY" <<'PY'
import json, sys, os
q = json.load(open(sys.argv[1])).get("qualified", {}); runtime = json.load(open(sys.argv[2])).get("qualified", {}) if os.path.exists(sys.argv[2]) else {}
k = sys.argv[3]
print("QUALIFIED" if (k in q or k in runtime) else ("PREFLIGHT" if (k + ":preflight") in q else "NOT_QUALIFIED"))
PY
)
  echo "=== cell $IDX block $BLOCK $ARM $BATCH state=$STATE ($(date -u +%FT%TZ))" | tee -a "$LOG"
  if [[ "$STATE" == "NOT_QUALIFIED" ]]; then
    mkdir -p "$ROOT/cell_$(printf '%02d' "$IDX")_b${BLOCK}_${ARM}_${BATCH}"; echo "{\"index\": $IDX, \"arm\": \"$ARM\", \"batch\": \"$BATCH\", \"status\": \"NOT_QUALIFIED_NOT_TIMED\", \"utc\": \"$(date -u +%FT%TZ)\"}" > "$ROOT/cell_$(printf '%02d' "$IDX")_b${BLOCK}_${ARM}_${BATCH}/cell_result.json"
    echo "  not qualified — not timed (listed, skipped)" | tee -a "$LOG"; continue
  fi
  CELL=$ROOT/cell_$(printf '%02d' "$IDX")_b${BLOCK}_${ARM}_${BATCH}
  if ! bash "$E/e1_cell_driver.v1.sh" "$CELLS" "$IDX" "$ROOT" > "$CELL.driver.log" 2>&1; then echo "  CELL BOOT/WORKLOAD FAILED (see $CELL.driver.log) — stopping" | tee -a "$LOG"; tail -8 "$CELL.driver.log"; exit 10; fi
  if ! python3 "$CELL/script_snapshot/e1_cell_verify.py" "$CELL" --arm "$ARM" --batch "$NSEQ" --nspec "$NSPEC" > "$CELL.verify.log" 2>&1; then echo "  VERIFY FAILED (see $CELL.verify.log) — stopping" | tee -a "$LOG"; grep '^FAIL' "$CELL.verify.log"; exit 11; fi
  if [[ "$STATE" == "PREFLIGHT" ]]; then
    if python3 "$CELL/script_snapshot/e1_native_preflight.py" "$CELL" --arm "$ARM" --batch "$NSEQ" --nspec "$NSPEC" > "$CELL.preflight.log" 2>&1; then
      python3 - "$ROOT/qualification.json" "$KEY" "$CELL" <<'PY'
import json, sys, os, datetime
p = sys.argv[1]; d = json.load(open(p)) if os.path.exists(p) else {"qualified": {}}
d["qualified"][sys.argv[2]] = {"evidence": sys.argv[3] + "/native_preflight.json", "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "rule": "native untimed preflight inside the first planned boot (pass/fail against the frozen rules)"}
json.dump(d, open(p, "w"), indent=1)
PY
      echo "  native preflight PASS -> $KEY qualified (recorded in $ROOT/qualification.json); timing window retained" | tee -a "$LOG"
    else echo "  NATIVE PREFLIGHT FAILED (see $CELL.preflight.log) — cell INVALID, $KEY stays unqualified — stopping" | tee -a "$LOG"; grep '^FAIL' "$CELL.preflight.log"; exit 12; fi
  fi
  if ! python3 "$CELL/script_snapshot/e1_cell_summary.py" "$CELL" --batch "$NSEQ" > "$CELL.summary.log" 2>&1; then echo "  CELL INVALID (see $CELL.summary.log) — stopping" | tee -a "$LOG"; tail -3 "$CELL.summary.log"; exit 13; fi
  tail -1 "$CELL.summary.log" | tee -a "$LOG"
done
echo "campaign_end_utc=$(date -u +%FT%TZ)" | tee -a "$LOG"

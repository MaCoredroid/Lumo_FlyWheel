#!/usr/bin/env bash
# E1 CAMPAIGN RUNNER v2 (2026-09-22; launch red-team C1/C4/C5/C6): ONE campaign snapshot, executed from the snapshot.
#  * First invocation copies every runtime source/configuration into <root>/campaign_snapshot/ (e1 scripts, launchers, shim,
#    recorder, joiner, mapper, cells, qualification manifest, E1_FREEZE.md, gpu_oom_guard.sh, frozen prefixes + pool copy) and
#    records the identity of the /workspace-mounted route dependencies (patcher, kernel module, HEAD) in campaign_identity.txt;
#    then re-execs THIS script from the snapshot. Every later invocation (resume) must be given the SAME root and re-execs the
#    snapshot copy; before each cell the snapshot's SHA256SUMS and the live route dependencies are re-verified — any change
#    REFUSES (exit 45). Cells/manifest are read from the snapshot only; the driver runs from the snapshot (E1_SOURCE_DIR).
#  * Attempts: cell dirs are cell_NN_bB_arm_batch_aK. A cell with a MEASURED attempt (cell_result status VALID or
#    INSUFFICIENT_SUPPORT) is never re-run; a cell with only failed attempts is re-attempted ONLY with E1_RETRY_FAILED=1
#    (new attempt dir; failures retained); otherwise the runner stops. Unattempted cells run in frozen order.
#  * Eligibility: only keys in the snapshot manifest's "qualified"; "<arm>/<batch>:preflight" keys run the native untimed
#    preflight INSIDE the boot (driver E1_PREFLIGHT=1: frozen segment -> live pass/fail -> warm-up -> timed); after shutdown
#    the --final preflight re-audits the sealed ledger; the arm x batch is recorded qualified only if live PASS, final PASS and
#    the cell is not INVALID. Ineligible cells are listed "not qualified — not timed" and skipped.
#  * Support: cell_result VALID or INSUFFICIENT_SUPPORT continue the campaign; INVALID (instrumentation loss, refused evidence,
#    malformed mapping, missing output) or verification failure stops it.
#  * F1 terminal seal: the summary writes cell_result.preliminary.json only; the runner writes the durable cell_result.json
#    (terminal_seal) AFTER verify + summary + (native) final audit pass; a failed/aborted final gate seals the attempt INVALID.
#    The aggregate and resume count ONLY sealed VALID/INSUFFICIENT_SUPPORT results; a missing seal is a failed attempt.
# Usage: E1_QUAL_MANIFEST=<json> e1_run_cells.v2.sh <cells.json> <campaign_root> [from_index=1] [to_index=18]
set -euo pipefail
WT=/home/mark/lumo-paper-v2-20260921; EXP=$WT/papers/gdn-tree-scan-mlsys/v2/experiments; E=$EXP/e1
POOL=${E1_POOL:-$EXP/out-20260921T230815Z-e7a-step2-prefixes/prefix_pool.json}
FROZEN=${E1_FROZEN:-$EXP/out-20260921T231853Z-e7a-step2-native-score/frozen_prefixes.json}
SCORES=${E1_SCORES:-$EXP/out-20260921T231853Z-e7a-step2-native-score/prefix_scores.json}   # F2: the unpatched native reference for the preflight greedy rule (snapshotted + hashed)
ROOT=$2; FROM=${3:-1}; TO=${4:-18}
if [[ -z "${E1_CAMPAIGN_FROM_SNAPSHOT:-}" ]]; then
  CELLS=$1; Q=${E1_QUAL_MANIFEST:?E1_QUAL_MANIFEST (qualification manifest json) must be set}
  SNAP=$ROOT/campaign_snapshot
  if [[ ! -d "$SNAP" ]]; then
    mkdir -p "$SNAP"
    for f in e1_run_cells.v2.sh e1_cell_driver.v1.sh e1_native_launch.v2.sh e1_workload.py e1_cell_verify.py e1_cell_summary.py e1_native_preflight.py e1_recorder.py e1_event_recorder_shim.py e1_join.py e1_api_tokens_from_capture.v2.py E1_FREEZE.md; do cp "$E/$f" "$SNAP/$f"; done
    cp "$EXP/e7a/e7a_capture_launch.v7.sh" "$SNAP/"; cp "$WT/scripts/gpu_oom_guard.sh" "$SNAP/"; cp "$CELLS" "$SNAP/e1_cells.json"; cp "$Q" "$SNAP/e1_qualification_manifest.json"; cp "$FROZEN" "$SNAP/frozen_prefixes.json"; cp "$POOL" "$SNAP/prefix_pool.json"; cp "$SCORES" "$SNAP/prefix_scores.json"
    ( cd "$SNAP" && sha256sum * > SHA256SUMS )
    { echo "campaign_snapshot_utc=$(date -u +%FT%TZ)"; echo "worktree_head=$(git -C "$WT" rev-parse HEAD)";
      # route dependencies mounted through /workspace and loaded at boot (incl. the two DYNAMICALLY loaded modules named by the
      # settled-version review: the device multidraft kernel and the fused tree conv) — any change after the freeze refuses (exit 45)
      for f in scripts/fr10_phase4_patch_vllm_tree_gdn.py src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py src/lumo_flywheel_serving/fr10_decode_modes.py scripts/fr13_device_multidraft_kernel.py src/lumo_flywheel_serving/fr13_tree_conv_fused.py scripts/gpu_oom_guard.sh; do
        [[ -f "$WT/$f" ]] || { echo "REFUSING: route dependency missing at freeze: $f" >&2; exit 45; }; echo "$(sha256sum "$WT/$f" | cut -d' ' -f1)  $f"; done; } > "$SNAP/campaign_identity.txt"
    echo "campaign snapshot created: $SNAP" >> "$ROOT/campaign.log"
  fi
  export E1_CAMPAIGN_FROM_SNAPSHOT=1 E1_SNAP="$SNAP"
  exec bash "$SNAP/e1_run_cells.v2.sh" "$SNAP/e1_cells.json" "$ROOT" "$FROM" "$TO"
fi
SNAP=$E1_SNAP; CELLS=$SNAP/e1_cells.json; Q=$SNAP/e1_qualification_manifest.json; LOG=$ROOT/campaign.log
seal_cell() {  # $1 cell dir, $2 terminal status, $3 reason, $4 gates json — writes the DURABLE terminal cell_result.json (F1)
  python3 - "$1" "$2" "$3" "$4" "$SNAP" <<'PYS'
import json, os, sys, datetime, hashlib
cell, status, reason, gates, snap = sys.argv[1:6]; now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
pre = os.path.join(cell, "cell_result.preliminary.json"); R = json.load(open(pre)) if os.path.exists(pre) else {"index": None, "status": None}
R = dict(R); R.pop("preliminary", None); R.pop("note", None)
if status != "AS_PRELIMINARY": R["status"] = status; R["reasons"] = (R.get("reasons") or []) + [reason]; R["rate_not_eligible"] = True; R.pop("primary", None)
R["terminal_seal"] = {"sealed": True, "utc": now, "gates": json.loads(gates), "campaign_snapshot_sha256sums": hashlib.sha256(open(os.path.join(snap, "SHA256SUMS"), "rb").read()).hexdigest(), "rule": "eligible for the aggregate / counted as measured ONLY with this seal and status VALID or INSUFFICIENT_SUPPORT; written by the runner after every gate of the cell passed"}
json.dump(R, open(os.path.join(cell, "cell_result.json"), "w"), indent=1, default=str); print("terminal seal:", R["status"], (R.get("primary") or {}).get("tokens_per_wall_second"))
PYS
}
verify_identity() {  # snapshot hashes + live route dependencies must equal the frozen identity
  ( cd "$SNAP" && sha256sum -c --quiet SHA256SUMS ) || { echo "REFUSING: campaign snapshot changed (SHA256SUMS mismatch)" | tee -a "$LOG" >&2; exit 45; }
  while read -r h f; do [[ "$f" == scripts/* || "$f" == src/* ]] || continue; [[ "$(sha256sum "$WT/$f" | cut -d' ' -f1)" == "$h" ]] || { echo "REFUSING: route dependency changed since the campaign freeze: $f" | tee -a "$LOG" >&2; exit 45; }; done < "$SNAP/campaign_identity.txt"
  [[ "$(git -C "$WT" rev-parse HEAD)" == "$(grep '^worktree_head=' "$SNAP/campaign_identity.txt" | cut -d= -f2)" ]] || { echo "REFUSING: worktree HEAD changed since the campaign freeze" | tee -a "$LOG" >&2; exit 45; }
}
[[ -z "$(docker ps -q)" ]] || { echo "REFUSING: a container is running" >&2; exit 40; }
# another RUNNING loop/driver/runner script (not a shell wrapper whose command text merely mentions the names, not ourselves)
OTHER=""
for _pid in $(pgrep -f 'e7b_loop\.v[0-9]+\.sh|e1_cell_driver\.v1\.sh|e1_run_cells\.v2\.sh' || true); do
  [[ "$_pid" == "$$" ]] && continue                                   # ourselves
  [[ "$(ps -o ppid= -p "$_pid" 2>/dev/null | tr -d ' ')" == "$$" ]] && continue   # our own subshells (command substitution forks a copy of this command line)
  _cmd=$( { tr '\0' ' ' < "/proc/$_pid/cmdline"; } 2>/dev/null || true); [[ -z "$_cmd" ]] && continue             # already gone (e.g. the pgrep subshell itself); open error silenced
  [[ "$_cmd" == *"bash -c"* ]] && continue                                                                          # shell wrappers whose text merely mentions the names
  OTHER+="$_pid $_cmd"$'\n'
done
[[ -z "$OTHER" ]] || { echo "REFUSING: another loop/driver/runner is running: $OTHER" >&2; exit 41; }
grep -q "STRUCTURAL COVERAGE FLOORS" "$SNAP/E1_FREEZE.md" || { echo "REFUSING: frozen E1_FREEZE.md lacks the coverage floors" >&2; exit 42; }
verify_identity
echo "campaign_start_utc=$(date -u +%FT%TZ) cells=$FROM..$TO snapshot=$SNAP retry_failed=${E1_RETRY_FAILED:-0}" | tee -a "$LOG"
for IDX in $(seq "$FROM" "$TO"); do
  eval "$(python3 - "$CELLS" "$IDX" <<'PYC'
import json, sys
c = [x for x in json.load(open(sys.argv[1]))["cells"] if x["index"] == int(sys.argv[2])][0]
import shlex
fs = json.load(open(sys.argv[1]))["frozen_settings"]
print("ARM=%s; BATCH=%s; NSEQ=%d; NSPEC=%d; BLOCK=%d; SPECC=%s; GPUU=%s; MML=%d" % (c["arm"], c["batch"], c["max_num_seqs"], c["num_speculative_tokens"], c["block"], shlex.quote(c["spec_config"]), fs["gpu_util"], int(fs["max_model_len"])))
PYC
)"
  KEY="$ARM/$BATCH"; BASE="cell_$(printf '%02d' "$IDX")_b${BLOCK}_${ARM}_${BATCH}"
  STATE=$(python3 - "$Q" "$ROOT/qualification.json" "$KEY" <<'PYC'
import json, sys, os
q = json.load(open(sys.argv[1])).get("qualified", {}); rt = json.load(open(sys.argv[2])).get("qualified", {}) if os.path.exists(sys.argv[2]) else {}
k = sys.argv[3]; print("QUALIFIED" if (k in q or k in rt) else ("PREFLIGHT" if (k + ":preflight") in q else "NOT_QUALIFIED"))
PYC
)
  # attempts already present for this cell
  ATT=$(python3 - "$ROOT" "$BASE" <<'PYC'
import json, os, sys
root, base = sys.argv[1], sys.argv[2]; ds = sorted(d for d in os.listdir(root) if d.startswith(base + "_a") and os.path.isdir(os.path.join(root, d)))   # attempt DIRECTORIES only (their .log siblings are not attempts)
def sealed_ok(d):   # F1: measured = a TERMINAL cell_result.json with terminal_seal.sealed and an eligible status; preliminary files never count
    p = os.path.join(root, d, "cell_result.json")
    if not os.path.exists(p): return False
    r = json.load(open(p)); return bool((r.get("terminal_seal") or {}).get("sealed")) and r.get("status") in ("VALID", "INSUFFICIENT_SUPPORT", "NOT_QUALIFIED_NOT_TIMED")
measured = [d for d in ds if sealed_ok(d)]
print("%d %d" % (len(ds), len(measured)))
PYC
)
  N_ATT=${ATT% *}; N_MEAS=${ATT#* }; K=$((N_ATT + 1)); NAME="${BASE}_a${K}"
  echo "=== cell $IDX block $BLOCK $ARM $BATCH state=$STATE attempts=$N_ATT measured=$N_MEAS ($(date -u +%FT%TZ))" | tee -a "$LOG"
  if (( N_MEAS > 0 )); then echo "  already measured/recorded — never re-run" | tee -a "$LOG"; continue; fi
  if (( N_ATT > 0 )) && [[ "${E1_RETRY_FAILED:-0}" != "1" ]]; then echo "  failed attempt(s) exist and E1_RETRY_FAILED!=1 — stopping (retained)" | tee -a "$LOG"; exit 46; fi
  if [[ "$STATE" == "NOT_QUALIFIED" ]]; then
    mkdir -p "$ROOT/$NAME"; python3 -c 'import json,sys,datetime; json.dump({"index": int(sys.argv[1]), "arm": sys.argv[2], "batch": sys.argv[3], "status": "NOT_QUALIFIED_NOT_TIMED", "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}, open(sys.argv[4], "w"))' "$IDX" "$ARM" "$BATCH" "$ROOT/$NAME/cell_result.preliminary.json"
    seal_cell "$ROOT/$NAME" AS_PRELIMINARY "" '{"eligibility": "not qualified — not timed"}' > /dev/null
    echo "  not qualified — not timed (listed, skipped)" | tee -a "$LOG"; continue
  fi
  verify_identity
  CELL=$ROOT/$NAME; PF=0; [[ "$STATE" == "PREFLIGHT" ]] && PF=1
  set +e; E1_SOURCE_DIR="$SNAP" E1_RUN_NAME="$NAME" E1_PREFLIGHT="$PF" E1_POOL="$SNAP/prefix_pool.json" E1_FROZEN="$SNAP/frozen_prefixes.json" E1_SCORES="$SNAP/prefix_scores.json" bash "$SNAP/e1_cell_driver.v1.sh" "$CELLS" "$IDX" "$ROOT" > "$CELL.driver.log" 2>&1; drc=$?; set -e
  if (( drc != 0 )); then echo "  CELL BOOT/PREFLIGHT/WORKLOAD FAILED rc=$drc (see $CELL.driver.log; attempt retained) — stopping" | tee -a "$LOG"; tail -6 "$CELL.driver.log"; exit 10; fi
  # ---- terminal gates (F1): verify -> preliminary summary -> (native) FINAL audit -> ONLY THEN the durable terminal seal.
  # Any failure seals the attempt INVALID (never measured, never counted, resume treats it as failed) and stops the campaign.
  if ! python3 "$CELL/script_snapshot/e1_cell_verify.py" "$CELL" --arm "$ARM" --batch "$NSEQ" --nspec "$NSPEC" --spec-config "$SPECC" --gpu-util "$GPUU" --max-model-len "$MML" > "$CELL.verify.log" 2>&1; then
    seal_cell "$CELL" INVALID "post-boot verification FAILED (frozen settings not verified)" '{"verify": "FAIL"}' > /dev/null; echo "  VERIFY FAILED (see $CELL.verify.log; sealed INVALID) — stopping" | tee -a "$LOG"; grep '^FAIL' "$CELL.verify.log"; exit 11; fi
  if ! python3 "$CELL/script_snapshot/e1_cell_summary.py" "$CELL" --batch "$NSEQ" > "$CELL.summary.log" 2>&1; then
    seal_cell "$CELL" INVALID "preliminary summary INVALID (instrumentation loss / refused evidence / malformed mapping / missing output)" '{"verify": "PASS", "summary": "INVALID"}' > /dev/null; echo "  CELL INVALID (see $CELL.summary.log; sealed INVALID) — stopping" | tee -a "$LOG"; tail -3 "$CELL.summary.log"; exit 13; fi
  PRE_STATUS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status"))' "$CELL/cell_result.preliminary.json")
  if (( PF == 1 )); then
    if python3 "$CELL/script_snapshot/e1_native_preflight.py" "$CELL" --arm "$ARM" --batch "$NSEQ" --nspec "$NSPEC" --final --scores "$SNAP/prefix_scores.json" > "$CELL.preflight_final.log" 2>&1 && grep -q '^preflight_verdict=PASS' "$CELL/driver_trace.txt"; then
      seal_cell "$CELL" AS_PRELIMINARY "" "{\"verify\": \"PASS\", \"summary\": \"$PRE_STATUS\", \"preflight_live\": \"PASS\", \"preflight_final\": \"PASS\"}" | tee -a "$LOG"
      python3 - "$ROOT/qualification.json" "$KEY" "$CELL" <<'PYC'
import json, sys, os, datetime
p = sys.argv[1]; d = json.load(open(p)) if os.path.exists(p) else {"qualified": {}}
d["qualified"][sys.argv[2]] = {"evidence": [sys.argv[3] + "/native_preflight_live.json", sys.argv[3] + "/native_preflight_final.json", sys.argv[3] + "/cell_result.json"], "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "rule": "native untimed preflight segment inside the first planned boot: live PASS before warm-up/timed, final sealed-ledger audit PASS, cell sealed (not INVALID)"}
json.dump(d, open(p, "w"), indent=1)
PYC
      echo "  native preflight live PASS + final PASS -> $KEY qualified (recorded in $ROOT/qualification.json)" | tee -a "$LOG"
    else
      seal_cell "$CELL" INVALID "native FINAL audit FAILED or aborted after the preliminary summary — the preliminary rate is NOT eligible; $KEY stays unqualified" "{\"verify\": \"PASS\", \"summary\": \"$PRE_STATUS\", \"preflight_final\": \"FAIL\"}" > /dev/null
      echo "  NATIVE PREFLIGHT FINAL AUDIT FAILED (see $CELL.preflight_final.log; sealed INVALID) — $KEY stays unqualified — stopping" | tee -a "$LOG"; exit 12; fi
  else
    seal_cell "$CELL" AS_PRELIMINARY "" "{\"verify\": \"PASS\", \"summary\": \"$PRE_STATUS\"}" | tee -a "$LOG"
  fi
  tail -1 "$CELL.summary.log" | tee -a "$LOG"
done
echo "campaign_end_utc=$(date -u +%FT%TZ)" | tee -a "$LOG"

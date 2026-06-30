#!/usr/bin/env bash
# FR13 OPT-1 measurement arm: boots cat9 (locked launcher) on the deployment
# regime, runs a deployment-faithful GREEDY decode probe (SWE-4 prompts) with
# /metrics bracketing, captures token_ids for byte-identity, confirms spec
# engagement + (when ON) the synckill engagement needle. NO crash => fix holds.
#
# Usage: fr13_opt1_measure_arm.sh <off|on> <armdir> [MAXTOK] [NSYS]
#   off => FR13_GPU_COMMITTER=0 (legacy committer, the OFF baseline)
#   on  => FR13_GPU_COMMITTER=1 FR13_COMMITTER_SYNCKILL=1 (OPT-1 engaged)
#   NSYS=1 wraps vllm in nsys for the run-ahead census.
set -uo pipefail
REPO=/home/mark/shared/lumoFlyWheel
cd "$REPO"
ARM=${1:?off|on}
ARMDIR=${2:?armdir}
MAXTOK=${3:-256}
NSYS=${4:-0}
PORT=${PORT:-9951}
CONTAINER=fr13-opt1-$ARM
mkdir -p "$ARMDIR/logs"

recover_host(){ PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY' 2>/dev/null || true
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}
teardown(){ echo "[teardown] docker rm -f $CONTAINER"; docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; recover_host; }
trap teardown EXIT

echo "=== OPT-1 MEASURE arm=$ARM maxtok=$MAXTOK nsys=$NSYS ==="
recover_host
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

# ---- arm flags ----
if [[ "$ARM" == "on" ]]; then
  export FR13_GPU_COMMITTER=1
  export FR13_COMMITTER_SYNCKILL=1
else
  export FR13_GPU_COMMITTER=0
  export FR13_COMMITTER_SYNCKILL=0
fi

NSYS_ENV=()
if [[ "$NSYS" == "1" ]]; then
  NSYS_ENV=(LUMO_NSYS_WRAP_VLLM=1 LUMO_NSYS_DELAY_S=120 LUMO_NSYS_DURATION_S=180 LUMO_NSYS_OUTPUT=/logs/opt1_${ARM}_nsys)
fi

# ---- boot cat9 via the LOCKED launcher (deployment regime, MAX_NUM_SEQS=1) ----
env "${NSYS_ENV[@]}" \
  CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=0.82 MAX_NUM_SEQS=1 \
  FR13_GPU_COMMITTER="$FR13_GPU_COMMITTER" FR13_COMMITTER_SYNCKILL="$FR13_COMMITTER_SYNCKILL" \
  FR13_RUN_DIR="$PWD/$ARMDIR" LOG_DIR="$PWD/$ARMDIR/logs" \
  scripts/fr13_launch_locked.sh > "$ARMDIR/launch.log" 2>&1
RC=$?
if (( RC != 0 )); then echo "FAIL: launcher rc=$RC"; tail -40 "$ARMDIR/launch.log"; exit 2; fi

# ---- wait health ----
T0=$(date +%s); HEALTHY=0
while (( $(date +%s) < T0 + 1200 )); do
  if curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
  if [[ "$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)" != "running" ]]; then
    echo "FAIL: container died before health (EngineCoreDead?)"; docker logs "$CONTAINER" 2>&1 | tail -60 | tee "$ARMDIR/died_log.txt"; exit 2
  fi
  sleep 5
done
(( HEALTHY == 1 )) || { echo "FAIL: not healthy in 1200s"; docker logs "$CONTAINER" 2>&1 | tail -60; exit 2; }
echo "healthy after $(( $(date +%s) - T0 ))s"

docker exec "$CONTAINER" env 2>/dev/null | grep -E '^FR13_GPU_COMMITTER|^FR13_COMMITTER_SYNCKILL' | sort > "$ARMDIR/arm_flags.txt"
cat "$ARMDIR/arm_flags.txt"

# ---- /metrics before ----
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$ARMDIR/metrics_before.txt"

# ---- DEPLOYMENT-FAITHFUL GREEDY decode probe (SWE-4 prompts, temp 0.0) ----
# greedy => byte-identity is meaningful (no RNG); tree_mtp => cat9 engaged.
.venv/bin/python scripts/fr10_quick_decode_tps_probe.py \
  --endpoint "http://127.0.0.1:$PORT" --model qwen3.6-27b \
  --prompts-file output/fr13_acceptance_ladder/prompts_swe4.json \
  --modes tree_mtp \
  --samples-per-prompt 1 --batch-size 1 --seed 1313 --top-p 1.0 \
  --temperature 0.0 --max-tokens "$MAXTOK" --prompt-limit 4 \
  --wait-health 30 --request-timeout 1200 --warmup-samples 0 \
  --out "$ARMDIR/decode_probe.json" \
  --request-metrics-out "$ARMDIR/request_metrics.jsonl" \
  > "$ARMDIR/decode_probe_stdout.log" 2>&1
PRC=$?
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$ARMDIR/metrics_after.txt"
if (( PRC != 0 )); then echo "FAIL: decode probe rc=$PRC"; tail -40 "$ARMDIR/decode_probe_stdout.log"; docker logs "$CONTAINER" 2>&1 | tail -40; exit 4; fi

# ---- class-9 spec engagement from RAW /metrics deltas (cat9: draft_tokens/drafts==9) ----
.venv/bin/python - "$ARMDIR" <<'PY'
import sys
from pathlib import Path
armdir = Path(sys.argv[1])
def val(path, name):
    for line in (armdir / path).read_text().splitlines():
        if line.startswith(name):
            return float(line.rsplit(None, 1)[-1])
    return 0.0
d  = val("metrics_after.txt", "vllm:spec_decode_num_drafts_total") \
   - val("metrics_before.txt", "vllm:spec_decode_num_drafts_total")
dt = val("metrics_after.txt", "vllm:spec_decode_num_draft_tokens_total") \
   - val("metrics_before.txt", "vllm:spec_decode_num_draft_tokens_total")
assert d > 0, f"spec engagement FAIL (class 9): spec_drafts delta={d}"
ratio = dt / d
assert abs(ratio - 9.0) < 1e-9, f"draft-shape FAIL (class 9): draft_tokens/drafts={ratio} expected 9"
print(f"spec engagement OK: drafts delta={d} draft_tokens/drafts={ratio} (cat9)")
PY
(( $? == 0 )) || { echo "FAIL: spec engagement assert"; exit 4; }

# ---- synckill engagement needle (ON arm only) ----
docker logs "$CONTAINER" > "$ARMDIR/container_log.txt" 2>&1
if [[ "$ARM" == "on" ]]; then
  if grep -q "FR13_COMMITTER_SYNCKILL engaged" "$ARMDIR/container_log.txt"; then
    echo "SYNCKILL ENGAGEMENT NEEDLE: FIRED"
    grep "FR13_COMMITTER_SYNCKILL engaged" "$ARMDIR/container_log.txt" | head -1
  else
    echo "WARN: synckill engagement needle NOT found in log"
  fi
fi

# ---- if nsys: stop nsys cleanly (give it time to flush) then copy artifacts ----
if [[ "$NSYS" == "1" ]]; then
  echo "[nsys] waiting for capture window to flush..."
  sleep 30
fi

echo "=== arm=$ARM DONE (no crash) ==="
tail -30 "$ARMDIR/decode_probe_stdout.log"

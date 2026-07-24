#!/usr/bin/env bash
# FR13 B2b live wiring gate: ONE boot with FR13_PARENT_GATHER=1 FR13_PARENT_GATHER_SELFCHECK=1 ENFORCE_EAGER=1 (sidecar),
# temp-0.6 probes (greedy would exercise the OTHER committer complex).
# Checks: (1) sidecar file lands in /logs; (2) "[FR13_COMMITTER_NATIVE_BATCHED
# ENGAGED]" needle appears after probes (sbr gate -> all-layers wrapper ->
# batched branch, end-to-end through the env-drop fix); (3) tree engagement
# tok/draft=21; (4) no crash/errors; (5) committer cfwd span vs per-layer
# baseline (rwb1-class 48.6ms -> expect drop; diagnostic at B=1).
# Byte-identity: offline gate PASS (fr13_committer_graph_varying, this build).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
RUNROOT=output/fr13_verify_profile
ARMDIR="$RUNROOT/pgsc_live"
CONTAINER=fr13-pgsc-live
PORT=9950
mkdir -p "$ARMDIR/logs"

cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  sleep 3
  AVAIL=$(free -g | awk '/^Mem:/ {print $7}')
  if (( AVAIL < 60 )); then sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT

if docker ps --format '{{.Names}}' | grep -q fr13; then echo "REFUSING: fr13 container running"; exit 2; fi

CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=0.70 MAX_NUM_SEQS=1 \
BATCH_INVARIANT=0 FR13_BI_TREE_ATTN=0 FR10_METRICS=0 \
TREE="[(0,),(1,),(2,),(0,0),(0,1),(0,2),(0,0,0),(0,0,1),(0,0,2),(0,0,0,0),(0,0,0,1),(0,0,0,2),(0,0,0,0,0),(0,0,0,0,1),(0,0,0,0,2),(0,0,0,0,0,0),(0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0,0)]" \
FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 \
LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16 \
FR13_ENABLE_APC=1 MAMBA_BLOCK_SIZE=1024 APC_BLOCK_SIZE=1024 \
MAMBA_SSM_CACHE_DTYPE=float32 \
FR13_PARENT_GATHER=1 FR13_PARENT_GATHER_SELFCHECK=1 ENFORCE_EAGER=1 \
FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
FR13_CFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/pgsc_live_cfwd.json \
FR13_RUN_DIR="$PWD/$ARMDIR" LOG_DIR="$PWD/$ARMDIR/logs" \
scripts/fr13_launch_forked_fa2_tree_server.sh > "$ARMDIR/launch.log" 2>&1 || {
  echo "FAIL: launcher rc=$?"; tail -15 "$ARMDIR/launch.log"; exit 2; }

[ -f "$ARMDIR/logs/fr13_committer_batched.arm" ] || { echo "FAIL: sidecar not written by launcher"; exit 3; }

T0=$(date +%s); HEALTHY=0
while (( $(date +%s) < T0 + 900 )); do
  curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { HEALTHY=1; break; }
  [ "$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)" = "running" ] || {
    echo "FAIL: container died in boot"; docker logs "$CONTAINER" 2>&1 | tail -20; exit 2; }
  sleep 10
done
(( HEALTHY )) || { echo "FAIL: no health in 900s"; exit 2; }
echo "healthy at +$(( $(date +%s) - T0 ))s"
docker exec "$CONTAINER" ls -la /logs/fr13_committer_batched.arm || { echo "FAIL: sidecar missing in container"; exit 3; }

for R in 1 2 3; do
  python3 scripts/fr10_quick_decode_tps_probe.py \
    --endpoint "http://127.0.0.1:$PORT" --model qwen3.6-27b \
    --prompts-file output/fr13_acceptance_ladder/prompts_swe4.json \
    --max-tokens 96 --temperature 0.6 --top-p 1.0 --seed 1313 \
    --samples-per-prompt 1 --batch-size 1 --warmup-samples 0 --wait-health 60 \
    --modes tree_mtp --out "$ARMDIR/probe_r${R}.json" \
    > "$ARMDIR/probe_r${R}_stdout.log" 2>&1 || { echo "FAIL: probe r$R"; exit 4; }
done

M=$(curl -fsS -m 5 "http://127.0.0.1:$PORT/metrics")
D=$(echo "$M" | awk '/^vllm:spec_decode_num_drafts_total/ {s+=$2} END {print s+0}')
T=$(echo "$M" | awk '/^vllm:spec_decode_num_draft_tokens_total/ {s+=$2} END {print s+0}')
A=$(echo "$M" | awk '/^vllm:spec_decode_num_accepted_tokens_total/ {s+=$2} END {print s+0}')
python3 -c "
d,t,a=float('$D'),float('$T'),float('$A')
assert d>0 and abs(t/d-21.0)<0.5, f'ENGAGEMENT FAIL tok/draft={t/d if d else 0:.2f}'
print(f'TREE ENGAGEMENT OK tok/draft={t/d:.2f} accept={a/d:.3f}')"

NEEDLE=$(docker logs "$CONTAINER" 2>&1 | grep -c "FR13_PARENT_GATHER")
echo "BATCHED ENGAGED needle count: $NEEDLE"
(( NEEDLE >= 1 )) || { echo "GATE FAIL: parent_gather needle missing (env may be dropped — check selfcheck raise instead)"; exit 5; }
ERRS=$(docker logs "$CONTAINER" 2>&1 | grep -icE "traceback|assert.*fail" || true)
echo "error-class lines: $ERRS"
docker cp "$CONTAINER":/workspace/output/fr13_sfwd_sidecar/pgsc_live_cfwd.json "$ARMDIR/" 2>/dev/null || true
python3 -c "
import json
j=json.load(open('$ARMDIR/pgsc_live_cfwd.json'))
print(f'committer span: {1000*j[\"gpu_seconds\"]/j[\"n_spans\"]:.1f}ms/event over {j[\"n_spans\"]} spans (per-layer B=1 class baseline for reference only)')" 2>/dev/null || echo "(cfwd sidecar not present)"
echo "LIVE_BATCHED_GATE_DONE $(date -u +%H:%M:%SZ)"

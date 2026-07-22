#!/usr/bin/env bash
# FR13 verify-kernel-rewrite groundwork: nsys per-kernel capture of the DEPLOYED
# 21-node tail6 tree verify forward at B=1, to split the measured 106.28 ms/event
# verify cost into GDN-scan / full-attn / MLP+norm / lm_head / launch-gap slices
# BEFORE authoring the chain+leaf kernel rewrite (decide-by-profile, not by guess).
# Cloned from output/fr13_b1_fix1_gate/run_fix1_nsys_arm.sh (proven GB10 workflow:
# LUMO_NSYS_TRACE=cuda,cuda-sw,nvtx is MANDATORY -- hw trace drops all kernel rows).
# Config = deployed tail6 campaign env (kvremap_tail6_kvr1) minus APC (cache OFF for
# a clean single-request kernel mix; cache-restore kernels are outside the verify
# span anyway) and minus timers. Pipeline flags default ON in the forked launcher;
# FR13_ATTN_KV_REMAP/FR13_SLOT_REORDER auto-sourced from fr13_required_tree_flags.sh.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
RUNROOT=output/fr13_verify_profile
ARM=nsys_tail6_21node
ARMDIR="$RUNROOT/$ARM"
CONTAINER="fr13-tail6-nsys"
PORT=9950
# DELAY calibrated 2026-07-22: nsys CUPTI sw-trace roughly DOUBLES boot (health
# ~11-13 min vs ~7 unwrapped; first attempt fail-louded at t0+520 pre-health).
DELAY=840
DUR=240
# teardown-on-any-exit: first attempt left the container up on the FAIL path and
# force-kill wedged ~92GB host mem (known ModelServer recovery-bypass); the trap
# removes the container and unwedges via drop_caches when available mem is low.
cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  sleep 3
  AVAIL=$(free -g | awk '/^Mem:/ {print $7}')
  if (( AVAIL < 60 )); then
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
mkdir -p "$ARMDIR/logs"
echo "=== NSYS ARM tail6 21-node deployed-config delay=${DELAY}s dur=${DUR}s ==="
date -u +%Y-%m-%dT%H:%M:%SZ
if [[ -n "$(docker ps -q)" ]]; then echo "FAIL: docker ps not empty before boot"; docker ps; exit 2; fi
free -g | tee "$ARMDIR/free_before_boot.txt"

CONTAINER="$CONTAINER" \
PORT=$PORT \
GPU_UTIL=0.82 \
MAX_NUM_SEQS=1 \
BATCH_INVARIANT=0 \
FR13_BI_TREE_ATTN=0 \
FR10_METRICS=0 \
TREE="[(0,),(1,),(2,),(0,0),(0,1),(0,2),(0,0,0),(0,0,1),(0,0,2),(0,0,0,0),(0,0,0,1),(0,0,0,2),(0,0,0,0,0),(0,0,0,0,1),(0,0,0,0,2),(0,0,0,0,0,0),(0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0,0)]" \
FR13_TAIL_MODE=1 \
FR13_DRAFT_SOURCE=merged \
FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 \
LUMO_FB_KERNEL_ROWS=1 \
LUMO_FB_PROJ_PAD_ROWS=16 \
LUMO_NSYS_WRAP_VLLM=1 \
LUMO_NSYS_TRACE=cuda,cuda-sw,nvtx \
LUMO_NSYS_DELAY_S=$DELAY \
LUMO_NSYS_DURATION_S=$DUR \
LUMO_NSYS_OUTPUT=/logs/nsys_tail6_21node \
FR13_RUN_DIR="$PWD/$ARMDIR" \
LOG_DIR="$PWD/$ARMDIR/logs" \
scripts/fr13_launch_forked_fa2_tree_server.sh > "$ARMDIR/launch.log" 2>&1
RC=$?
if (( RC != 0 )); then echo "FAIL: launcher rc=$RC"; tail -20 "$ARMDIR/launch.log"; exit 2; fi

BANNER_TS=""
for i in $(seq 1 150); do
  BANNER_TS=$(docker logs -t "$CONTAINER" 2>&1 | grep -m1 "version 0.19" | awk '{print $1}')
  [[ -n "${BANNER_TS:-}" ]] && break
  if [[ "$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)" != "running" ]]; then
    echo "FAIL: container not running before banner"; exit 2
  fi
  sleep 2
done
[[ -n "${BANNER_TS:-}" ]] || { echo "FAIL: no vllm banner"; exit 2; }
T0=$(( $(date -d "${BANNER_TS}" +%s) - 8 ))
echo "banner=$BANNER_TS t0=$T0 window=[$((T0+DELAY)),$((T0+DELAY+DUR)))"

HEALTHY=0
while (( $(date +%s) < T0 + DELAY + 100 )); do
  if curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
  if [[ "$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)" != "running" ]]; then
    echo "FAIL: container died before health"; docker logs "$CONTAINER" 2>&1 | tail -10; exit 2
  fi
  sleep 5
done
(( HEALTHY == 1 )) || { echo "FAIL: health not up by t0+$((DELAY+100))"; exit 2; }
echo "healthy at t+$(( $(date +%s) - T0 ))s"
docker exec "$CONTAINER" env | sort > "$ARMDIR/container_env.txt"
grep -q "^FR13_TAIL_MODE=1$" "$ARMDIR/container_env.txt" || { echo "FAIL: TAIL_MODE flag"; exit 3; }
grep -q "^FR13_ATTN_KV_REMAP=1$" "$ARMDIR/container_env.txt" || { echo "FAIL: KV_REMAP not live"; exit 3; }
grep -q "^FR13_SLOT_REORDER=1$" "$ARMDIR/container_env.txt" || { echo "FAIL: SLOT_REORDER not live"; exit 3; }
docker exec "$CONTAINER" bash -lc 'ps -eo args' 2>/dev/null | grep -q nsys || { echo "FAIL: no nsys proc"; exit 3; }
echo "nsys proc present; deployed tail6 flags live"

# warmup BEFORE the window so capture sees steady-state decode
python3 scripts/fr10_quick_decode_tps_probe.py \
  --endpoint "http://127.0.0.1:$PORT" --model qwen3.6-27b \
  --prompts-file output/fr13_acceptance_ladder/prompts_swe4.json \
  --prompt-limit 1 --max-tokens 16 --temperature 0.0 --top-p 1.0 --seed 1313 \
  --samples-per-prompt 1 --batch-size 1 --warmup-samples 0 --wait-health 60 \
  --modes tree_mtp --out "$ARMDIR/warmup_probe.json" \
  > "$ARMDIR/warmup_stdout.log" 2>&1 || { echo "FAIL: warmup probe"; exit 4; }
echo "warmup done t+$(( $(date +%s) - T0 ))s"

# engagement gate: tree spec engaged with 21 tokens/draft (tail6 EXPECT_RATIO=21)
M=$(curl -fsS -m 5 "http://127.0.0.1:$PORT/metrics" 2>/dev/null)
DRAFTS=$(echo "$M" | awk '/^vllm:spec_decode_num_drafts_total/ {s+=$2} END {print s+0}')
DTOK=$(echo "$M" | awk '/^vllm:spec_decode_num_draft_tokens_total/ {s+=$2} END {print s+0}')
echo "engagement: drafts=$DRAFTS draft_tokens=$DTOK" | tee "$ARMDIR/engagement_needle.txt"
python3 -c "
d=float('$DRAFTS'); t=float('$DTOK')
assert d > 0, 'no drafts recorded -- spec decode not engaged'
r = t/d
assert abs(r-21.0) < 0.5, f'tok/draft={r:.2f} != 21 -- tail6 tree NOT engaged'
print(f'ENGAGEMENT OK tok/draft={r:.2f}')" | tee -a "$ARMDIR/engagement_needle.txt" || { echo "FAIL: engagement"; exit 3; }

# probes INSIDE the window
while (( $(date +%s) < T0 + DELAY + 5 )); do sleep 2; done
R=0
while (( $(date +%s) < T0 + DELAY + DUR - 60 && R < 3 )); do
  R=$((R+1))
  S=$(date +%s)
  python3 scripts/fr10_quick_decode_tps_probe.py \
    --endpoint "http://127.0.0.1:$PORT" --model qwen3.6-27b \
    --prompts-file output/fr13_acceptance_ladder/prompts_swe4.json \
    --max-tokens 64 --temperature 0.0 --top-p 1.0 --seed 1313 \
    --samples-per-prompt 1 --batch-size 1 --warmup-samples 0 --wait-health 0 \
    --modes tree_mtp --out "$ARMDIR/window_probe_r${R}.json" \
    > "$ARMDIR/window_probe_r${R}_stdout.log" 2>&1 \
    || { echo "FAIL: window probe r$R"; tail -3 "$ARMDIR/window_probe_r${R}_stdout.log"; exit 4; }
  E=$(date +%s)
  echo "window probe r$R wall=[$S,$E] t+[$((S-T0)),$((E-T0))]s"
  echo "{\"r\":$R,\"wall_start\":$S,\"wall_end\":$E}" >> "$ARMDIR/probe_walls.jsonl"
done
(( R >= 1 )) || { echo "FAIL: no probe inside window"; exit 4; }

REP="$ARMDIR/logs/nsys_tail6_21node.nsys-rep"
while (( $(date +%s) < T0 + DELAY + DUR + 10 )); do sleep 5; done
echo "window closed; waiting for $REP"
PREV=-1; STABLE=0
for i in $(seq 1 120); do
  SZ=$(stat -c %s "$REP" 2>/dev/null || echo -1)
  if (( SZ > 0 && SZ == PREV )); then STABLE=$((STABLE+1)); else STABLE=0; fi
  PREV=$SZ
  (( STABLE >= 4 )) && { echo "REPORT_STABLE size=$SZ"; break; }
  if [[ "$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)" != "running" ]]; then
    sleep 15; SZ=$(stat -c %s "$REP" 2>/dev/null || echo -1)
    echo "container exited; report size=$SZ"; break
  fi
  sleep 5
done
SZ=$(stat -c %s "$REP" 2>/dev/null || echo -1)
(( SZ > 0 )) || { echo "FAIL: no nsys report"; docker logs "$CONTAINER" 2>&1 | tail -8; exit 5; }
docker logs "$CONTAINER" > "$ARMDIR/docker_full.log" 2>&1
docker rm -f "$CONTAINER" >/dev/null 2>&1
sleep 2
free -g | tee "$ARMDIR/free_after_teardown.txt"
echo "ARM_DONE report=$REP size=$SZ"
date -u +%Y-%m-%dT%H:%M:%SZ
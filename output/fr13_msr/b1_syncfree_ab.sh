#!/usr/bin/env bash
# B=1 CONTROLLED syncfree A/B (methodology: B=1 = controlled speed arm;
# B=4 16-task = behavior + final speed verdict — never promote B=1 numbers
# to deployment claims). Two boots, ONLY delta = FR13_KV_REMAP_SYNCFREE
# (patch-time baked from env). Fixed probe workload (MAX_NUM_SEQS=1, swe4
# prompts, temp 0.6 — committer path needs temp>0), all committer timers
# (observer-effect-audited). ~25min/boot.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
RUNROOT=output/fr13_msr
PORT=9950

cleanup() {
  docker ps -q --filter name=fr13-b1ab | xargs -r docker rm -f >/dev/null 2>&1 || true
  sleep 3
  AVAIL=$(free -g | awk '/^Mem:/ {print $7}')
  if (( AVAIL < 60 )); then sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT

run_arm() {
  local ARM="$1" SYNCFREE="$2"
  local ARMDIR="$RUNROOT/b1ab_${ARM}"
  local CONTAINER="fr13-b1ab-${ARM}"
  mkdir -p "$ARMDIR/logs"
  echo "=== ARM $ARM (KV_REMAP_SYNCFREE=$SYNCFREE) $(date -u +%H:%M:%SZ) ==="
  if [[ -n "$(docker ps -q --filter name=fr13)" ]]; then echo "FAIL: fr13 container running"; return 2; fi

  CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=0.70 MAX_NUM_SEQS=1 \
  BATCH_INVARIANT=0 FR13_BI_TREE_ATTN=0 FR10_METRICS=0 \
  TREE="[(0,),(1,),(2,),(0,0),(0,1),(0,2),(0,0,0),(0,0,1),(0,0,2),(0,0,0,0),(0,0,0,1),(0,0,0,2),(0,0,0,0,0),(0,0,0,0,1),(0,0,0,0,2),(0,0,0,0,0,0),(0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0,0)]" \
  FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 \
  LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16 \
  FR13_ENABLE_APC=1 MAMBA_BLOCK_SIZE=1024 APC_BLOCK_SIZE=1024 \
  MAMBA_SSM_CACHE_DTYPE=float32 \
  FR13_KV_REMAP_SYNCFREE="$SYNCFREE" \
  FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
  FR13_CFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/b1ab_${ARM}_cfwd.json \
  FR13_COMMITTER_SG_TIMER=1 \
  FR13_COMMITTER_SG_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/b1ab_${ARM}_sg.json \
  FR13_REPLAY_ONLY_GPU_TIMER=1 \
  FR13_REPLAY_ONLY_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/b1ab_${ARM}_replayonly.json \
  FR13_KVREMAP_TIMER=1 \
  FR13_KVREMAP_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/b1ab_${ARM}_kvremap.json \
  FR13_RUN_DIR="$PWD/$ARMDIR" LOG_DIR="$PWD/$ARMDIR/logs" \
  scripts/fr13_launch_forked_fa2_tree_server.sh > "$ARMDIR/launch.log" 2>&1 || {
    echo "FAIL: launcher rc=$?"; tail -12 "$ARMDIR/launch.log"; return 2; }

  local T0=$(date +%s) HEALTHY=0
  while (( $(date +%s) < T0 + 900 )); do
    curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { HEALTHY=1; break; }
    [[ "$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)" == "running" ]] || {
      echo "FAIL: died in boot"; docker logs "$CONTAINER" 2>&1 | tail -12; return 2; }
    sleep 10
  done
  (( HEALTHY )) || { echo "FAIL: no health"; return 2; }
  echo "healthy at +$(( $(date +%s) - T0 ))s"

  for R in 1 2 3 4; do
    python3 scripts/fr10_quick_decode_tps_probe.py \
      --endpoint "http://127.0.0.1:$PORT" --model qwen3.6-27b \
      --prompts-file output/fr13_acceptance_ladder/prompts_swe4.json \
      --max-tokens 128 --temperature 0.6 --top-p 1.0 --seed 1313 \
      --samples-per-prompt 1 --batch-size 1 --warmup-samples 0 --wait-health 60 \
      --modes tree_mtp --out "$ARMDIR/probe_r${R}.json" \
      > "$ARMDIR/probe_r${R}_stdout.log" 2>&1 || { echo "FAIL: probe r$R"; return 4; }
  done
  # Running dist for the batching label (expect R=1 ~100%)
  docker logs "$CONTAINER" 2>&1 | grep -oE "Running: [0-9]+ reqs" | \
    awk '{n[$2]++; t++} END {printf "Running dist: "; for (k=0;k<=4;k++) if (n[k]) printf "R=%d:%.0f%% ", k, 100*n[k]/t; print ""}'
  docker logs "$CONTAINER" 2>&1 | grep -cE "Assertion|RuntimeError" || true
  docker rm -f "$CONTAINER" >/dev/null 2>&1
  sleep 5
  echo "ARM $ARM done"
}

run_arm syncfree 1 || exit 2
run_arm legacy   0 || exit 2

python3 - <<'EOF'
import json, glob
print("=== B=1 controlled syncfree A/B ===")
for arm in ("syncfree", "legacy"):
    row = {}
    for tag, ks, kn in (("cfwd","gpu_seconds","n_spans"), ("sg","sg_gpu_seconds","n_sg"),
                        ("replayonly","replay_only_gpu_seconds","n_calls"), ("kvremap","span_gpu_seconds","n")):
        fs = sorted(glob.glob(f"output/fr13_sfwd_sidecar/b1ab_{arm}_{tag}.json*"))
        if fs:
            j = json.load(open(fs[-1]))
            row[tag] = f"{1000*j[ks]/j[kn]:.1f}ms(n={j[kn]})" if j[kn] else "0"
    print(arm, row)
print("A/B basis: B=1, fixed probes, temp 0.6; deployment claims require the batched 16-task gate")
EOF
echo "B1AB_DONE $(date -u +%H:%M:%SZ)"

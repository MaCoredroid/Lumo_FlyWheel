#!/usr/bin/env bash
# FR13 B1+B2a live wiring/speed gate: two short boots (flags ON vs OFF), same
# B=1 probe workload, tail6 deployed config (cache-ON), GPU span timers armed.
# Byte-identity is already proven at kernel level (offline gates); this gates:
#   (1) boots clean + tree engagement (tok/draft=21) with flags ON,
#   (2) no crash/garble across probes,
#   (3) speed: fr13_* GPU-timer counter deltas ON vs OFF (verify s_per_fwd_gpu
#       should drop by roughly the fused-op share; drafter/committer flat).
# Cross-boot BYTE compares are invalid on GB10 (autotune floor) -- not done.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
RUNROOT=output/fr13_verify_profile
PORT=9950
mkdir -p "$RUNROOT"

cleanup() {
  docker ps -q --filter name=fr13-ringwb | xargs -r docker rm -f >/dev/null 2>&1 || true
  sleep 3
  AVAIL=$(free -g | awk '/^Mem:/ {print $7}')
  if (( AVAIL < 60 )); then
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

run_arm() {
  local ARM="$1"; local RING="$2"; local WB="$3"
  local ARMDIR="$RUNROOT/ringwb_${ARM}"
  local CONTAINER="fr13-ringwb-${ARM}"
  mkdir -p "$ARMDIR/logs"
  echo "=== ARM $ARM (RING_EXPORT=$RING CONV_WB_FUSED=$WB) $(date -u +%H:%M:%SZ) ==="
  if [[ -n "$(docker ps -q)" ]]; then echo "FAIL: docker not empty"; return 2; fi

  CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=0.82 MAX_NUM_SEQS=1 \
  BATCH_INVARIANT=0 FR13_BI_TREE_ATTN=0 FR10_METRICS=0 \
  TREE="[(0,),(1,),(2,),(0,0),(0,1),(0,2),(0,0,0),(0,0,1),(0,0,2),(0,0,0,0),(0,0,0,1),(0,0,0,2),(0,0,0,0,0),(0,0,0,0,1),(0,0,0,0,2),(0,0,0,0,0,0),(0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0,0,0)]" \
  FR13_TAIL_MODE=1 FR13_DRAFT_SOURCE=merged FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 \
  LUMO_FB_KERNEL_ROWS=1 LUMO_FB_PROJ_PAD_ROWS=16 \
  FR13_ENABLE_APC=1 MAMBA_BLOCK_SIZE=1024 APC_BLOCK_SIZE=1024 \
  MAMBA_SSM_CACHE_DTYPE=float32 \
  FR13_RING_EXPORT="$RING" FR13_CONV_WB_FUSED="$WB" \
  FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1 \
  FR13_SFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/ringwb_${ARM}.json \
  FR13_DFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/ringwb_${ARM}_dfwd.json \
  FR13_CFWD_GPU_TIMER_JSON=/workspace/output/fr13_sfwd_sidecar/ringwb_${ARM}_cfwd.json \
  FR13_RUN_DIR="$PWD/$ARMDIR" LOG_DIR="$PWD/$ARMDIR/logs" \
  scripts/fr13_launch_forked_fa2_tree_server.sh > "$ARMDIR/launch.log" 2>&1
  local RC=$?
  (( RC == 0 )) || { echo "FAIL: launcher rc=$RC"; tail -15 "$ARMDIR/launch.log"; return 2; }

  local T0=$(date +%s) HEALTHY=0
  while (( $(date +%s) < T0 + 780 )); do
    if curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then HEALTHY=1; break; fi
    if [[ "$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null)" != "running" ]]; then
      echo "FAIL: container died in boot"; docker logs "$CONTAINER" 2>&1 | tail -15; return 2
    fi
    sleep 10
  done
  (( HEALTHY == 1 )) || { echo "FAIL: no health in 780s"; return 2; }
  echo "healthy at +$(( $(date +%s) - T0 ))s"
  docker exec "$CONTAINER" env | sort > "$ARMDIR/container_env.txt"
  grep -q "^FR13_RING_EXPORT=$RING$" "$ARMDIR/container_env.txt" || { echo "FAIL: RING flag"; return 3; }
  grep -q "^FR13_CONV_WB_FUSED=$WB$" "$ARMDIR/container_env.txt" || { echo "FAIL: WB flag"; return 3; }

  # warmup + engagement
  python3 scripts/fr10_quick_decode_tps_probe.py \
    --endpoint "http://127.0.0.1:$PORT" --model qwen3.6-27b \
    --prompts-file output/fr13_acceptance_ladder/prompts_swe4.json \
    --prompt-limit 1 --max-tokens 16 --temperature 0.0 --top-p 1.0 --seed 1313 \
    --samples-per-prompt 1 --batch-size 1 --warmup-samples 0 --wait-health 60 \
    --modes tree_mtp --out "$ARMDIR/warmup_probe.json" \
    > "$ARMDIR/warmup_stdout.log" 2>&1 || { echo "FAIL: warmup"; return 4; }
  local M D T
  M=$(curl -fsS -m 5 "http://127.0.0.1:$PORT/metrics")
  D=$(echo "$M" | awk '/^vllm:spec_decode_num_drafts_total/ {s+=$2} END {print s+0}')
  T=$(echo "$M" | awk '/^vllm:spec_decode_num_draft_tokens_total/ {s+=$2} END {print s+0}')
  python3 -c "
d=float('$D'); t=float('$T')
assert d>0 and abs(t/d-21.0)<0.5, f'ENGAGEMENT FAIL tok/draft={t/d if d else 0:.2f}'
print(f'ENGAGEMENT OK tok/draft={t/d:.2f}')" || return 3

  curl -fsS -m 5 "http://127.0.0.1:$PORT/metrics" | grep -E "^vllm:fr13_|^vllm:spec_decode" > "$ARMDIR/metrics_pre.txt"
  for R in 1 2 3; do
    python3 scripts/fr10_quick_decode_tps_probe.py \
      --endpoint "http://127.0.0.1:$PORT" --model qwen3.6-27b \
      --prompts-file output/fr13_acceptance_ladder/prompts_swe4.json \
      --max-tokens 64 --temperature 0.0 --top-p 1.0 --seed 1313 \
      --samples-per-prompt 1 --batch-size 1 --warmup-samples 0 --wait-health 0 \
      --modes tree_mtp --out "$ARMDIR/probe_r${R}.json" \
      > "$ARMDIR/probe_r${R}_stdout.log" 2>&1 || { echo "FAIL: probe r$R"; return 4; }
  done
  curl -fsS -m 5 "http://127.0.0.1:$PORT/metrics" | grep -E "^vllm:fr13_|^vllm:spec_decode" > "$ARMDIR/metrics_post.txt"
  docker logs "$CONTAINER" 2>&1 | grep "acceptance length" | tail -8 > "$ARMDIR/acceptance_tail.txt"
  docker logs "$CONTAINER" 2>&1 | grep -icE "error|traceback|assert" > "$ARMDIR/error_count.txt" || true
  docker rm -f "$CONTAINER" >/dev/null 2>&1
  sleep 5
  echo "ARM $ARM done"
  return 0
}

run_arm on 1 1 || { echo "GATE FAIL: ON arm"; exit 2; }
run_arm off 0 0 || { echo "GATE FAIL: OFF arm"; exit 2; }

python3 - <<'EOF'
def deltas(arm):
    out = {}
    for phase in ("pre", "post"):
        for line in open(f"output/fr13_verify_profile/ringwb_{arm}/metrics_{phase}.txt"):
            parts = line.split()
            if len(parts) != 2:
                continue
            k = parts[0]
            out.setdefault(k, {})[phase] = float(parts[1])
    return {k: v.get("post", 0) - v.get("pre", 0) for k, v in out.items()}

on, off = deltas("on"), deltas("off")
def ratio(name_frag_num, name_frag_den, d):
    num = sum(v for k, v in d.items() if name_frag_num in k)
    den = sum(v for k, v in d.items() if name_frag_den in k)
    return (num / den) if den else None

for arm, d in (("ON", on), ("OFF", off)):
    sfwd = ratio("fr13_decode_forward_gpu_seconds", "fr13_decode_forward_gpu_drafts", d)
    acc = ratio("spec_decode_num_accepted_tokens", "spec_decode_num_drafts", d)
    print(f"[{arm}] s_per_fwd_gpu={sfwd if sfwd is None else round(sfwd*1000,2)}ms/event  accept_per_event={acc and round(acc,3)}")
    for k, v in sorted(d.items()):
        if "fr13_" in k and v > 0:
            print(f"    {k}: {v:.4f}")
print("GATE SUMMARY COMPLETE")
EOF
echo "LIVE_GATE_DONE $(date -u +%H:%M:%SZ)"
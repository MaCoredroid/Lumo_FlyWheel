#!/usr/bin/env bash
# FR13 — behavioral test of the TWO real garble suspects (diffmap wf_e41b2cf1, 2026-07-11):
#   PRIME: conv1d prior-window ADDRESSING (default reads accepted-leaf NODE column; native slides col-0
#          running row). At num_accepted>1 = physically different bank rows -> wrong conv input to every
#          downstream GDN layer. A/B knob: FR12_TREE_CONV_NATIVE_PRIOR_READ=1 (force native col-0 read).
#   #2:    scan KERNEL GEOMETRY (ours num_warps=8/BV=16 vs native num_warps=4/BV=32/num_stages=3 ->
#          ~1-ULP/token reduction-order diff, compounds over 48 layers). Knob: FR13_TREE_GDN_GEOM_OVERRIDE.
# SCAN_ALIGN was REFUTED (10.27%, aimed at wrong oracle packed_decode not spec fused_sigmoid_gating).
# Same-session 3-arm A/B on the temp-0.6 identifier-bait garble gate (native=0%, tree-default=8-11%).
#   conv or geom -> ~0% => compute-only no-HBM FIX found. Both still 8-11% => redirect to real-task math-diff gate.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
PORT=9955; N=${N:-24}; CONC=${CONC:-4}
CAT8='[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,0,0)]'
TS=$(date -u +%Y%m%dT%H%M%SZ); RUN=output/fr13_conv_geom_garble/run_$TS; mkdir -p "$RUN"
echo "=== conv-addr vs geom garble gate @ temp0.6 -> $RUN ==="

boot_and_gate() {  # $1=arm  $2..=extra "K=V" env
  local ARM=$1; shift; local C=fr13-garble-cg-$ARM
  echo "--- [$ARM] boot cat8 tree  env: $* @ $(date -u +%H:%M:%S) ---"
  docker ps -aq --filter "name=fr13" | xargs -r docker rm -f >/dev/null 2>&1 || true
  PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
  env CONTAINER=$C PORT=$PORT GPU_UTIL=0.8 MAX_NUM_SEQS=8 \
    ATTENTION_BACKEND=TREE_ATTN \
    SPEC_CONFIG='{"method":"qwen3_5_mtp","num_speculative_tokens":8,"speculative_token_tree":"'"$CAT8"'"}' \
    GPU_GUARD_FLOOR_MIB=3000 "$@" \
    bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$RUN/boot_$ARM.log" 2>&1 &
  local LPID=$! T0=$SECONDS OK=0
  while [ $((SECONDS-T0)) -lt 720 ]; do
    curl -fsS -m5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { OK=1; break; }
    [[ -n "$(docker ps -aq -f name=$C -f status=exited)" ]] && { echo "[$ARM] exited"; break; }
    sleep 10
  done
  [ "$OK" = 1 ] || { echo "FAIL [$ARM] not healthy"; tail -25 "$RUN/boot_$ARM.log"; docker rm -f "$C" >/dev/null 2>&1; return 1; }
  # fail-loud: confirm the intended flag actually engaged in the worker
  echo "  [$ARM] engaged: CONV_NATIVE=$(docker exec $C printenv FR12_TREE_CONV_NATIVE_PRIOR_READ 2>/dev/null||echo -) GEOM=$(docker exec $C printenv FR13_TREE_GDN_GEOM_OVERRIDE 2>/dev/null||echo -)"
  local MODEL=$(curl -fsS -m5 "http://127.0.0.1:$PORT/v1/models" | .venv/bin/python -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  .venv/bin/python scripts/fr13_garble_gate.py run --endpoint "http://127.0.0.1:$PORT/v1" \
    --model "$MODEL" --arm "$ARM" --n "$N" --concurrency "$CONC" --out "$RUN/$ARM.jsonl" > "$RUN/gate_$ARM.log" 2>&1
  echo "  [$ARM] SCORE: $(.venv/bin/python scripts/fr13_garble_gate.py score --samples "$RUN/$ARM.jsonl" 2>&1 | tee "$RUN/${ARM}_score.txt" | head -1)"
  docker rm -f "$C" >/dev/null 2>&1 || true; wait $LPID 2>/dev/null || true
}

# default already measured this session at 9.62% (same-session baseline); re-run just the 2 flag arms
boot_and_gate conv_native  FR12_TREE_CONV_NATIVE_PRIOR_READ=1
boot_and_gate geom_native  FR13_TREE_GDN_GEOM_OVERRIDE=BV=32,num_warps=4,num_stages=3

echo ""; echo "=== CONV/GEOM garble gate SUMMARY (temp 0.6, undefined-name rate) ==="
for a in default conv_native geom_native; do echo "  $a: $(grep -oE 'undefined-name-rate=[0-9.]+%|samples-with-undef=[0-9]+/[0-9]+' "$RUN/${a}_score.txt" 2>/dev/null|tr '\n' ' ')"; done
echo "  (native=0.00%; SCAN_ALIGN refuted=10.27%)"
echo "  VERDICT: an arm at ~0% => that op is the birthplace + compute-only no-HBM fix; both ~8-11% => neither, go real-task math-diff gate"
echo "=== DONE $RUN @ $(date -u +%H:%M:%S)Z ==="

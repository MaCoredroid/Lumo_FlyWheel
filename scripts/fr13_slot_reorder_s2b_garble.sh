#!/bin/bash
# FR13_SLOT_REORDER S2b — garble gate (G1) for the slot-reorder fix.
# 3 arms, GPU-solo sequential, identical prompts+seeds (temp 0.6 inside the gate):
#   native   : FLASH_ATTN naive mtp5 (clean baseline)
#   treectrl : TREE_ATTN cat8 + REMAP, FR13_SLOT_REORDER=0 (fix OFF control)
#   treefix  : TREE_ATTN cat8 + REMAP, FR13_SLOT_REORDER=1 (the fix)
# PASS: treefix undefined-name rate <= native's (and ~= treectrl's, both ~0 since
# the FR13_ATTN_KV_REMAP garble fix). Mirrors fr13_garble_gate_shot.sh boot flow.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
RUNDIR=output/fr13_slot_reorder/s2b_garble_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUNDIR"; PORT=9950; N=${N:-30}
CAT8_TREE='[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,0,0)]'
[[ -z "$(docker ps -q --filter name=fr13 2>/dev/null)" ]] || { echo "ABORT: fr13 container running"; exit 1; }

boot_run () {   # arm container ATTN SPEC extra_env...
  local arm=$1 C=$2 ATTN=$3 SPEC=$4; shift 4
  echo "=== [s2b] boot $arm ($ATTN) extra=[$*] ==="
  docker rm -f "$C" 2>/dev/null || true
  env "$@" CONTAINER="$C" PORT=$PORT GPU_UTIL=0.82 MAX_NUM_SEQS=4 \
    ATTENTION_BACKEND="$ATTN" SPEC_CONFIG="$SPEC" \
    bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$RUNDIR/launch_$arm.log" 2>&1 &
  local LPID=$!
  for i in $(seq 1 400); do
    curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    [[ -n "$(docker ps -aq -f name=$C -f status=exited)" ]] && { echo "FAIL $arm boot"; tail -25 "$RUNDIR/launch_$arm.log"; return 2; }
    sleep 3
  done
  curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "FAIL $arm /health"; return 2; }
  # engagement audit per arm (fail-loud, not vacuous)
  docker logs "$C" 2>&1 | grep -o "FR13_SLOT_REORDER ENGAGED ([a-z_ ]*): tree_n=[0-9]* pi=\[[^]]*\]" | sort -u > "$RUNDIR/engage_$arm.txt"
  echo "  engagement lines: $(wc -l < "$RUNDIR/engage_$arm.txt")"
  .venv/bin/python scripts/fr13_garble_gate.py run --endpoint "http://127.0.0.1:$PORT/v1" \
    --model qwen3.6-27b --arm "$arm" --n "$N" --out "$RUNDIR/$arm.jsonl" 2>&1 | tee "$RUNDIR/score_$arm.txt"
  # post-run engagement recheck (reorder must have PERMUTED during generation)
  docker logs "$C" 2>&1 | grep -o "FR13_SLOT_REORDER ENGAGED ([a-z_ ]*): tree_n=[0-9]* pi=\[[^]]*\]" | sort -u > "$RUNDIR/engage_post_$arm.txt"
  docker rm -f "$C" 2>/dev/null; wait $LPID 2>/dev/null || true; sleep 5
}

boot_run native   fr13-s2b-native   FLASH_ATTN '{"method":"qwen3_5_mtp","num_speculative_tokens":5}'
boot_run treectrl fr13-s2b-treectrl TREE_ATTN  '{"method":"qwen3_5_mtp","num_speculative_tokens":8,"speculative_token_tree":"'"$CAT8_TREE"'"}' \
  FR13_ATTN_KV_REMAP=1 FR13_SLOT_REORDER=0
boot_run treefix  fr13-s2b-treefix  TREE_ATTN  '{"method":"qwen3_5_mtp","num_speculative_tokens":8,"speculative_token_tree":"'"$CAT8_TREE"'"}' \
  FR13_ATTN_KV_REMAP=1 FR13_SLOT_REORDER=1

echo "=== [s2b] VERDICT ==="
grep -H "undefined-name-rate" "$RUNDIR"/score_*.txt 2>/dev/null
echo "--- treefix engagement (MUST have runner+bias lines) ---"
cat "$RUNDIR/engage_post_treefix.txt" 2>/dev/null
echo "--- treectrl engagement (MUST be empty) ---"
cat "$RUNDIR/engage_post_treectrl.txt" 2>/dev/null
echo "RUNDIR=$RUNDIR"

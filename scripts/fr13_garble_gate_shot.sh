#!/bin/bash
# FR13 G1 GARBLE GATE shot — measures near-neighbor code-token corruption per arm.
# Boots each arm's server (GPU-solo, sequential), runs scripts/fr13_garble_gate.py (identifier-
# consistency probes + AST undefined-name scorer), compares. Tree FAILS iff undefined-name rate >
# native's. This is the fix-iteration gate (G1) from FR13_TREE_GARBLE_GATE_AND_FIX.md.
# Boot pattern mirrors fr13_serialization_shot.sh (kept in sync; run AFTER the serialization shot
# validates the boot). GPU-SOLO: only when GPU free.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
RUNDIR=output/fr13_garble_gate; mkdir -p "$RUNDIR"; PORT=9950; N=${N:-30}
CAT6ROOT_TREE='[(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]'
if [[ -n "$(docker ps -q --filter name=fr13 2>/dev/null)" ]]; then echo "ABORT: fr13 container running (GPU-solo)"; exit 1; fi

boot_run () {   # arm_label  container  ATTENTION  SPEC_CONFIG
  local arm=$1 C=$2 ATTN=$3 SPEC=$4
  echo "=== [garble] boot $arm ($ATTN) ==="
  docker rm -f "$C" 2>/dev/null || true
  CONTAINER="$C" PORT=$PORT GPU_UTIL=0.82 MAX_NUM_SEQS=4 \
    ATTENTION_BACKEND="$ATTN" SPEC_CONFIG="$SPEC" \
    bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$RUNDIR/launch_$arm.log" 2>&1 &
  local LPID=$!
  for i in $(seq 1 400); do
    curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    [[ -n "$(docker ps -aq -f name=$C -f status=exited)" ]] && { echo "FAIL $arm boot"; tail -25 "$RUNDIR/launch_$arm.log"; return 2; }
    sleep 3
  done
  curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "FAIL $arm /health"; return 2; }
  echo "=== [garble] run gate on $arm (N=$N) ==="
  .venv/bin/python scripts/fr13_garble_gate.py run --endpoint "http://127.0.0.1:$PORT/v1" \
    --model qwen3.6-27b --arm "$arm" --n "$N" --out "$RUNDIR/$arm.jsonl" 2>&1 | tee "$RUNDIR/score_$arm.txt"
  docker rm -f "$C" 2>/dev/null; wait $LPID 2>/dev/null || true; sleep 5
}

# ARM 1: native-MTP-decode (FLASH_ATTN naive_mtp) = the clean baseline (flash_ns5_nocache config)
boot_run native fr13-garble-native FLASH_ATTN '{"method":"qwen3_5_mtp","num_speculative_tokens":5}'
# ARM 2: tree (cat6root, TREE_ATTN)
boot_run tree   fr13-garble-tree   TREE_ATTN   '{"method":"qwen3_5_mtp","num_speculative_tokens":6,"speculative_token_tree":"'"$CAT6ROOT_TREE"'"}'

echo "=== [garble] VERDICT: native vs tree undefined-name rate ==="
grep -H undefined-name-rate "$RUNDIR"/score_*.txt 2>/dev/null
.venv/bin/python -c "
import re
def rate(f):
    try: return float(re.search(r'undefined-name-rate=([0-9.]+)%', open(f).read()).group(1))
    except: return None
nat=rate('$RUNDIR/score_native.txt'); tre=rate('$RUNDIR/score_tree.txt')
print(f'  native={nat}%  tree={tre}%')
if nat is not None and tre is not None:
    print('  GARBLE CONFIRMED (tree > native): tree corrupts code identifiers more' if tre>nat+0.5 else '  no garble delta (tree <= native) — reconsider')
"
echo "=== [garble] DONE -> $RUNDIR. Add no-spec ground-truth arm + fix arms (M-invariance) next. ==="

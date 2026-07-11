#!/usr/bin/env bash
# FR13 — does SCAN_ALIGN=body (K1 bf16-store-boundary alignment) reduce the GARBLE at temp 0.6?
# ============================================================================
# The fixed-N_PAD scan probe (2026-07-11) confirmed: intra-request co-residency is M-invariant (0.0);
# the ONLY residual tree-vs-native diff is the carried STATE at 1 bf16 ULP = the K1 store-boundary seam
# (fr10_gdn_tree_kernel.py:568). SCAN_ALIGN=body reproduces native's bf16 state carry (compute-only,
# NO HBM tax). Prior live tests of it were TEMP 0.0 (greedy) -> undercount this garble (near-neighbor is
# only sampled at temp>0). This gates K1 in the CORRECT regime with the identifier-bait garble gate that
# already showed tree=8-11% / native=0%. SAME-SESSION A/B (two boots back-to-back) to control boot-noise.
#   K1 -> ~0%  => compute-only no-HBM fix FOUND. K1 still 8-11% => K1 insufficient (cross-basis/committer).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
PORT=9955; N=${N:-24}; CONC=${CONC:-4}
CAT8='[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,0,0)]'
TS=$(date -u +%Y%m%dT%H%M%SZ); RUN=output/fr13_scan_align_garble/run_$TS; mkdir -p "$RUN"
echo "=== SCAN_ALIGN=body vs OFF garble gate @ temp0.6  -> $RUN ==="

boot_and_gate() {  # $1=arm_label  $2=SCAN_ALIGN(0/1)
  local ARM=$1 SA=$2 C=fr13-garble-sa-$1
  echo "--- [$ARM] boot cat8 tree SCAN_ALIGN=$SA @ $(date -u +%H:%M:%S) ---"
  docker ps -aq --filter "name=fr13" | xargs -r docker rm -f >/dev/null 2>&1 || true
  PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
  CONTAINER=$C PORT=$PORT GPU_UTIL=0.8 MAX_NUM_SEQS=8 \
    ATTENTION_BACKEND=TREE_ATTN \
    SPEC_CONFIG='{"method":"qwen3_5_mtp","num_speculative_tokens":8,"speculative_token_tree":"'"$CAT8"'"}' \
    FR13_SCAN_ALIGN=$SA FR13_SCAN_ALIGN_MODE=body \
    GPU_GUARD_FLOOR_MIB=3000 \
    bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$RUN/boot_$ARM.log" 2>&1 &
  local LPID=$!
  local T0=$SECONDS OK=0
  while [ $((SECONDS-T0)) -lt 720 ]; do
    curl -fsS -m5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { OK=1; break; }
    [[ -n "$(docker ps -aq -f name=$C -f status=exited)" ]] && { echo "[$ARM] container exited"; break; }
    sleep 10
  done
  [ "$OK" = 1 ] || { echo "FAIL [$ARM] not healthy"; tail -25 "$RUN/boot_$ARM.log"; docker rm -f "$C" >/dev/null 2>&1; return 1; }
  # confirm SCAN_ALIGN actually engaged in the served worker (fail-loud on vacuous test)
  echo "  [$ARM] SCAN_ALIGN env in worker: $(docker exec $C printenv FR13_SCAN_ALIGN 2>/dev/null || echo '?') MODE=$(docker exec $C printenv FR13_SCAN_ALIGN_MODE 2>/dev/null || echo '?')"
  local MODEL=$(curl -fsS -m5 "http://127.0.0.1:$PORT/v1/models" | .venv/bin/python -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  echo "  [$ARM] healthy, model=$MODEL; gate n=$N conc=$CONC temp0.6"
  .venv/bin/python scripts/fr13_garble_gate.py run --endpoint "http://127.0.0.1:$PORT/v1" \
    --model "$MODEL" --arm "$ARM" --n "$N" --concurrency "$CONC" --out "$RUN/$ARM.jsonl" > "$RUN/gate_$ARM.log" 2>&1
  echo "  [$ARM] SCORE: $(.venv/bin/python scripts/fr13_garble_gate.py score --samples "$RUN/$ARM.jsonl" 2>&1 | tee "$RUN/${ARM}_score.txt" | head -1)"
  docker rm -f "$C" >/dev/null 2>&1 || true; wait $LPID 2>/dev/null || true
}

boot_and_gate sa_body_K1 1
boot_and_gate sa_off     0

echo ""; echo "=== SCAN_ALIGN garble gate SUMMARY (temp 0.6, undefined-name rate) ==="
for a in sa_body_K1 sa_off; do echo "  $a: $(grep -oE 'undefined-name-rate=[0-9.]+%|samples-with-undef=[0-9]+/[0-9]+' "$RUN/${a}_score.txt" 2>/dev/null | tr '\n' ' ')"; done
echo "  (reference: native=0.00%, tree-off prior=7.89-11.29%)"
echo "  VERDICT: K1 ~0% => compute-only no-HBM FIX; K1 still ~8-11% => K1 insufficient (cross-basis/committer)"
echo "=== DONE $RUN @ $(date -u +%H:%M:%S)Z ==="

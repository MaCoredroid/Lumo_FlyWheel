#!/usr/bin/env bash
# DECISIVE engagement test (workflow wigtzu39d reversal).
#
# Same deployed APC config as the failing tree/za arms (cache-ON, SNAP_FIX=1, cap=1e6,
# PIECEWISE, fp32-SSM, block1024, conv, zeroaccept) BUT with FR13_APC_REQUIRE_SNAP_FIX=1
# + FR13_APC_REQUIRE_HIT_SUFFIX_CAP=1000000 so variant.sh reads the worker's UNCONDITIONAL
# engagement marker (pid-176 live os.environ) and HARD-FAILS the boot (exit 4) if the
# corrections are vacuous (SNAP_FIX!=1 / cap defaulted to 64 / marker absent).
#
# Two questions, in order:
#   (Q0 engagement) does the boot pass the marker assert? marker shows SNAP_FIX=1 cap=1e6
#       in pid 176 -> the corrections were engaging ALL ALONG (the /proc 'drop' was a
#       setproctitle artifact). If exit 4 -> the spawn-inherit reproduction was unfaithful
#       (vLLM DOES curate) -> back to a real bridge.
#   (Q1 sufficiency) given engagement, does cat6root cache-ON now SOLVE?
#       solves ~N/N -> corrections sufficient (deliverable). fails -> corrections are
#       ENGAGED-BUT-INSUFFICIENT; residual carrier elsewhere (full-attn KV fp8 next).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
N=${N:-2}
SUBSET=output/fr13_b1_gold_swe/subset_astropy12907.json
TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=output/fr13_apc_engaged/run_$TS
mkdir -p "$ROOT"
echo "$ROOT" > /tmp/claude-1000/-home-mark-shared/46f03809-5059-4e30-936d-1adda7f44337/scratchpad/apc_engaged_root.txt 2>/dev/null || true
export RUNROOT="$ROOT"
export MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 OFFLOAD_AGENT=0 DEPLOY_FORCE_TEMP=0.6 SEED=1313
export FR10_METRICS=0 BATCH_INVARIANT=0
# IDENTICAL deployed APC config:
export FR13_ENABLE_APC=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32 CUDAGRAPH_MODE=PIECEWISE
export FR13_APC_HIT_RECURRENT_SUFFIX=1 FR13_APC_HIT_SUFFIX_CAP=1000000
export FR13_APC_SNAP_FIX_ZEROACCEPT=1
# THE ENGAGEMENT ASSERT (this is the new part): fail-loud if the worker marker is vacuous.
export FR13_APC_REQUIRE_SNAP_FIX=1 FR13_APC_REQUIRE_HIT_SUFFIX_CAP=1000000
echo "ENGAGED APC TEST: cat6root cache-ON cap=1e6 + REQUIRE marker asserts, astropy-12907, N=$N -> $ROOT" | tee "$ROOT/RESULTS.txt"

run_one() {
  local i=$1 A="eng_r$i"
  PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
  docker ps -a --format '{{.Names}}' | grep -qi fr13 && docker ps -a --format '{{.Names}}' | grep -i fr13 | xargs -r docker rm -f >/dev/null 2>&1 || true
  echo "=== [$A] cat6root cache-ON ENGAGED-ASSERT $(date -u +%H:%M:%SZ) ===" | tee -a "$ROOT/RESULTS.txt"
  bash scripts/fr13_bigdenom_swe_serve_variant.sh "$A" cat6root "$SUBSET" > "$ROOT/${A}.serve.log" 2>&1 || true
  local SO="$ROOT/$A/swe_orchestrator.log"
  # engagement proof from the worker marker:
  local mk; mk=$(cat "$ROOT/$A/apc_bridge_marker.txt" 2>/dev/null)
  [ -z "$mk" ] && mk=$(grep -oE "APC worker-env engagement OK: .*" "$ROOT/${A}.serve.log" 2>/dev/null | tail -1)
  local eng; eng=$(grep -qE "FAIL: APC (worker|engagement|bridge)" "$ROOT/${A}.serve.log" 2>/dev/null && echo "VACUOUS(exit4)" || echo "ENGAGED")
  local ru; ru=$(grep -lE "Unterminated string" "$ROOT/$A/swe_out/verified/per_task/"*/{agent,codex}_trace*.jsonl 2>/dev/null | wc -l)
  local v; v=$(grep -oE "verdict=(resolved|failed)" "$SO" 2>/dev/null | tail -1)
  [ -z "$v" ] && v="verdict=NORESULT(see ${A}.serve.log)"
  echo "$A -> $v | engagement=$eng | marker=[$mk] | runaway=$ru" | tee -a "$ROOT/RESULTS.txt"
}
for i in $(seq 1 "$N"); do run_one "$i"; done
res=$(grep -cE "^eng_r[0-9]+ -> verdict=resolved" "$ROOT/RESULTS.txt" || true)
tot=$(grep -cE "^eng_r[0-9]+ -> verdict=" "$ROOT/RESULTS.txt" || true)
echo "=== ENGAGED APC TEST (cat6root) resolved: ${res}/${tot}  [vs vacuous-read tree 0/3, za 1/2, cacheoff 3/3] ===" | tee -a "$ROOT/RESULTS.txt"
echo "=== engaged test done -> $ROOT/RESULTS.txt ==="

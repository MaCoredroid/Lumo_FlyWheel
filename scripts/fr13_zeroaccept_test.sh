#!/usr/bin/env bash
# ZERO-ACCEPT FIX validation.
#
# cat6root + cache-ON at the SAME config as the failing tree arm (FR13_ENABLE_APC=1 + cap=1e6 +
# PIECEWISE + fp32-SSM + block1024 + SNAP_FIX=1) PLUS the new FR13_APC_SNAP_FIX_ZEROACCEPT=1, which
# publishes the committed-root node-bank row (_row[0]) for zero-accept (accepted_len==0) steps so
# SNAP_FIX restores the committed-root seed instead of the stale bias row block_ids[cur-1].
#   solves ~3/3 => the zero-accept SSM restore was the (shared spine+tree) carrier => FIX WORKS.
#   fails  ~0/3 => zero-accept was rare/not the carrier => pivot to full-attn KV (FR13_FULL_ATTN_KV_FP8).
# Baseline for comparison: the big-N tree arm (same config, ZEROACCEPT=0) went 0/3.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
N=${N:-3}
SUBSET=output/fr13_b1_gold_swe/subset_astropy12907.json
TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=output/fr13_zeroaccept/run_$TS
mkdir -p "$ROOT"
echo "$ROOT" > /tmp/claude-1000/-home-mark-shared/46f03809-5059-4e30-936d-1adda7f44337/scratchpad/zeroaccept_root.txt 2>/dev/null || true
export RUNROOT="$ROOT"
export MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 OFFLOAD_AGENT=0 DEPLOY_FORCE_TEMP=0.6 SEED=1313
export FR10_METRICS=0 BATCH_INVARIANT=0
# IDENTICAL to the failing tree arm, PLUS the zero-accept fix:
export FR13_ENABLE_APC=1 MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32 CUDAGRAPH_MODE=PIECEWISE
export FR13_APC_HIT_RECURRENT_SUFFIX=1 FR13_APC_HIT_SUFFIX_CAP=1000000
export FR13_APC_SNAP_FIX_ZEROACCEPT=1
echo "ZERO-ACCEPT FIX: cat6root cache-ON cap=1e6 PIECEWISE SNAP_FIX_ZEROACCEPT=1, astropy-12907, N=$N -> $ROOT" | tee "$ROOT/RESULTS.txt"

run_one() {
  local i=$1 A="za_r$i"
  PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
  docker ps -a --format '{{.Names}}' | grep -qi fr13 && docker ps -a --format '{{.Names}}' | grep -i fr13 | xargs -r docker rm -f >/dev/null 2>&1 || true
  echo "=== [$A] cat6root cache-ON ZEROACCEPT=1 $(date -u +%H:%M:%SZ) ===" | tee -a "$ROOT/RESULTS.txt"
  bash scripts/fr13_bigdenom_swe_serve_variant.sh "$A" cat6root "$SUBSET" > "$ROOT/${A}.serve.log" 2>&1 || true
  local SO="$ROOT/$A/swe_orchestrator.log"
  local ru; ru=$(grep -lE "Unterminated string" "$ROOT/$A/swe_out/verified/per_task/"*/{agent,codex}_trace*.jsonl 2>/dev/null | wc -l)
  local v; v=$(grep -oE "verdict=(resolved|failed)" "$SO" 2>/dev/null | tail -1)
  [ -z "$v" ] && v="verdict=NORESULT(see ${A}.serve.log)"
  echo "$A -> $v | runaway=$ru" | tee -a "$ROOT/RESULTS.txt"
}
for i in $(seq 1 "$N"); do run_one "$i"; done
res=$(grep -cE "^za_r[0-9]+ -> verdict=resolved" "$ROOT/RESULTS.txt" || true)
tot=$(grep -cE "^za_r[0-9]+ -> verdict=" "$ROOT/RESULTS.txt" || true)
echo "=== ZERO-ACCEPT FIX (cat6root) resolved: ${res}/${tot}  [vs tree ZEROACCEPT=0: 0/3, cacheoff: 3/3] ===" | tee -a "$ROOT/RESULTS.txt"
echo "=== zero-accept test done -> $ROOT/RESULTS.txt ==="

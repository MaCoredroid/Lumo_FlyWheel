#!/usr/bin/env bash
# FR13 APC TTFT SWEEP — measure the prefix-cache TTFT speedup at each mamba_block_size,
# the COST side of the losslessness<->TTFT dial that pairs with the drift curve.
# Per block B in {1024,2048,4096,8192}: boot the DEPLOYED spec+cache server CLEAN
# (cuda-graph ON / ENFORCE_EAGER=0, NO capture, NO config-only) @ block B, then run the
# cold(miss)-vs-warm(hit) TTFT probe on the real seq49 prefix. Serial (concurrency=1).
# Run this AFTER the drift curve (cannot overlap inference). Reduce -> block | cold | warm |
# speedup | warm_cached. Pair with drift_curve_full.jsonl to pick the smallest block that is
# BOTH lossless (drift) AND keeps the most TTFT (speedup).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
BLOCKS=${BLOCKS:-"1024 2048 4096 8192"}
SRC=${SRC:-output/fr13_apc_rategate/run_20260625T084654Z/rg_OFF_r1/proxy_pair_dumps}
SEQ=${SEQ:-49}; PORT=${PORT:-9953}; GPU_UTIL=${GPU_UTIL:-0.82}
CAT6ROOT_TREE="[(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]"
CONTAINER="fr13-apc-multiturn"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RD=output/fr13_apc_ttft/run_${TS}; mkdir -p "$RD/logs"
echo "$RD" > /home/mark/.claude/jobs/22c39bb9/tmp/ttft_root.txt
echo "=== FR13 APC TTFT SWEEP  rundir=$RD  blocks=[$BLOCKS]  seq=$SEQ  (cuda-graph ON, no capture) ==="

teardown() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap teardown EXIT

for B in $BLOCKS; do
  echo "--- [block $B] boot clean deployed spec+cache server @ $(date -u +%H:%M:%S) ---"
  PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
  docker ps -a --format '{{.Names}}'|grep -i fr13|xargs -r docker rm -f >/dev/null 2>&1 || true
  sleep 2
  LAUNCH="$RD/logs/launch_b${B}.log"
  CONTAINER="$CONTAINER" PORT=$PORT GPU_UTIL=$GPU_UTIL MAX_NUM_SEQS=1 \
    TREE="$CAT6ROOT_TREE" FR10_METRICS=0 ENFORCE_EAGER=0 BATCH_INVARIANT=0 FR13_BI_TREE_ATTN=1 \
    FR13_ENABLE_APC=1 FR13_APC_CONFIG_ONLY=0 \
    MAMBA_BLOCK_SIZE="$B" MAMBA_SSM_CACHE_DTYPE=float32 \
    FR13_RUN_DIR="$PWD/$RD" LOG_DIR="$PWD/$RD/logs" \
    setsid bash scripts/fr13_launch_forked_fa2_tree_server.sh > "$LAUNCH" 2>&1 &
  T0=$SECONDS; HEALTHY=0
  while [ $((SECONDS-T0)) -lt 1200 ]; do
    curl -fsS -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { HEALTHY=1; break; }
    grep -qiE "CUDA out of memory|NVRM_NO_MEMORY" "$LAUNCH" 2>/dev/null && { echo "[b$B] OOM"; tail -15 "$LAUNCH"; break; }
    sleep 10
  done
  if [ "$HEALTHY" != 1 ]; then echo "[b$B] NOT healthy — skipping"; teardown; continue; fi
  echo "[b$B] healthy at $((SECONDS-T0))s; probing TTFT..."
  .venv/bin/python scripts/fr13_apc_ttft_probe.py \
    --port "$PORT" --dumps-dir "$SRC" --seq "$SEQ" --block "$B" \
    --out "$RD/ttft_b${B}.json" 2>&1 | tee -a "$RD/logs/probe_b${B}.log"
  teardown; sleep 2
done

echo "=== TTFT SWEEP REDUCE ==="
.venv/bin/python - "$RD" <<'PY'
import sys, json, glob, os
rd = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(rd, "ttft_b*.json")), key=lambda p: int(p.split("_b")[-1].split(".")[0])):
    try: rows.append(json.load(open(f)))
    except Exception: pass
print("\n=== FR13 APC TTFT vs mamba_block_size (real seq49 prefix, temp 0.6) ===")
print("  block | in_tok | ttft_cold | ttft_warm | speedup | warm_cached | note")
for r in rows:
    note = "VACUOUS(warm miss)" if r.get("vacuous_warm_miss") else "ok"
    print(f"  {r['block']:>5} | {str(r.get('input_tokens')):>6} | {r['ttft_cold_s']:>9.2f} | "
          f"{r['ttft_warm_s']:>9.2f} | {str(r['speedup']):>6}x | {r['warm_cached']:>11} | {note}")
print("  (smaller block = more TTFT kept on a hit but more drift; pick smallest LOSSLESS block)")
json.dump(rows, open(os.path.join(rd, "ttft_curve.json"), "w"))
print(f"  -> {os.path.join(rd, 'ttft_curve.json')}")
PY
echo "=== TTFT SWEEP DONE -> $RD ==="

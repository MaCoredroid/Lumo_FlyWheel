#!/usr/bin/env bash
# FR13 — does NATIVE MTP-5 + CACHE(APC) under REAL batching garble?
# ============================================================================
# User's control (2026-07-11): our machinery = tree(cat8)+cache garbles at effective B~1
# via the tree's INTRA-request batched-scan co-residency. Sharp control = the native analog
# native-MTP5 + cache at REAL inter-request batch B=8 (FORCED co-residency, since live SWE
# serializes to ~1.3). If native+cache+realB8 garbles -> it's general batch/cache co-residency.
# If it stays at the B=1 rate -> the garble is specifically the tree's intra-request scan.
#
# Instrument = fr13_garble_gate.py (identifier-consistency probes that BAIT near-neighbor
# corruption + AST undefined-name-rate scorer). SAME-BOOT A/B: one native+APC server at
# max_num_seqs=8, gate at concurrency=8 (real batch) vs concurrency=1 (B=1), identical
# (prompt,seed) pairs. Fail-loud if co-residency never establishes (the conc4!=b4 trap).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
CONT=fr13-native-apc-b8; PORT=9955; N=${N:-24}
TS=$(date -u +%Y%m%dT%H%M%SZ); RUN=output/fr13_native_batch_garble/run_$TS; mkdir -p "$RUN"
echo "=== NATIVE-MTP5 + CACHE + REAL-BATCH garble gate  -> $RUN ==="

# hygiene + free-GPU assert
docker ps -aq --filter "name=$CONT" | xargs -r docker rm -f >/dev/null 2>&1 || true
PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true

echo "[boot] native MTP-5 + APC(cache) max_num_seqs=8 GPU_UTIL=0.8 ..."
CONTAINER=$CONT PORT=$PORT GPU_UTIL=0.8 MAX_NUM_SEQS=8 SEED=0 \
  NATIVE_ENABLE_APC=1 MAMBA_BLOCK_SIZE=1024 APC_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32 \
  GPU_GUARD_FLOOR_MIB=3000 \
  bash scripts/fr13_launch_native_mtp_server.sh > "$RUN/boot.log" 2>&1

echo "[boot] waiting /health (<=12min)..."
T0=$SECONDS; OK=0
while [ $((SECONDS-T0)) -lt 720 ]; do
  curl -fsS -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { OK=1; break; }
  docker ps --format '{{.Names}}' | grep -q "$CONT" || { echo "[boot] container died"; break; }
  sleep 10
done
[ "$OK" = 1 ] || { echo "FAIL: not healthy"; tail -30 "$RUN/boot.log"; docker logs --tail 40 "$CONT" 2>&1; exit 2; }
MODEL=$(curl -fsS -m 5 "http://127.0.0.1:$PORT/v1/models" | .venv/bin/python -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
echo "[boot] healthy at $((SECONDS-T0))s; model=$MODEL"

# ---- ARM B8: REAL batch (concurrency 8). Sample engine Running: to PROVE co-residency ----
echo "[B8] gate concurrency=8 (real batch) n=$N ..."
.venv/bin/python scripts/fr13_garble_gate.py run --endpoint "http://127.0.0.1:$PORT/v1" \
  --model "$MODEL" --arm nat_cache_b8 --n "$N" --concurrency 8 --out "$RUN/b8.jsonl" > "$RUN/b8_run.log" 2>&1 &
GATE=$!
# sample engine Running: for the WHOLE gate lifetime (proves real co-residency, not conc4!=b4)
while kill -0 $GATE 2>/dev/null; do
  docker logs --tail 4 "$CONT" 2>&1 | grep -oE 'Running: [0-9]+ reqs' | tail -1
  sleep 3
done > "$RUN/running_b8.log" 2>&1
wait $GATE; cat "$RUN/b8_run.log"
MAXRUN=$(grep -oE 'Running: [0-9]+' "$RUN/running_b8.log" | grep -oE '[0-9]+' | sort -n | tail -1)
echo "[B8] max Running observed during gate = ${MAXRUN:-0} (need >>1 for a valid batch test)"

# ---- ARM B1: same boot, concurrency=1 (serial, Running=1) ----
echo "[B1] gate concurrency=1 (B=1 baseline) n=$N ..."
.venv/bin/python scripts/fr13_garble_gate.py run --endpoint "http://127.0.0.1:$PORT/v1" \
  --model "$MODEL" --arm nat_cache_b1 --n "$N" --concurrency 1 --out "$RUN/b1.jsonl" 2>&1 | tee "$RUN/b1_run.log"

# ---- score both ----
echo ""; echo "=== SCORES (undefined-name rate = garble metric) ==="
echo "--- B8 (real batch, max Running=${MAXRUN:-0}) ---"; .venv/bin/python scripts/fr13_garble_gate.py score --samples "$RUN/b8.jsonl" 2>&1 | tee "$RUN/b8_score.txt"
echo "--- B1 (serial) ---";                              .venv/bin/python scripts/fr13_garble_gate.py score --samples "$RUN/b1.jsonl" 2>&1 | tee "$RUN/b1_score.txt"
echo ""
echo "=== cache engaged? (prefix hit-rate during run) ==="
docker logs "$CONT" 2>&1 | grep -oE 'Prefix cache hit rate: [0-9.]+%' | sort -u | tail -3
echo "=== VERDICT: B8 undefined-rate > B1 => native+cache+realbatch GARBLES; ~equal => tree-specific ==="
echo "=== teardown ==="; docker rm -f "$CONT" >/dev/null 2>&1 || true
echo "=== DONE $RUN @ $(date -u +%H:%M:%S)Z ==="

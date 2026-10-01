#!/usr/bin/env bash
# Boot the deployed LumoTree (hydra27 fixed32, promoted stack) in serve-only mode,
# replay the request corpus with signed requests, then stop it.
# Usage: run_tree_arm.sh <mode greedy|sampled> [max_tokens]
set -uo pipefail
MODE=${1:-greedy}; MAXTOK=${2:-1024}
ROOT=/home/mark/shared/lumotree-v2exp-runs
WT=/home/mark/shared/lumotree-v2exp-20260930
PORTWT=/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$ROOT/tree-driver-$MODE-$TS.log
exec > >(tee -a "$LOG") 2>&1
[[ -z "$(docker ps -q)" ]] || { echo "a container is running; refusing"; exit 3; }
mkdir -p "$ROOT/tree"
before=$(ls -d "$ROOT"/tree/fr14_promoab_C_v2exp_* 2>/dev/null | sort | tail -1)
ARM_KIND=C PROMOAB_ARM_SUFFIX=_v2exp setsid bash "$WT/scripts/v2exp/promoab_serve_only.sh" \
  > "$ROOT/tree-promoab-$TS.out" 2>&1 < /dev/null &
DRV=$!
echo "driver pid=$DRV"
RUNROOT=""
t0=$(date +%s)
while :; do
  sleep 15
  cand=$(ls -d "$ROOT"/tree/fr14_promoab_C_v2exp_* 2>/dev/null | sort | tail -1)
  [[ -n "$cand" && "$cand" != "$before" ]] && RUNROOT=$cand
  [[ -n "$RUNROOT" && -f "$RUNROOT/READY.json" ]] && break
  kill -0 $DRV 2>/dev/null || { echo "driver exited before READY"; tail -40 "$ROOT/tree-promoab-$TS.out"; exit 1; }
  (( $(date +%s) - t0 > 3000 )) && { echo "READY timeout"; [[ -n "$RUNROOT" ]] && touch "$RUNROOT/STOP"; exit 1; }
done
echo "READY after $(( $(date +%s) - t0 ))s: $(cat "$RUNROOT/READY.json")"
OUT=$ROOT/replay/tree-$MODE-$TS; mkdir -p "$OUT"; echo "$RUNROOT" > "$OUT/serve_runroot.txt"
export V2EXP_READY_FILE="$RUNROOT/READY.json"
cd "$WT/scripts/v2exp"
PY=$PORTWT/.venv/bin/python
curl -s http://127.0.0.1:9950/metrics > "$OUT/metrics_boot.txt"
$PY replay.py --arm tree-warmup --requests "$ROOT/corpus/requests" --out "$OUT/warmup.jsonl" \
  --mode greedy --max-tokens 64 --limit 1 --auth-hook fixed32_auth:headers
$PY replay.py --arm tree --requests "$ROOT/corpus/requests" --out "$OUT/replay.jsonl" \
  --mode "$MODE" --max-tokens "$MAXTOK" --auth-hook fixed32_auth:headers
curl -s http://127.0.0.1:9950/metrics > "$OUT/metrics_end.txt"
touch "$RUNROOT/STOP"
echo "stop requested; waiting for driver teardown"
wait $DRV; echo "driver rc=$? $(date -u +%FT%TZ)"

#!/usr/bin/env bash
# FR13 B=1 depth-5 LOSSLESS p-rescore (binding gate): per-token clear-margin argmax
# flip-rate of each arm's served stream vs the no-spec RECURRENT decode oracle.
# E5 = the floor/bar; cat9/cat6 lossless iff within E5's floor. (p-only; the temp-0.6
# q-capture/TV half is blocked on vLLM 0.19.2.) GPU-serialized, one oracle boot per arm.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
REPO=/home/mark/shared/lumoFlyWheel
RUNROOT=output/fr13_bigdenom_swe
OUTROOT=output/fr13_b1_lossless
IMAGE="vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"
mkdir -p "$OUTROOT"

recover() { PYTHONPATH="$REPO/src" .venv/bin/python - <<'PY' 2>/dev/null || true
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}

for arm in nativeE5_b1 cat9_b1 cat6root_b1; do
  SRC="$OUTROOT/${arm}_src.json"
  POUT="$OUTROOT/rescore_${arm}.json"
  dump="$RUNROOT/$arm/proxy_pair_dumps"
  n=$(ls "$dump"/pair_*.json 2>/dev/null | wc -l)
  echo "===== LOSSLESS p-rescore $arm (pair-dumps=$n) ====="
  [ "$n" -gt 0 ] || { echo "[$arm] FAIL: no pair-dumps"; continue; }
  # reduce: pair-dumps -> oracle src (CPU in container)
  docker rm -f "fr13-reducer-$arm" >/dev/null 2>&1 || true
  docker run --rm --name "fr13-reducer-$arm" \
    -v "$REPO:/workspace" -v /models:/models -e PYTHONPATH=/workspace/src \
    --entrypoint bash "$IMAGE" -lc "cd /workspace; python3 scripts/fr13_swe_stream_to_oracle_src.py --dump-dir '/workspace/${dump#$REPO/}' --out '/workspace/${SRC#$REPO/}'" \
    2>&1 | tail -4
  recover
  # rescore p: no-spec recurrent oracle -> clear-margin flip rate
  SEED=1313 TOPK=20 THRESH=1.0 GPU_UTIL=0.88 \
    scripts/fr13_recur_rescore_in_container.sh "$arm" "$SRC" "$POUT" \
    2>&1 | tee "$OUTROOT/rescore_${arm}.log" | tail -6
  docker rm -f fr13-recur-oracle >/dev/null 2>&1 || true
  recover
done
echo "===== LOSSLESS p-rescore DONE ====="

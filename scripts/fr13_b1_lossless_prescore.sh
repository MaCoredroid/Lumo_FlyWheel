#!/usr/bin/env bash
# FR13 B=1 depth-5 LOSSLESS p-rescore (binding gate): per-token clear-margin argmax
# flip-rate of each arm's served stream vs the no-spec RECURRENT decode oracle.
# E5 = floor/bar; cat9/cat6 lossless iff within E5's floor.
#
# SAMPLED: the recurrent oracle scores ONE stream at a time (~200s/turn incl rep x2),
# so the full ~2315-turn streams are ~64h/arm (intractable). The flip-rate is
# statistical, so we SAMPLE N random pair-dumps/arm (single-stream determinism kept =
# instrument unchanged; batching would risk GDN batch-dependence). N=$NTURNS turns
# (~$NTURNS x ~223 pos) is enough to confirm within-floor.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
REPO=/home/mark/shared/lumoFlyWheel
RUNROOT=output/fr13_bigdenom_swe
OUTROOT=output/fr13_b1_lossless
IMAGE="vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"
NTURNS="${NTURNS:-40}"
mkdir -p "$OUTROOT"

recover() { PYTHONPATH="$REPO/src" .venv/bin/python - <<'PY' 2>/dev/null || true
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}

for arm in ${LOSSLESS_ARMS:-nativeE5_b1 cat9_b1 cat6root_b1}; do
  dump="$RUNROOT/$arm/proxy_pair_dumps"
  sdump="$OUTROOT/${arm}_sample_dumps"
  SRC="$OUTROOT/${arm}_src.json"
  POUT="$OUTROOT/rescore_${arm}.json"
  ntot=$(ls "$dump"/pair_*.json 2>/dev/null | wc -l)
  echo "===== LOSSLESS p-rescore $arm (sample $NTURNS of $ntot pair-dumps) ====="
  [ "$ntot" -gt 0 ] || { echo "[$arm] FAIL: no pair-dumps"; continue; }
  # deterministic random sample (seed by arm) of N pair-dumps
  rm -rf "$sdump"; mkdir -p "$sdump"
  ls "$dump"/pair_*.json | sort | awk 'BEGIN{srand(1313)} {print rand()"\t"$0}' | sort -k1,1n | head -n "$NTURNS" | cut -f2 | while read f; do cp "$f" "$sdump/"; done
  echo "  sampled $(ls "$sdump"/pair_*.json 2>/dev/null | wc -l) dumps"
  # reduce: sampled pair-dumps -> oracle src (CPU in container)
  docker rm -f "fr13-reducer-$arm" >/dev/null 2>&1 || true
  docker run --rm --name "fr13-reducer-$arm" \
    -v "$REPO:/workspace" -v /models:/models -e PYTHONPATH=/workspace/src \
    --entrypoint bash "$IMAGE" -lc "cd /workspace; python3 scripts/fr13_swe_stream_to_oracle_src.py --dump-dir '/workspace/${sdump#$REPO/}' --out '/workspace/${SRC#$REPO/}'" \
    2>&1 | tail -3
  recover
  # rescore p: no-spec recurrent oracle -> clear-margin flip rate
  SEED=1313 TOPK=20 THRESH=1.0 GPU_UTIL=0.88 \
    scripts/fr13_recur_rescore_in_container.sh "$arm" "$SRC" "$POUT" \
    2>&1 | tee "$OUTROOT/rescore_${arm}.log" | tail -8
  docker rm -f fr13-recur-oracle >/dev/null 2>&1 || true
  recover
done
echo "===== LOSSLESS p-rescore (sampled) DONE ====="

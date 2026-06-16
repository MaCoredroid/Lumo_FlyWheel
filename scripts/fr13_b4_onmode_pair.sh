#!/usr/bin/env bash
# FR13 B=4 ON-mode (lossless + temp-0.6 drift) for ONE arm pair: the candidate vs
# its depth-matched native. Per arm: reduce pair-dump -> oracle src (CPU, in
# container), recurrent rescore p (temp-0 flips), capture-q-deploy q (temp-0.6 TV).
# Then consolidate -> deploy-lossless verdict + per-position deploy-temp06-drift.
# GPU-SERIALIZED — run AFTER the speed campaign frees the box.
#
# Usage: fr13_b4_onmode_pair.sh <cand_arm_dir> <cand_spec_config_json> \
#                              <native_arm_dir> <native_spec_config_json>
#   e.g. fr13_b4_onmode_pair.sh cat9_b4 '<tree env via locked launcher; see note>' \
#                               nativeE5_b4 '{"method":"qwen3_5_mtp","num_speculative_tokens":5}'
# NOTE: tree arms capture q through the locked tree serve (the q-capture path needs
# the deployment spec config). For the FIRST pass we do the NATIVE pair (E5) which
# is the within-floor BAR; cat9 q-capture reuses the deployed tree config.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
REPO=/home/mark/shared/lumoFlyWheel
RUNROOT=output/fr13_bigdenom_swe
OUTROOT=output/fr13_b4_onmode
IMAGE="vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"
mkdir -p "$OUTROOT"

CAND_ARM=${1:?cand arm dir}
CAND_SPEC=${2:?cand spec_config json}
NAT_ARM=${3:?native arm dir}
NAT_SPEC=${4:?native spec_config json}

reduce_arm() {  # arm_dir out_src
  local arm="$1" out="$2"
  local dump="$RUNROOT/$arm/proxy_pair_dumps"
  local n; n=$(ls "$dump"/pair_*.json 2>/dev/null | wc -l)
  (( n > 0 )) || { echo "FAIL: $arm pair-dump empty"; return 2; }
  echo "[$arm] pair-dump files: $n -> $out"
  docker rm -f "fr13-reducer-$arm" >/dev/null 2>&1 || true
  docker run --rm --name "fr13-reducer-$arm" \
    -v "$REPO:/workspace" -v /models:/models -e PYTHONPATH=/workspace/src \
    --entrypoint bash "$IMAGE" -lc "
cd /workspace
python3 scripts/fr13_swe_stream_to_oracle_src.py \
  --dump-dir '/workspace/${dump#$REPO/}' --out '/workspace/${out#$REPO/}'
"
}

recover() { PYTHONPATH="$REPO/src" .venv/bin/python - <<'PY'
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}

for pair in "$CAND_ARM:$CAND_SPEC" "$NAT_ARM:$NAT_SPEC"; do
  arm="${pair%%:*}"; spec="${pair#*:}"
  SRC="$OUTROOT/${arm}_src.json"
  P_OUT="$OUTROOT/rescore_${arm}.json"
  Q_OUT="$OUTROOT/captureq_${arm}.json"
  echo "===== ON-mode $arm ====="
  reduce_arm "$arm" "$SRC" || { echo "FAIL reduce $arm"; exit 3; }
  recover || true
  # p: no-spec recurrent oracle (temp-0 flip basis)
  echo "=== rescore p (recurrent oracle) $arm ==="
  SEED=1313 TOPK=20 THRESH=1.0 GPU_UTIL=0.88 \
    scripts/fr13_recur_rescore_in_container.sh "$arm" "$SRC" "$P_OUT" \
    2>&1 | tee "$OUTROOT/rescore_${arm}.log"
  docker rm -f fr13-recur-oracle >/dev/null 2>&1 || true
  recover || true
  # q: forced-decode the served stream through the spec serve (temp-0.6 TV basis)
  echo "=== capture-q-deploy $arm spec=$spec ==="
  docker rm -f "fr13-captureq-$arm" >/dev/null 2>&1 || true
  docker run --rm --name "fr13-captureq-$arm" --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$REPO:/workspace" -v /models:/models -e PYTHONPATH=/workspace/src \
    -e VLLM_USE_V1=1 --entrypoint bash "$IMAGE" -lc "
cd /workspace
python3 scripts/fr13_recurrent_decode_oracle.py capture-q-deploy \
  --arm '$arm' --src '/workspace/${SRC#$REPO/}' \
  --out '/workspace/${Q_OUT#$REPO/}' \
  --spec-config '$spec' --seed 1313 --top-k 20 \
  --gpu-util 0.88 --attn-backend FLASH_ATTN
" 2>&1 | tee "$OUTROOT/captureq_${arm}.log"
  docker rm -f "fr13-captureq-$arm" >/dev/null 2>&1 || true
  recover || true
done
echo "===== ON-mode pair done: srcs/rescores/captureq in $OUTROOT ====="

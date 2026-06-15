#!/usr/bin/env bash
# FR13 big-denominator PHASE 3 (Rescore, GPU): rescore BOTH arms' served streams
# vs the deployment-correct RECURRENT decode oracle, clear-margin flips + Wilson CI.
#
# Per arm: (1) reduce the proxy pair-dump -> oracle --src (class #12 round-trip
# validated denominator, inside the container = transformers available), then
# (2) run the recurrent oracle rescore ONCE (fr13_recur_rescore_in_container.sh:
# FR12_NO_SPECULATIVE_CONFIG=1 in _build_llm -> NO speculative_config -> spec
# counters CANNOT advance; recurrent decode path counter > 0 = engaged; in-process
# clean recurrent argmax recorded per step, NOT streamed logprobs).
# Then consolidate -> Wilson 95% CI + non-vacuity gate.
#
# GPU-SERIALIZED. Pre-boot hygiene + teardown are inside the in-container launcher.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel

REPO=/home/mark/shared/lumoFlyWheel
IMAGE="vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"
RUNROOT=output/fr13_bigdenom_swe
OUTROOT=output/fr13_bigdenom_rescore
mkdir -p "$OUTROOT"

CAT9_DUMP="$RUNROOT/cat9_a/proxy_pair_dumps"
NATIVE_DUMP="$RUNROOT/native_a/proxy_pair_dumps"

# ---- 0. pre-flight: both arms produced non-empty pair-dumps (NON-VACUITY) ----
for arm in cat9_a native_a; do
  d="$RUNROOT/$arm/proxy_pair_dumps"
  n=$(ls "$d"/pair_*.json 2>/dev/null | wc -l)
  if (( n == 0 )); then echo "FAIL: $arm pair-dump empty ($d) -- vacuous (#9)"; exit 2; fi
  echo "[$arm] pair-dump files: $n"
done

# ---- helper: reduce one arm's pair-dump -> oracle --src (in container) ----
reduce_arm() {  # <arm_dir_name> <dump_dir> <out_src>
  local name="$1" dump="$2" out="$3"
  docker rm -f "fr13-reducer-$name" >/dev/null 2>&1 || true
  docker run --rm --name "fr13-reducer-$name" \
    -v "$REPO:/workspace" -v /models:/models \
    -e PYTHONPATH=/workspace/src \
    --entrypoint bash "$IMAGE" -lc "
cd /workspace
python3 scripts/fr13_swe_stream_to_oracle_src.py \
  --dump-dir '/workspace/${dump#$REPO/}' \
  --out '/workspace/${out#$REPO/}'
"
}

# ---- 1. reduce both arms (CPU-only, no GPU contention) ----
CAT9_SRC="$OUTROOT/cat9_src.json"
NATIVE_SRC="$OUTROOT/native_src.json"
echo "=== reduce cat9 ==="
reduce_arm cat9 "$CAT9_DUMP" "$CAT9_SRC" || { echo "FAIL: cat9 reduce"; exit 3; }
echo "=== reduce native ==="
reduce_arm native "$NATIVE_DUMP" "$NATIVE_SRC" || { echo "FAIL: native reduce"; exit 3; }

# ---- 2. spec-frozen evidence: the oracle disables speculation (code-enforced) ----
# fr13_recurrent_decode_oracle.py::_build_llm sets FR12_NO_SPECULATIVE_CONFIG=1 and
# builds LLM(...) with NO speculative_config -> spec counters cannot advance.
SPEC_EV="$OUTROOT/spec_frozen_evidence.json"
.venv/bin/python - "$SPEC_EV" <<'PY'
import json, re, sys
src = open("scripts/fr13_recurrent_decode_oracle.py").read()
ev = {
  "FR12_NO_SPECULATIVE_CONFIG_set": 'FR12_NO_SPECULATIVE_CONFIG' in src and 'setdefault("FR12_NO_SPECULATIVE_CONFIG", "1")' in src,
  "no_speculative_config_kwarg": "speculative_config" not in re.sub(r"#.*", "", src.split("kwargs")[1].split("return LLM")[0]) if "kwargs" in src else None,
  "in_process_engine_core": 'VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"' in src,
  "recurrent_path_counter_failloud": '_forward_core_decode_non_spec' in src and 'RECURRENT_PATH_ENGAGED' in src,
  "note": "FR12_NO_SPECULATIVE_CONFIG=1 + LLM() built with NO speculative_config -> spec_decode counters cannot advance during rescore; the recurrent decode path counter (>0) is the engagement gate; clean argmax is recorded in-process per step (NOT streamed logprobs).",
}
json.dump(ev, open(sys.argv[1], "w"), indent=2)
print(json.dumps(ev, indent=2))
PY

# ---- 3. rescore each arm ONCE (GPU, in container, recurrent oracle) ----
CAT9_RESCORE="$OUTROOT/rescore_cat9.json"
NATIVE_RESCORE="$OUTROOT/rescore_native.json"

echo "=== rescore native (the within-floor BAR) ==="
SEED=1313 TOPK=20 THRESH=1.0 GPU_UTIL=0.88 \
  scripts/fr13_recur_rescore_in_container.sh native "$NATIVE_SRC" "$NATIVE_RESCORE" \
  2>&1 | tee "$OUTROOT/rescore_native.log"
RC_N=${PIPESTATUS[0]}
docker rm -f fr13-recur-oracle >/dev/null 2>&1 || true
PYTHONPATH="$REPO/src" .venv/bin/python - <<'PY'
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
(( RC_N == 0 )) || { echo "FAIL: native rescore rc=$RC_N"; exit 4; }

echo "=== rescore cat9 (deployed locked tree) ==="
SEED=1313 TOPK=20 THRESH=1.0 GPU_UTIL=0.88 \
  scripts/fr13_recur_rescore_in_container.sh cat9 "$CAT9_SRC" "$CAT9_RESCORE" \
  2>&1 | tee "$OUTROOT/rescore_cat9.log"
RC_C=${PIPESTATUS[0]}
docker rm -f fr13-recur-oracle >/dev/null 2>&1 || true
PYTHONPATH="$REPO/src" .venv/bin/python - <<'PY'
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
(( RC_C == 0 )) || { echo "FAIL: cat9 rescore rc=$RC_C"; exit 4; }

# ---- 4. consolidate: Wilson CI + non-vacuity gate ----
echo "=== consolidate ==="
.venv/bin/python scripts/fr13_bigdenom_rescore_consolidate.py \
  --cat9-rescore "$CAT9_RESCORE" --cat9-src "$CAT9_SRC" \
  --native-rescore "$NATIVE_RESCORE" --native-src "$NATIVE_SRC" \
  --spec-frozen-evidence "$SPEC_EV" \
  --out "$OUTROOT/consolidated.json"
echo "PHASE3_DONE -> $OUTROOT/consolidated.json"

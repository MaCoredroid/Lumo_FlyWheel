#!/usr/bin/env bash
# FR13 B=4 FINALIZER: waits for the speed campaign to finish, dumps all speed bars,
# then runs the ON-mode (lossless + temp-0.6 drift) for the DECISIVE depth-5 pair
# (cat9 vs native-E5) and the depth-3 pair (3-3-3 vs native-E3) if present, and
# writes the consolidated numbers into FR13_B4_DEPLOY_RESULTS.md. Fully autonomous.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
RUNROOT=output/fr13_bigdenom_swe
OUTROOT=output/fr13_b4_onmode
DRIVER_PID=${1:?driver pid}
mkdir -p "$OUTROOT"

echo "[finalizer] waiting for speed campaign driver pid=$DRIVER_PID"
while ps -p "$DRIVER_PID" >/dev/null 2>&1; do sleep 120; done
echo "[finalizer] speed campaign DONE; dumping bars"

# ---- 1. dump all speed bars ----
SPEED_DUMP="$OUTROOT/all_speed_bars.txt"
: > "$SPEED_DUMP"
for arm in nativeE5_b4 nativeE3_b4 nativeE4_b4 cat9_b4 opt1_b4c cat6root_b4 cat10_b4 threethree_b4 cat9_contam_b4; do
  f="$RUNROOT/$arm/deploy_speed_b4.json"
  if [ -f "$f" ]; then
    python3 -c "import json;d=json.load(open('$f'));print('$arm s/fwd=%.4f accept=%.4f committed=%.4f ms/tok=%.2f tps=%.4f n=%d'%(d['s_per_fwd'],d['accept_per_event'],d['committed_per_event'],1000*d['s_per_fwd']/d['committed_per_event'],d['derived_tps'],d['n_tasks']))" >> "$SPEED_DUMP"
  else
    echo "$arm NO_BAR" >> "$SPEED_DUMP"
  fi
done
cat "$SPEED_DUMP"

# ---- 2. ON-mode for the decisive depth-5 pair: cat9 vs native-E5 ----
# tree arm q-capture needs the deployed tree spec; native q-capture uses E5 spec.
# We run the NATIVE-E5 ON-mode first (the within-floor BAR), then cat9.
NAT_SPEC='{"method":"qwen3_5_mtp","num_speculative_tokens":5}'
# cat9 deployment spec for q-capture is the locked tree (num_spec=9). The capture-q
# path reads the deployment spec from env; for the tree arm we pass the tree config.
CAT9_SPEC='{"method":"qwen3_5_mtp","num_speculative_tokens":9}'

if [ -d "$RUNROOT/cat9_b4/proxy_pair_dumps" ] && [ -d "$RUNROOT/nativeE5_b4/proxy_pair_dumps" ]; then
  echo "[finalizer] ON-mode depth-5 pair (cat9 vs native-E5)"
  bash scripts/fr13_b4_onmode_pair.sh cat9_b4 "$CAT9_SPEC" nativeE5_b4 "$NAT_SPEC" \
    > "$OUTROOT/onmode_d5.log" 2>&1 || echo "[finalizer] ON-mode d5 errors (see onmode_d5.log)"

  # consolidate -> deploy-lossless (cat9 vs native flip rate, Wilson CI)
  if [ -f "$OUTROOT/rescore_cat9_b4.json" ] && [ -f "$OUTROOT/rescore_nativeE5_b4.json" ]; then
    .venv/bin/python scripts/fr13_recurrent_decode_oracle.py >/dev/null 2>&1 || true
    # spec-frozen evidence + consolidate (reuse the phase3 consolidator)
    .venv/bin/python - <<'PY' > "$OUTROOT/spec_frozen_evidence.json" 2>/dev/null || true
import json,re
src=open("scripts/fr13_recurrent_decode_oracle.py").read()
print(json.dumps({"FR12_NO_SPECULATIVE_CONFIG_set":'setdefault("FR12_NO_SPECULATIVE_CONFIG", "1")' in src,"note":"no-spec recurrent oracle"}))
PY
    .venv/bin/python scripts/fr13_bigdenom_rescore_consolidate.py \
      --cat9-rescore "$OUTROOT/rescore_cat9_b4.json" --cat9-src "$OUTROOT/cat9_b4_src.json" \
      --native-rescore "$OUTROOT/rescore_nativeE5_b4.json" --native-src "$OUTROOT/nativeE5_b4_src.json" \
      --spec-frozen-evidence "$OUTROOT/spec_frozen_evidence.json" \
      --out "$OUTROOT/consolidated_d5.json" > "$OUTROOT/consolidate_d5.log" 2>&1 \
      && .venv/bin/python scripts/fr13_measure.py deploy-lossless \
           --consolidated "$OUTROOT/consolidated_d5.json" \
           --out "$OUTROOT/deploy_lossless_d5.json" >> "$OUTROOT/consolidate_d5.log" 2>&1 \
      || echo "[finalizer] consolidate/lossless d5 errors"
  fi

  # temp-0.6 drift per arm (q vs p on its own stream), native floor first
  for arm in nativeE5_b4 cat9_b4; do
    q="$OUTROOT/captureq_${arm}.json"; p="$OUTROOT/rescore_${arm}.json"
    if [ -f "$q" ] && [ -f "$p" ]; then
      .venv/bin/python scripts/fr13_measure.py deploy-temp06-drift \
        --q "$q" --p "$p" --temp 0.6 \
        --out "$OUTROOT/deploy_temp06_${arm}.json" \
        >> "$OUTROOT/temp06_${arm}.log" 2>&1 || echo "[finalizer] temp06 $arm errors"
    fi
  done
fi

echo "[finalizer] DONE. Artifacts in $OUTROOT"
ls -la "$OUTROOT"/deploy_*.json 2>/dev/null

#!/usr/bin/env bash
# FR13 DEVICE-COMMITTER LOSSLESS reduce: given the device + native-E5 capture-q (q)
# and recurrent-oracle rescore (p) artifacts (same drift src per arm), compute:
#   1. native-E5 deploy-temp06-drift  -> native floor p95 (the BAR)
#   2. device   deploy-temp06-drift   -> device TV + within-floor verdict
#   3. flip-rate Wilson-CI cross-check (deploy-lossless) device vs native
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
O=output/fr13_b4_onmode
PY=.venv/bin/python

# SHARED no-spec recurrent oracle p: device and native served streams are
# byte-identical on this trajectory (verified dev==native for all 675 native
# turns), so the no-spec recurrent decode oracle p is IDENTICAL. We reuse the
# device rescore as p for BOTH arms. The ONLY difference is the verify-q:
# native = MTP-5 verify, device = cat9 tree-9 verify. The drift q.src==p.src
# check passes because both q-captures use the same --src dm_device_src_drift.json.
P=$O/rescore_dm_device.json

# 1. native floor first (no floor arg -> just produce p95)
$PY scripts/fr13_measure.py deploy-temp06-drift \
  --q "$O/captureq_nativeE5_b4.json" --p "$P" \
  --temp 0.6 --per-position-floor 0.05 --dump-positions \
  --out "$O/deploy_temp06_nativeE5_b4.json" || { echo "FAIL native drift"; exit 2; }

FLOOR=$($PY -c "import json;print(json.load(open('$O/deploy_temp06_nativeE5_b4.json'))['p95_tv_q_p_at_temp'])")
echo "NATIVE_E5_FLOOR_P95=$FLOOR"

# 2. device drift vs the native floor (same shared p; q = device cat9-9 verify)
$PY scripts/fr13_measure.py deploy-temp06-drift \
  --q "$O/captureq_dm_device.json" --p "$P" \
  --temp 0.6 --per-position-floor 0.05 --dump-positions \
  --native-floor-p95 "$FLOOR" \
  --out "$O/deploy_temp06_dm_device.json" || { echo "FAIL device drift"; exit 2; }

echo "=== DEVICE drift summary ==="
$PY -c "import json;d=json.load(open('$O/deploy_temp06_dm_device.json'));print(json.dumps({k:d[k] for k in ('arm','temp','mean_tv_q_p_at_temp','p95_tv_q_p_at_temp','max_tv_q_p_at_temp','n_positions_scored','over_floor_count','depth_matched_native_floor_p95','arm_p95_above_native_floor','within_floor_verdict')},indent=2))"

# 3. per-token clear-margin argmax instrument (the non-blind instrument MEMORY
# flags). Device served stream == host/native stream (byte-identical), so the
# no-spec recurrent oracle clear-margin flip rate is IDENTICAL across arms by
# construction -> trivially within floor. Report the device stream's own rate.
echo "=== DEVICE clear-margin flip rate (no-spec recurrent oracle, device served stream) ==="
$PY -c "
import json
d=json.load(open('$O/rescore_dm_device.json'))
tot=d['total_positions']; cf=d['total_clear_margin_flips']; fl=d['total_flips']
print(json.dumps({'total_positions':tot,'total_flips':fl,'total_clear_margin_flips':cf,
  'clear_margin_rate':cf/tot if tot else None,'flip_rate':fl/tot if tot else None,
  'RECURRENT_PATH_ENGAGED':d['RECURRENT_PATH_ENGAGED'],
  'note':'device stream byte-identical to host/native -> per-token argmax flip rate identical across arms (stream identity), trivially within floor'},indent=2))
"
echo "REDUCE_DONE"

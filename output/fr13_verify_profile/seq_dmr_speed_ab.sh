#!/usr/bin/env bash
# #59a speed A/B (LOCAL): FR13_DRAFTER_META_REUSE=1 vs 0, graph mode, B=1
# probes (seed 1313). Selfcheck OFF (it rebuilds fresh metadata — defeats the
# lever). Equivalence already proven: dmr4 eager gate, 291 selfcheck-OK steps.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel

echo "===== ARM A: dmr-graph (REUSE=1) ====="
env FR13_HC_INTERNAL=0 FR13_PARENT_GATHER=0 \
  FR13_DRAFTER_META_REUSE=1 FR13_DRAFTER_META_REUSE_SELFCHECK=0 ENFORCE_EAGER=0 \
  ARMDIR=output/fr13_verify_profile/dmr_graph \
  GATE_CONTAINER=fr13-dmr-graph \
  NEEDLE_PAT="SpecDecoding metrics" \
  bash output/fr13_verify_profile/gate_live_hc_eager.sh
echo "ARM A rc=$?"

echo "===== ARM B: nodmr-graph (REUSE=0) ====="
env FR13_HC_INTERNAL=0 FR13_PARENT_GATHER=0 \
  FR13_DRAFTER_META_REUSE=0 FR13_DRAFTER_META_REUSE_SELFCHECK=0 ENFORCE_EAGER=0 \
  ARMDIR=output/fr13_verify_profile/nodmr_graph \
  GATE_CONTAINER=fr13-nodmr-graph \
  NEEDLE_PAT="SpecDecoding metrics" \
  bash output/fr13_verify_profile/gate_live_hc_eager.sh
echo "ARM B rc=$?"

echo "===== A/B warm_decode_tps ====="
python3 - <<'EOF'
import json, glob
for arm in ("dmr_graph", "nodmr_graph"):
    vals = []
    for f in sorted(glob.glob(f"output/fr13_verify_profile/{arm}/probe_r*.json")):
        d = json.load(open(f))
        for k, v in d.items():
            if isinstance(v, dict) and "warm_decode_tps" in v:
                vals.append(v["warm_decode_tps"])
    m = sum(vals) / len(vals) if vals else None
    print(arm, "rounds:", [round(v, 2) for v in vals], "mean:", round(m, 3) if m else "NONE")
EOF
echo "DMR_SPEED_AB_DONE"

#!/usr/bin/env bash
# g2 leg-ii (LOCAL pre-gate while offload line is down): capture-mode B=1
# probe A/B — subtree path-route vs monolith, same boot config, graph mode.
# Signal = warm_decode_tps across 3 probe rounds each (same prompts, seed
# 1313). Selfcheck OFF (host compare syncs; capture-illegal). Byte identity
# already proven by the eager gate this morning. Capture-first-call coverage:
# subtree_preseed at builder init is exactly what arm A validates under graph.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel

echo "===== ARM A: subtree-graph ====="
env FR13_HC_INTERNAL=0 FR13_PARENT_GATHER=0 \
  FR13_SUBTREE_PARALLEL=1 FR13_SUBTREE_PARALLEL_SELFCHECK=0 ENFORCE_EAGER=0 \
  ARMDIR=output/fr13_verify_profile/subtree_graph \
  GATE_CONTAINER=fr13-subtree-graph \
  NEEDLE_PAT="FR13_SUBTREE_PARALLEL] preseeded" \
  bash output/fr13_verify_profile/gate_live_hc_eager.sh
rcA=$?
echo "ARM A rc=$rcA"

echo "===== ARM B: monolith-graph ====="
env FR13_HC_INTERNAL=0 FR13_PARENT_GATHER=0 \
  FR13_SUBTREE_PARALLEL=0 FR13_SUBTREE_PARALLEL_SELFCHECK=0 ENFORCE_EAGER=0 \
  ARMDIR=output/fr13_verify_profile/mono_graph \
  GATE_CONTAINER=fr13-mono-graph \
  NEEDLE_PAT="SpecDecoding metrics" \
  bash output/fr13_verify_profile/gate_live_hc_eager.sh
rcB=$?
echo "ARM B rc=$rcB"

echo "===== A/B warm_decode_tps ====="
python3 - <<'EOF'
import json, glob
for arm in ("subtree_graph", "mono_graph"):
    vals = []
    for f in sorted(glob.glob(f"output/fr13_verify_profile/{arm}/probe_r*.json")):
        d = json.load(open(f))
        for k, v in d.items():
            if isinstance(v, dict) and "warm_decode_tps" in v:
                vals.append(v["warm_decode_tps"])
    m = sum(vals) / len(vals) if vals else None
    print(arm, "rounds:", [round(v, 2) for v in vals], "mean:", round(m, 3) if m else "NONE")
EOF
echo "SUBTREE_SPEED_AB_DONE"

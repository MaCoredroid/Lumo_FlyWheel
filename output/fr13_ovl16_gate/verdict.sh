#!/usr/bin/env bash
# ovl16 overlap-gate three-leg verdict (run AFTER driver PID exits).
# Baselines: rwb1 (same code class + B1/B2a, no overlap): accept 4.520,
# measured_tps_fullstep_wall 38.01, eps 3.295, committer 48.6ms, 8P/8F.
# kvr1: accept 4.286, tps 32.85, 10P/6F. Golden accept band 4.286..4.52+noise.
# Bake rule (feedback_bake_on_golden_signal): pass band 8-9/16 wall-free => bake.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
ARM=output/fr13_ovl16_gate/ovl16_tail6_ov1

echo "=== leg 2: pass band (report as X pass, Y fail, X+Y finished) ==="
P=0; F=0
for f in "$ARM"/swe_out/verified/per_task/*/eval/eval_report.json; do
  v=$(python3 -c "import json;print(json.load(open('$f')).get('passed'))" 2>/dev/null)
  case "$v" in True) P=$((P+1));; False) F=$((F+1));; esac
done
echo "$P pass, $F fail, $((P+F)) finished (kvr1 10/6, rwb1 8/8)"

echo "=== legs 1+3: accept + speed (fr13_measure deploy-speed json) ==="
J=$(ls "$ARM"/deploy_speed_*.json 2>/dev/null | head -1)
if [ -z "$J" ]; then
  echo "deploy_speed json MISSING — run:"
  echo "  .venv/bin/python scripts/fr13_measure.py deploy-speed --arm ovl16_tail6_ov1 \\"
  echo "    --out-root $ARM/swe_out --expected-tok-per-draft 21 --batch-size 4 --out $ARM/deploy_speed_ov1.json"
else
  python3 - "$J" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1]))
def g(*ks):
    for k in ks:
        if k in j: return j[k]
    return None
print(json.dumps({k: j[k] for k in j if isinstance(j[k], (int, float, str))}, indent=1)[:1200])
print("\nCOMPARE: accept vs rwb1 4.520 / kvr1 4.286 | tps_fullstep_wall vs rwb1 38.01 | eps vs 3.295")
EOF
fi

echo "=== committer span (same basis as rwb1 48.6ms) ==="
python3 - <<'EOF'
import json
try:
    j = json.load(open("output/fr13_sfwd_sidecar/ovl16_tail6_ov1_cfwd.json.216"))
    print(f"committer: {1000*j['gpu_seconds']/j['n_spans']:.1f}ms/event over {j['n_spans']} spans (rwb1 48.6)")
except Exception as e:
    print("cfwd sidecar read failed:", e)
EOF
echo "=== red-team checklist: engagement needle count, WALL=0 confirmed, temp 0.6, ==="
echo "=== eval failures eyeballed for garble/giveup before any bake decision      ==="

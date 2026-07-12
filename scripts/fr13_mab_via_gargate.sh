#!/bin/bash
# fr13_mab_via_gargate.sh — MAB co-residency localization DURING real garbling decodes.
#
# The prior CAPTURE_ONLY attempt fired 0 events (warmup too shallow). This boots cat8 via the
# reusable garble gate (which drives N real temp-0.6 garbling decodes) with FR13_GDN_SUBOP_MAB armed,
# so the M10-vs-M5 sub-op localizer fires at layer-0 DURING actual garbles. FR13_RUN_DIR routes the
# container /logs to a host dir so the MAB dump survives teardown. EXPECT_TREE_N='*' accepts cat8's
# tree_n (default 10 = cat9 would engage-fail). One boot => localization + the garble SCORE together.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$PWD/output/fr13_mab_coresidency/gargate_$STAMP"
mkdir -p "$RUN/logs"
echo "=== MAB via garble-gate $STAMP -> $RUN ==="
if [[ -n "$(docker ps -q)" ]]; then echo "FAIL: docker not empty"; docker ps; exit 2; fi

# MAB is EAGER-ONLY (host .item()/json syncs); it is wrong under FULL graph capture AND the
# graph boot overruns the gate's 720s health window. ENFORCE_EAGER=1 = correct mode + fast boot.
# Eager localizes the M-dependent sub-op fine (M-dependence is in the math, capture-mode-agnostic);
# the FIX is validated on the graph ship config. N small (enough for LIMIT layer-0 events).
N=6 bash scripts/fr13_flag_garble_gate.sh mabloc \
  "FR13_RUN_DIR=$RUN" \
  ENFORCE_EAGER=1 \
  FR13_GDN_SUBOP_MAB=1 \
  FR13_GDN_SUBOP_MAB_DUMP=/logs/fr13_gdn_subop_mab.jsonl \
  'FR13_GDN_SUBOP_MAB_EXPECT_TREE_N=*' \
  FR13_GDN_SUBOP_MAB_LIMIT=40 \
  > "$RUN/gargate.log" 2>&1
echo "  gate rc=$? ; score: $(grep -iE 'SCORE' "$RUN/gargate.log" | tail -1)"

DUMP="$RUN/logs/fr13_gdn_subop_mab.jsonl"
if [[ ! -s "$DUMP" ]]; then
  echo "FAIL: no MAB dump at $DUMP"; ls -la "$RUN/logs" 2>/dev/null
  echo "--- engage-fail lines from boot? ---"; grep -iE 'engage-fail|SUBOP_STAGE|MAB' "$RUN"/../gargate_*/logs/* "$RUN/gargate.log" 2>/dev/null | tail -10
  exit 3
fi
echo "=== MAB co-residency verdict (M10=branches present vs M5=absent, same deep node) ==="
python3 - "$DUMP" <<'PY'
import json,sys
from collections import Counter
recs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(f"events: {len(recs)}")
hist=Counter(); histm1=Counter()
for r in recs:
    s=r.get('subops',{})
    g=lambda n: s.get(n,{}).get('m10_deep_vs_m5_deep_max_abs')
    print(f"  ev{r.get('capture_event_index')} tn={r.get('tree_n')} nacc={r.get('num_accepted_tokens')} deep={r.get('deep_row')} | "
          f"M10vsM5: pre_conv={g('pre_conv')} conv1d_out={g('conv1d_out')} scan_out={g('scan_out')} | "
          f"first={r.get('first_coresidency_subop_m10_vs_m5')} first_m5v1={r.get('first_subop_m5_vs_m1')} pcMinv={r.get('pre_conv_m_invariant')}")
    hist[r.get('first_coresidency_subop_m10_vs_m5')]+=1
    histm1[r.get('first_subop_m5_vs_m1')]+=1
print("VERDICT first_coresidency_subop_m10_vs_m5:", dict(hist))
print("        first_subop_m5_vs_m1        :", dict(histm1))
PY
echo "=== MAB VIA GARGATE DONE $RUN ==="

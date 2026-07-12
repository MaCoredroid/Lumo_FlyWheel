#!/bin/bash
# fr13_mab_coresidency_localize.sh — the CLEAN co-residency birthplace on the CURRENT served build.
#
# FR13_GDN_SUBOP_MAB re-runs each L0 sub-op (pre_conv -> conv1d_out -> scan_out) for the SAME deep
# node with branch rows PRESENT (M10=full cat8 tree) vs ABSENT (M5=spine only). The first sub-op whose
# deep-row M10-vs-M5 RAW max_abs != 0 = where branch co-residency corrupts the forward = the garble seed
# birthplace. Deep node's context is otherwise identical => clean control (not a reference confound).
#
# Works on the served replay path (FR13_REPLAY_ROUTE=1): the MAB re-runs the scan itself, needs NO
# deleted scratch and NO payload capture (unlike the rotted branch-path oracle). Threshold 0.0 = raw
# int-view, NOT atol. Deterministic capture; no temp-0.6, no SWE agent.
#
#   READ: first_coresidency_subop_m10_vs_m5 across deep events (nacc>1):
#     conv1d_out => branch co-residency corrupts the conv (despite path-correct windows) = fixable there
#     scan_out (conv=0) => corrupts the scan input state = the recurrent-state co-residency
#     None (all 0) => conv/scan are M-invariant on the current build => seed is in_proj_ba (MAB-blind,
#         but padded) or the cross-event state handoff => re-aim.
set -uo pipefail
REPO=/home/mark/shared/lumoFlyWheel; cd "$REPO"
source "$REPO/.lumo.local.env" 2>/dev/null || true
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
RUNROOT=${RUNROOT:-output/fr13_mab_coresidency/run_$STAMP}
SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_carrier_four.json}
ARM=mab
mkdir -p "$RUNROOT"
echo "=== MAB CO-RESIDENCY LOCALIZE $STAMP (served cat8; scan re-run M10 vs M5 vs M1) ==="
if [[ -n "$(docker ps -q)" ]]; then echo "FAIL: docker not empty before capture"; docker ps; exit 2; fi

env RUNROOT="$RUNROOT" CAPTURE_ONLY=1 \
    FR13_REPLAY_ROUTE=1 \
    FR13_GDN_SUBOP_MAB=1 \
    FR13_GDN_SUBOP_MAB_DUMP=/logs/fr13_gdn_subop_mab.jsonl \
    FR13_GDN_SUBOP_MAB_LIMIT=16 \
    FR13_GDN_SUBOP_MAB_THRESHOLD=0.0 \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" cat8 "$SUBSET" \
    > "$RUNROOT/${ARM}.log" 2>&1
RC=$?
echo "capture rc=$RC"
[[ -n "$(docker ps -q)" ]] && docker rm -f "fr13-bigdenom-$ARM" >/dev/null 2>&1 || true

DUMP="$RUNROOT/$ARM/logs/fr13_gdn_subop_mab.jsonl"
if [[ ! -s "$DUMP" ]]; then
  echo "FAIL: no MAB dump at $DUMP — instrument did not fire. tail:"; tail -35 "$RUNROOT/${ARM}.log"; exit 3
fi
echo "=== MAB co-residency verdict (per deep event) ==="
python3 - "$DUMP" <<'PY'
import json,sys
from collections import Counter
recs=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(f"events: {len(recs)}")
hist=Counter()
for r in recs:
    s=r.get('subops',{})
    g=lambda n: s.get(n,{}).get('m10_deep_vs_m5_deep_max_abs')
    print(f"  ev{r.get('capture_event_index')} tree_n={r.get('tree_n')} nacc={r.get('num_accepted_tokens')} deep={r.get('deep_row')} | "
          f"M10vsM5: pre_conv={g('pre_conv')} conv1d_out={g('conv1d_out')} scan_out={g('scan_out')} | "
          f"first={r.get('first_coresidency_subop_m10_vs_m5')} first_m5v1={r.get('first_subop_m5_vs_m1')} pcMinv={r.get('pre_conv_m_invariant')}")
    hist[r.get('first_coresidency_subop_m10_vs_m5')]+=1
print("VERDICT first_coresidency_subop_m10_vs_m5 histogram:", dict(hist))
PY
echo "MAB CO-RESIDENCY LOCALIZE DONE run=$RUNROOT"

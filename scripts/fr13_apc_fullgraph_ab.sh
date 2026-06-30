#!/usr/bin/env bash
# FR13 full-graph A/B: PIECEWISE (control, known on-task) vs FULL_AND_PIECEWISE (test) on the
# deterministic temp-0 12907 replay. Two proofs: (1) BIT-EXACT output diff (identical canon per
# turn => full graph is lossless, not just garble-free), (2) decode-TPS delta (the speed prize).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
TMP=/home/mark/.claude/jobs/22c39bb9/tmp
echo "=== FULL-GRAPH A/B  PIECEWISE vs FULL_AND_PIECEWISE  (EXACT_SEED+block1024, temp-0.6 12907 replay) ==="
echo "--- run 1/2: PIECEWISE (control) @ $(date -u +%H:%M:%S) ---"
CGMODE=PIECEWISE bash scripts/fr13_apc_fullgraph_probe.sh || echo "PIECEWISE run rc=$?"
RD_PW=$(cat "$TMP/fullgraph_probe_root.txt" 2>/dev/null)
echo "--- run 2/2: FULL_AND_PIECEWISE (test) @ $(date -u +%H:%M:%S) ---"
CGMODE=FULL_AND_PIECEWISE bash scripts/fr13_apc_fullgraph_probe.sh || echo "FULL run rc=$?"
RD_FULL=$(cat "$TMP/fullgraph_probe_root.txt" 2>/dev/null)

echo "=================== A/B RESULT ==================="
echo "PIECEWISE rundir: $RD_PW"
echo "FULL      rundir: $RD_FULL"
echo "--- (1) GARBLE check per arm @ temp 0.6 (bit-exact diff is N/A at temp>0 by design; true losslessness = the LIVE e2e gate resolve-parity, not the replay) ---"
for tag in PW FULL; do
  [ "$tag" = PW ] && RD="$RD_PW" || RD="$RD_FULL"
  .venv/bin/python - "$RD/replay.json" "$tag" <<'PY'
import json,sys,re
try: recs=json.load(open(sys.argv[1])).get('records',[])
except Exception as e: print(sys.argv[2],"load fail",e); raise SystemExit
txt=" ".join((r.get('canon') or '') for r in recs)
cjk=len(re.findall(r'[一-鿿぀-ヿ가-힯]{4,}', txt))
ncached=sum(1 for r in recs if (r.get('cached_tokens') or 0)>0)
print(f"{sys.argv[2]}: turns={len(recs)} cached={ncached} CJK_garble={cjk} char8={txt.count('char 8')} -> {'CLEAN' if cjk==0 else 'GARBLE'}")
PY
done
echo "--- (2) decode TPS (metrics-based) PIECEWISE vs FULL ---"
for tag in PW FULL; do
  [ "$tag" = PW ] && RD="$RD_PW" || RD="$RD_FULL"
  .venv/bin/python - "$RD/metrics_pre.txt" "$RD/metrics_post.txt" "$tag" <<'PY'
import sys,re
def p(f):
    d={}
    try:
        for ln in open(f):
            ln=ln.strip()
            if ln.startswith('#') or not ln: continue
            m=re.match(r'(\S+?)(\{[^}]*\})?\s+([0-9eE.+-]+)$',ln)
            if m: d[m.group(1)]=d.get(m.group(1),0.0)+float(m.group(3))
    except Exception: pass
    return d
a=p(sys.argv[1]); b=p(sys.argv[2]); tag=sys.argv[3]
def df(k): return b.get(k,0)-a.get(k,0)
gen=df('vllm:generation_tokens_total'); dt=df('vllm:request_decode_time_seconds_sum')
acc=df('vllm:spec_decode_num_accepted_tokens_total'); drf=df('vllm:spec_decode_num_draft_tokens_total')
if dt>0:
    s=f"{tag}: DECODE TPS = {gen/dt:.1f} tok/s  (gen {int(gen)} / {dt:.1f}s)"
    if drf>0: s+=f"  accept {acc/drf*100:.0f}%"
    print(s)
else: print(f"{tag}: metrics_post empty/unusable (gen={int(gen)})")
PY
done
echo "=================== A/B DONE @ $(date -u +%H:%M:%S) ==================="

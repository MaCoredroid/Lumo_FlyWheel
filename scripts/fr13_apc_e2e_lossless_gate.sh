#!/usr/bin/env bash
# FR13 APC SPEC+CACHE E2E LOSSLESS GATE — the BINDING gate (real SWE-Verified, temp 0.6).
#
# The goal: spec-decode (cat6root tree+spine) + APC prefix cache, LOSSLESS, both speedups.
# This is the ONLY gate that judges that goal — a real agentic SWE-Verified task at the
# DEPLOYMENT condition (temp 0.6, NO teacher forcing). State-faithfulness ladders run on
# FIXED replay inputs and do NOT reproduce the e2e derail, so they are diagnostics, not
# the gate.
#
# SINGLE VARIABLE: toggle ONLY FR13_ENABLE_APC (cache ON vs OFF). Everything else held
# equal — cat6root spec, B=1, cuda-graph PIECEWISE, temp 0.6, offload, same git HEAD,
# baked deployed fixes (NO experimental SNAP_FIX flags; those were confounds). N independent
# boots per arm => N independent rollouts (each exercises within-task cross-turn cache reuse).
#
# NON-VACUITY (the rategate's blind spot — its noted "breaking turn ran at 0% cache hit"):
#   - ON arm MUST show real cache hits (cached_tokens>0) or the rollout is VACUOUS (cache
#     never engaged -> the comparison is meaningless). Recorded + flagged per rollout.
#   - spec MUST be engaged (serve_variant asserts spec_drafts delta>0 or it exits 4).
#   - temp 0.6 pinned + asserted by serve_variant (LUMO_PROXY_FORCE_TEMPERATURE).
#   - GARBLE (char8 / CJK) measured at the codex_trace (NOT the engine log) to separate a
#     real cache-state derail (garble) from the cache-INDEPENDENT qwen tool-call flake.
#
# VERDICT: cache is a real problem ONLY if cache-ON resolve-rate is statistically BELOW
# cache-OFF (Fisher) or garble-rate ABOVE. Match within band => spec+cache is lossless e2e.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true

N=${N:-5}                                    # rollouts per arm
ARMS=${ARMS:-"ON OFF"}                        # ON=cache+spec ; OFF=no-cache+spec (single var)
KIND=${KIND:-cat6root}
SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_astropy12907.json}
OFFLOAD_AGENT=${OFFLOAD_AGENT:-${OFFLOAD_CODEX:-1}}
DOCKER_MEM_CAP=${DOCKER_MEM_CAP:-105g}; export DOCKER_MEM_CAP
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT=${RUNROOT:-output/fr13_apc_e2e_gate/run_${TS}}
mkdir -p "$RUNROOT"
echo "$RUNROOT" > /tmp/claude-1000/-home-mark-shared/46f03809-5059-4e30-936d-1adda7f44337/scratchpad/e2e_gate_root.txt
SUMMARY="$RUNROOT/e2e_gate_rollouts.tsv"
printf 'arm\trollout\tverdict\tcached_max\tspec_ok\tchar8\ttrace_bytes\trun_log\n' > "$SUMMARY"
echo "=== FR13 APC SPEC+CACHE E2E LOSSLESS GATE  N=$N/arm  arms='$ARMS'  KIND=$KIND  subset=$SUBSET ==="
echo "    runroot=$RUNROOT  offload=$OFFLOAD_AGENT  memcap=$DOCKER_MEM_CAP"

_hygiene() {
  pgrep -af 'oom_protect_watchdog\.sh' | grep -qv pgrep || { setsid bash scripts/oom_protect_watchdog.sh >/dev/null 2>&1 </dev/null & disown; }
  bash scripts/oom_protect_session.sh >/dev/null 2>&1 || true
  PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
  echo "  [hygiene] avail=$(free -g|awk '/Mem/{print $7}')GiB watchdog=$(pgrep -f oom_protect_watchdog|head -1||echo DEAD)"
}

for i in $(seq 1 "$N"); do
  for arm in $ARMS; do
    case "$arm" in
      ON)  APC=1 ;;   # deployed cache-ON (baked CONV_FIX/CONV_SNAPSHOT/SSM_SNAPSHOT) + spec
      OFF) APC=0 ;;   # no prefix cache + spec  (the lossless reference)
      *) echo "unknown arm $arm"; exit 2;;
    esac
    ARM="eg_${arm}_r${i}"
    RLOG="$RUNROOT/${ARM}.log"
    echo "--- [$arm rollout $i/$N] FR13_ENABLE_APC=$APC arm=$ARM ---"
    _hygiene
    docker rm -f "fr13-bigdenom-$ARM" >/dev/null 2>&1 || true
    FR13_ENABLE_APC=$APC FR13_APC_CONFIG_ONLY=0 \
      MAMBA_BLOCK_SIZE="${MAMBA_BLOCK_SIZE:-8192}" MAMBA_SSM_CACHE_DTYPE="${MAMBA_SSM_CACHE_DTYPE:-float32}" \
      APC_MAX_NUM_BATCHED_TOKENS="${APC_MAX_NUM_BATCHED_TOKENS:-1024}" \
      DEPLOY_FORCE_TEMP=0.6 OFFLOAD_AGENT="$OFFLOAD_AGENT" MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 \
      FR10_METRICS=0 BATCH_INVARIANT=0 DOCKER_MEM_CAP="$DOCKER_MEM_CAP" RUNROOT="$RUNROOT" \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" "$KIND" "$SUBSET" > "$RLOG" 2>&1 </dev/null
    # --- parse ---
    VERD=$(grep -hoE "resolved_rate=[0-9.]+" "$RLOG" 2>/dev/null | tail -1 | grep -qE "resolved_rate=1" && echo resolved || echo failed)
    SPEC_OK=$(grep -qE "spec engagement OK" "$RLOG" 2>/dev/null && echo 1 || echo 0)
    # real cache hits: max cached_tokens anywhere in this arm's artifacts (proxy/trace)
    CMAX=$(grep -rhoE "\"cached_tokens\"\s*:\s*[0-9]+" "$RUNROOT/$ARM" "$RLOG" 2>/dev/null | grep -oE "[0-9]+" | sort -rn | head -1)
    CMAX=${CMAX:-0}
    TR=$(find "$RUNROOT/$ARM" \( -path "*per_task*/agent_trace.jsonl" -o -path "*per_task*/codex_trace.jsonl" \) 2>/dev/null | head -1)
    C8=0; TB=0
    [[ -n "$TR" ]] && { C8=$(grep -acE 'Unterminated|column 9 \(char 8\)|EOF while parsing a string|[\x{4e00}-\x{9fff}]' "$TR" 2>/dev/null || echo 0); TB=$(wc -c <"$TR" 2>/dev/null || echo 0); }
    printf '%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$i" "$VERD" "$CMAX" "$SPEC_OK" "$C8" "$TB" "$RLOG" >> "$SUMMARY"
    VAC=""; { [ "$arm" = "ON" ] && [ "${CMAX:-0}" -le 0 ] 2>/dev/null; } && VAC=" !!VACUOUS(0 cache hits)"
    echo "  => verdict=$VERD cached_max=$CMAX spec_ok=$SPEC_OK char8=$C8$VAC"
    docker rm -f "fr13-bigdenom-$ARM" >/dev/null 2>&1 || true
  done
done

echo "=== E2E GATE DONE — reducing ==="
.venv/bin/python - "$SUMMARY" <<'PY'
import sys, csv
from collections import defaultdict
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
agg=defaultdict(lambda:{'n':0,'resolved':0,'char8':0,'cache_hit':0,'spec_ok':0,'vac':0})
for r in rows:
    a=agg[r['arm']]; a['n']+=1
    a['resolved']+=(r['verdict']=='resolved'); a['char8']+=(int(r['char8'])>0)
    ch=int(r['cached_max'])>0; a['cache_hit']+=ch; a['spec_ok']+=(r['spec_ok']=='1')
    if r['arm']=='ON' and not ch: a['vac']+=1
print("\n=== FR13 APC SPEC+CACHE E2E LOSSLESS GATE VERDICT ===")
for arm in ('OFF','ON'):
    a=agg[arm]
    if a['n']:
        print(f"  {arm}: resolved {a['resolved']}/{a['n']} ({a['resolved']/a['n']:.0%})  "
              f"garble {a['char8']}/{a['n']}  cache_hit {a['cache_hit']}/{a['n']}  spec_ok {a['spec_ok']}/{a['n']}"
              + (f"  !!{a['vac']} VACUOUS(0 hits)" if arm=='ON' and a['vac'] else ""))
on,off=agg['ON'],agg['OFF']
if on['n'] and on['cache_hit']==0:
    print("  GATE INVALID: ON arm never got cache hits -> vacuous, fix cache engagement before judging.")
try:
    from scipy.stats import fisher_exact
    if on['n'] and off['n']:
        _,p=fisher_exact([[on['resolved'],on['n']-on['resolved']],[off['resolved'],off['n']-off['resolved']]])
        print(f"  Fisher resolve ON vs OFF: p={p:.3f} => {'LOSSLESS (no sig diff)' if p>0.05 else 'CACHE DEGRADES — real bug'}")
        _,p2=fisher_exact([[on['char8'],on['n']-on['char8']],[off['char8'],off['n']-off['char8']]])
        print(f"  Fisher garble ON vs OFF:  p={p2:.3f} => {'no sig garble diff' if p2>0.05 else 'cache-linked garble'}")
except Exception as e:
    print(f"  (scipy unavailable: {e}; compare rates by hand)")
print("====================================================")
PY
echo "summary: $SUMMARY"

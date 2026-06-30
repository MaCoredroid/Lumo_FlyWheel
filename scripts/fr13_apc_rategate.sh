#!/usr/bin/env bash
# FR13 APC RATE GATE — the HONEST single-variable agentic A/B (rebuilt 2026-06-24).
#
# WHY: the prior "cache-ON breaks 12907 / cache-OFF solves" matrix was CONFOUNDED
# (7-agent audit wuez11596): n=1 per arm at temp=0.6 with NO seed, the failure is the
# cache-INDEPENDENT qwen tool-call flake (a free-form runaway that 400s at the generic
# chat_utils json.loads INPUT path used for ALL tool calls — there is no apply-patch
# code path for the cache to uniquely poison), and the breaking turn ran at 0% cache hit.
# A single temp=0.6 draw per arm is a coin flip (146 historical 12907 runs ~= 74 fail/69
# resolve). So we measure a RATE, not a draw.
#
# DESIGN: real agentic SWE-Verified at temp=0.6 (the deployment condition — NO teacher
# forcing). Toggle ONLY FR13_ENABLE_APC (cache on vs off), holding EVERYTHING else equal
# (cat6root tree, B=1, cuda-graph, offload, same git HEAD). VANILLA deployment cache-on
# (FR13_ENABLE_APC=1 -> baked CONV_FIX/CONV_SNAPSHOT/SSM_SNAPSHOT) — NO experimental
# SNAP_FIX/LEAF_SRC/HIT_RECURRENT_SUFFIX (those were themselves confounds; the audit showed
# the snapshot carrier did not even fire on the failing turn). N independent boots per arm
# => N independent rollouts (each exercises within-task cross-turn cache reuse), clean rate.
#
# VERDICT: cache is a real problem ONLY if cache-ON resolve-rate is statistically BELOW
# cache-OFF (Fisher exact) OR cache-ON char-8 rate is statistically ABOVE cache-OFF. If they
# match within the band, APC is shippable and the "cache breaks it" story was noise.
#
# OOM-HARDENED: DOCKER_MEM_CAP caps the container cgroup so it self-OOMs (relaunchable)
# instead of the host taking Claude; watchdog kept alive; recover_host_memory between boots.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel

N=${N:-6}                                  # rollouts per arm
SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_astropy12907.json}
RUNROOT=${RUNROOT:-output/fr13_apc_rategate/run_$(date -u +%Y%m%dT%H%M%SZ 2>/dev/null || echo manual)}
DOCKER_MEM_CAP=${DOCKER_MEM_CAP:-105g}
export DOCKER_MEM_CAP
mkdir -p "$RUNROOT"
SUMMARY="$RUNROOT/rategate_rollouts.tsv"
printf 'arm\trollout\tverdict\tchar8\ttrace_bytes\trun_log\n' > "$SUMMARY"
echo "=== FR13 APC RATE GATE  N=$N/arm  subset=$SUBSET  runroot=$RUNROOT  memcap=$DOCKER_MEM_CAP ==="

_hygiene() {
  # keep Claude protected + reclaim wedged unified mem before each boot
  pgrep -af 'oom_protect_watchdog\.sh' | grep -qv pgrep || { setsid bash scripts/oom_protect_watchdog.sh >/dev/null 2>&1 </dev/null & disown; }
  bash scripts/oom_protect_session.sh >/dev/null 2>&1 || true
  PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
  local avail; avail=$(free -g | awk '/Mem/{print $7}')
  echo "  [hygiene] avail=${avail}GiB watchdog=$(pgrep -f oom_protect_watchdog | head -1 || echo DEAD) claude_oom=$(cat /proc/$(pgrep -x claude|head -1)/oom_score_adj 2>/dev/null)"
}

ARMS=${ARMS:-"ON CFG OFF"}   # ON=full cache; CFG=APC config (chunked+1024) NO cache; OFF=base
for i in $(seq 1 "$N"); do
  for arm in $ARMS; do
    # 3-arm isolation: does the rate-gate failure come from the cache, or the APC serve config?
    case "$arm" in
      ON)  APC=1; CFG=0 ;;   # full cache-ON (chunked+1024+mamba+caching)
      CFG) APC=1; CFG=1 ;;   # config-only: chunked+1024, NO --enable-prefix-caching
      OFF) APC=0; CFG=0 ;;   # base (no chunked/1024/cache)
    esac
    ARM="rg_${arm}_r${i}"
    RLOG="$RUNROOT/${ARM}.log"
    echo "--- [$arm rollout $i/$N] FR13_ENABLE_APC=$APC CONFIG_ONLY=$CFG arm=$ARM ---"
    _hygiene
    # belt: remove any stale container of this name
    docker rm -f "fr13-bigdenom-$ARM" >/dev/null 2>&1 || true
    FR13_ENABLE_APC=$APC FR13_APC_CONFIG_ONLY=$CFG MAMBA_BLOCK_SIZE=1024 MAMBA_SSM_CACHE_DTYPE=float32 \
      DEPLOY_FORCE_TEMP=0.6 OFFLOAD_CODEX=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 \
      FR10_METRICS=0 BATCH_INVARIANT=0 DOCKER_MEM_CAP="$DOCKER_MEM_CAP" \
      RUNROOT="$RUNROOT" \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" cat6root "$SUBSET" > "$RLOG" 2>&1 </dev/null
    # parse verdict + char-8 from the rollout artifacts
    V=$(grep -hoE "resolved_rate=[0-9.]+" "$RLOG" 2>/dev/null | tail -1)
    VERD=$(echo "$V" | grep -qE "resolved_rate=1" && echo resolved || echo failed)
    TR=$(find "$RUNROOT/$ARM" -path "*per_task*/codex_trace.jsonl" 2>/dev/null | head -1)
    C8=0; TB=0
    [[ -n "$TR" ]] && { C8=$(grep -cE 'Unterminated|column 9 \(char 8\)|EOF while parsing a string' "$TR" 2>/dev/null); TB=$(wc -c <"$TR" 2>/dev/null); }
    printf '%s\t%d\t%s\t%s\t%s\t%s\n' "$arm" "$i" "$VERD" "$C8" "$TB" "$RLOG" >> "$SUMMARY"
    echo "  => verdict=$VERD char8=$C8 trace_bytes=$TB"
    docker rm -f "fr13-bigdenom-$ARM" >/dev/null 2>&1 || true
  done
done

echo "=== RATE GATE DONE — reducing ==="
.venv/bin/python - "$SUMMARY" <<'PY'
import sys, csv
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
agg = defaultdict(lambda: {'n':0,'resolved':0,'char8':0})
for r in rows:
    a = agg[r['arm']]; a['n']+=1
    a['resolved'] += (r['verdict']=='resolved')
    a['char8'] += (int(r['char8'])>0)
print("\n=== FR13 APC RATE GATE VERDICT ===")
for arm in ('OFF','ON'):
    a=agg[arm]
    if a['n']: print(f"  cache-{arm}: resolved {a['resolved']}/{a['n']} ({a['resolved']/a['n']:.0%})  char8 {a['char8']}/{a['n']} ({a['char8']/a['n']:.0%})")
# Fisher exact on resolve counts (ON vs OFF)
try:
    from scipy.stats import fisher_exact
    on,off=agg['ON'],agg['OFF']
    if on['n'] and off['n']:
        # table: [[on_resolved, on_failed],[off_resolved, off_failed]]
        tbl=[[on['resolved'], on['n']-on['resolved']],[off['resolved'], off['n']-off['resolved']]]
        _,p=fisher_exact(tbl)
        print(f"  Fisher-exact (resolve ON vs OFF): p={p:.3f}  => {'NO significant difference (cache shippable)' if p>0.05 else 'SIGNIFICANT difference — investigate'}")
        tbl2=[[on['char8'], on['n']-on['char8']],[off['char8'], off['n']-off['char8']]]
        _,p2=fisher_exact(tbl2)
        print(f"  Fisher-exact (char8 ON vs OFF):   p={p2:.3f}  => {'NO significant char8 difference' if p2>0.05 else 'SIGNIFICANT char8 difference — cache-linked'}")
except Exception as e:
    print(f"  (scipy unavailable for Fisher: {e}; compare rates by hand)")
print("==================================")
PY
echo "summary: $SUMMARY"

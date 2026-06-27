#!/usr/bin/env bash
# RE-MEASURE the APC residual on the CURRENT 06-27 baked stack + A/B HIT_RECURRENT_SUFFIX 0 vs 1.
# Decided by the breadth-survey adversarial verify (workflow wzyghz2b6): the +3.12pp residual is
# UNPROVEN (n=512 overlapping CIs, pre-bake), and HIT_RECURRENT_SUFFIX=1 (sequential rank-1 roll)
# may be a NET CARRIER vs the chunked-WY kernel cache-OFF uses.
#
# NON-DESTRUCTIVE: changes NO baked code. Each arm only PRE-SETS FR13_APC_HIT_RECURRENT_SUFFIX in the
# env; the launcher := keeps a pre-set value, so the bake is untouched on disk.
#
# Binding metric: per-token CLEAR-MARGIN argmax-flip vs the no-spec RECURRENT decode oracle, SAME-BOOT
# cache-ON vs cache-OFF (the precheck captures both in one boot). NOT SWE-rate, NOT a proxy.
#   STEP 0: is cache-ON(HRS=1) clear-margin <= cache-OFF (within-floor)?  if yes -> already lossless.
#   STEP 1: does HRS=0 flip FEWER than HRS=1?  if yes -> the recurrent roll is a net carrier -> un-bake.
# Arms: HRS=1 (current baked default) and HRS=0 (subtractive). Each captures cache-ON + cache-OFF.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
export MAX_TOKENS=${MAX_TOKENS:-2000} IGNORE_EOS=1 TEMP=0.6 SEED=1313 TOP_K=20
TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=output/fr13_apc_remeasure/run_$TS
mkdir -p "$ROOT"
echo "$ROOT" > /tmp/claude-1000/-home-mark-shared/46f03809-5059-4e30-936d-1adda7f44337/scratchpad/remeasure_root.txt 2>/dev/null || true
PRECHECK_ROOT=output/fr13_apc_temp06
echo "APC RE-MEASURE + HRS A/B (binding=clear-margin-vs-recurrent-oracle, same-boot cache-ON vs cache-OFF), n>=$MAX_TOKENS -> $ROOT" | tee "$ROOT/RESULTS.txt"

capture_arm() {  # $1=HRS value
  local HRS=$1 A="hrs$1"
  local AR="$ROOT/$A"; mkdir -p "$AR"
  echo "=== [capture $A] cache-ON(HRS=$HRS)+cache-OFF same-boot $(date -u +%H:%M:%SZ) ===" | tee -a "$ROOT/RESULTS.txt"
  # PRE-SET HRS so the launcher := keeps it (bake on disk untouched). cap stays baked 1e6.
  FR13_APC_HIT_RECURRENT_SUFFIX=$HRS bash scripts/fr13_apc_temp06_precheck.sh > "$AR/precheck.log" 2>&1 || true
  # copy the same-boot captures out of the fixed precheck dir before the next arm overwrites them
  cp "$PRECHECK_ROOT/cat9_apc_on_src.json"  "$AR/on_src.json"  2>/dev/null || echo "MISS on_src $A" | tee -a "$ROOT/RESULTS.txt"
  cp "$PRECHECK_ROOT/cat9_apc_off_src.json" "$AR/off_src.json" 2>/dev/null || echo "MISS off_src $A" | tee -a "$ROOT/RESULTS.txt"
  # ENGAGEMENT: cache fired (precheck non-vacuity) + HRS at intended value in the boot
  local fired; fired=$(grep -cE "non-vacuity. OK: cache fired" "$AR/precheck.log" 2>/dev/null || echo 0)
  local hrsseen; hrsseen=$(grep -oE "HIT_RECURRENT_SUFFIX=[01]" "$AR/precheck.log" 2>/dev/null | tail -1)
  echo "  engagement: cache_fired=$fired  marker[$hrsseen] (want HIT_RECURRENT_SUFFIX=$HRS)" | tee -a "$ROOT/RESULTS.txt"
  local non; non=$(grep -E "served_token_ids n=" "$AR/precheck.log" 2>/dev/null | tail -1)
  echo "  $non" | tee -a "$ROOT/RESULTS.txt"
}

rescore() {  # $1=arm-label $2=src $3=out
  [ -s "$2" ] || { echo "SKIP rescore $1 (no src $2)" | tee -a "$ROOT/RESULTS.txt"; return; }
  bash scripts/fr13_recur_rescore_in_container.sh "$1" "$2" "$3" > "${3%.json}.log" 2>&1 || echo "rescore $1 rc=$?" | tee -a "$ROOT/RESULTS.txt"
}

# --- STEP 0+1: capture both arms (each = cache-ON + cache-OFF same boot) ---
capture_arm 1
capture_arm 0

# --- rescore the 4 streams vs the no-spec recurrent oracle (separate GPU boots) ---
rescore on_hrs1  "$ROOT/hrs1/on_src.json"  "$ROOT/hrs1/on_rescore.json"
rescore off_hrs1 "$ROOT/hrs1/off_src.json" "$ROOT/hrs1/off_rescore.json"
rescore on_hrs0  "$ROOT/hrs0/on_src.json"  "$ROOT/hrs0/on_rescore.json"
rescore off_hrs0 "$ROOT/hrs0/off_src.json" "$ROOT/hrs0/off_rescore.json"

# --- analyze: clear-margin flip rate + Wilson-95 per arm; STEP 0 + STEP 1 verdicts ---
python3 - "$ROOT" <<'PY' | tee -a "$ROOT/RESULTS.txt"
import json, math, sys
root = sys.argv[1]
def load(p):
    try: d = json.load(open(p))
    except Exception as e: return None, f"load-fail {e}"
    # find clear-margin flips + positions across likely key names
    flips = d.get("total_clear_margin_flips", d.get("clear_margin_flips"))
    pos   = d.get("total_positions", d.get("positions", d.get("total_clear_margin_positions")))
    if flips is None or pos is None:
        # search nested
        for k,v in d.items():
            if isinstance(v,dict):
                flips = flips if flips is not None else v.get("total_clear_margin_flips") or v.get("clear_margin_flips")
                pos   = pos   if pos   is not None else v.get("total_positions") or v.get("positions")
    return (flips, pos), (None if (flips is not None and pos) else f"keys={list(d.keys())[:8]}")
def wilson(k, n):
    if not n: return (0,0,0)
    p=k/n; z=1.96; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (100*p, 100*max(0,c-h), 100*min(1,c+h))
arms={}
for a in ["on_hrs1","off_hrs1","on_hrs0","off_hrs0"]:
    sub = "hrs1" if a.endswith("hrs1") else "hrs0"
    (kp), err = load(f"{root}/{sub}/{a.split('_')[0]}_rescore.json")
    if isinstance(kp,tuple) and kp[0] is not None and kp[1]:
        k,n=kp; lo=wilson(k,n)
        arms[a]=(k,n,lo)
        print(f"  {a}: flips={k}/{n} = {lo[0]:.2f}% Wilson95[{lo[1]:.2f},{lo[2]:.2f}]")
    else:
        print(f"  {a}: UNAVAILABLE ({err})")
print("=== VERDICTS ===")
if "on_hrs1" in arms and "off_hrs1" in arms:
    on=arms["on_hrs1"][2]; off=arms["off_hrs1"][2]
    print(f"  STEP0 (residual on baked stack, HRS=1): cache-ON {on[0]:.2f}% [up={on[2]:.2f}] vs cache-OFF {off[0]:.2f}% [up={off[2]:.2f}] + 12.90% floor")
    print(f"    -> WITHIN-FLOOR: { 'YES (cache-ON upper <= cache-OFF point AND <=12.90)' if on[2]<=max(off[0],12.90) else 'NO -> real residual, do STEP1' }")
if "on_hrs1" in arms and "on_hrs0" in arms:
    h1=arms["on_hrs1"][2][0]; h0=arms["on_hrs0"][2][0]
    print(f"  STEP1 (HRS help/hurt): HRS=1 {h1:.2f}% vs HRS=0 {h0:.2f}%  -> { 'HRS=1 is a NET CARRIER, un-bake (use chunked kernel)' if h0 < h1-0.5 else ('HRS=1 helps, keep' if h1 < h0-0.5 else 'HRS neutral') }")
PY
echo "=== remeasure done -> $ROOT/RESULTS.txt ==="

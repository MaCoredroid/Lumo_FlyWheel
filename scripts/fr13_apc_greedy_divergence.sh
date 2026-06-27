#!/usr/bin/env bash
# GREEDY (temp=0) same-boot cache-ON vs cache-OFF first-divergence — the clean lossless test.
# The temp-0.6 clear-margin gate (fr13_apc_remeasure_ab.sh) was CONFOUNDED: the recurrent-oracle
# rescore truncates at each stream's stochastic stop-token, giving incomparable n (2000/67/20/158)
# and the cache-OFF reference disagreed with itself (8.96% vs 15.19% across boots). Greedy removes
# the stochastic ramble: if the APC cache restore is LOSSLESS, greedy cache-ON == greedy cache-OFF
# token-for-token (same argmax path); if not, they FIRST DIVERGE at position X = the carrier acts at X.
# NON-DESTRUCTIVE: pre-sets FR13_APC_HIT_RECURRENT_SUFFIX via env per arm only.
#   Arms: HRS=1 (baked default) and HRS=0. Each captures cache-ON + cache-OFF GREEDY same-boot.
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
export TEMP=0.0 IGNORE_EOS=0 MAX_TOKENS=${MAX_TOKENS:-512} SEED=1313 TOP_K=1
TS=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=output/fr13_apc_greedy_div/run_$TS
mkdir -p "$ROOT"
echo "$ROOT" > /tmp/claude-1000/-home-mark-shared/46f03809-5059-4e30-936d-1adda7f44337/scratchpad/greedydiv_root.txt 2>/dev/null || true
PRECHECK_ROOT=output/fr13_apc_temp06
echo "GREEDY DIVERGENCE: same-boot cache-ON vs cache-OFF temp=0, first-divergence position. arms HRS=1/HRS=0 -> $ROOT" | tee "$ROOT/RESULTS.txt"

capture_arm() {  # $1=HRS
  local HRS=$1 A="hrs$1"; local AR="$ROOT/$A"; mkdir -p "$AR"
  echo "=== [greedy $A] cache-ON(HRS=$HRS)+cache-OFF temp=0 same-boot $(date -u +%H:%M:%SZ) ===" | tee -a "$ROOT/RESULTS.txt"
  FR13_APC_HIT_RECURRENT_SUFFIX=$HRS bash scripts/fr13_apc_temp06_precheck.sh > "$AR/precheck.log" 2>&1 || true
  cp "$PRECHECK_ROOT/cat9_apc_on_src.json"  "$AR/on_src.json"  2>/dev/null || echo "MISS on $A" | tee -a "$ROOT/RESULTS.txt"
  cp "$PRECHECK_ROOT/cat9_apc_off_src.json" "$AR/off_src.json" 2>/dev/null || echo "MISS off $A" | tee -a "$ROOT/RESULTS.txt"
  # engagement: HRS marker from the precheck's run dir
  local mk; mk=$(cat $(ls -td "$PRECHECK_ROOT"/run_*/logs/fr13_apc_bridge_loaded.flag 2>/dev/null | head -1) 2>/dev/null)
  echo "  marker[$mk]  cache_fired=$(grep -cE 'non-vacuity. OK: cache fired' "$AR/precheck.log" 2>/dev/null)" | tee -a "$ROOT/RESULTS.txt"
}

capture_arm 1
capture_arm 0

python3 - "$ROOT" <<'PY' | tee -a "$ROOT/RESULTS.txt"
import json, sys
root=sys.argv[1]
def ids(p):
    try: return json.load(open(p))["records"][0]["served_token_ids"]
    except Exception as e: return None
print("=== GREEDY FIRST-DIVERGENCE (cache-ON vs cache-OFF, same boot) ===")
for hrs in (1,0):
    on=ids(f"{root}/hrs{hrs}/on_src.json"); off=ids(f"{root}/hrs{hrs}/off_src.json")
    if on is None or off is None: print(f"  HRS={hrs}: MISSING streams"); continue
    n=min(len(on),len(off)); div=next((i for i in range(n) if on[i]!=off[i]), None)
    if div is None and len(on)==len(off):
        print(f"  HRS={hrs}: IDENTICAL ({len(on)} tokens) -> cache restore LOSSLESS (greedy)")
    elif div is None:
        print(f"  HRS={hrs}: identical up to min-len {n} but lengths differ (on={len(on)} off={len(off)}) -> diverge at stop")
    else:
        print(f"  HRS={hrs}: FIRST DIVERGENCE at pos {div}/{n}  on[{div}]={on[div]} off[{div}]={off[div]}  (on_len={len(on)} off_len={len(off)})")
        print(f"           on[{max(0,div-2)}:{div+3}]={on[max(0,div-2):div+3]}  off[..]={off[max(0,div-2):div+3]}")
PY
echo "=== greedy divergence done -> $ROOT/RESULTS.txt ==="

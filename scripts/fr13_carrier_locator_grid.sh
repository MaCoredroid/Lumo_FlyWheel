#!/bin/bash
# fr13_carrier_locator_grid.sh — FAST generation-level carrier localization across THREE candidate carriers.
#
# THREE potential carriers (user, 2026-07-02) — the forked launcher copies the forked FA2 .so over stock
# at L590 REGARDLESS of ATTENTION_BACKEND, so "forked FA2 .so" is a distinct axis from "TREE_ATTN kernel":
#   (1) TREE_ATTN tree kernel      (2) forked FA2 .so (vs stock flash-attn)     (3) APC/EXACT_SEED cache x tree
#
# Boots each cell SERIALLY (variant PROBE_ONLY=1), replays the banked tool-call-boundary prompts N times
# DIRECTLY against local vLLM at temp 0.6, scores malformed-markup rate. No agent/nudge/tunnel/eval.
#
# CELLS (depth=5 held except B/C; SAME prompt corpus + temp 0.6 everywhere):
#   N  = nativemtp5         native  FLASH  STOCK-.so   cache off  ns5   TRUE clean baseline
#   A0 = flash_ns5_nocache  forked  FLASH  FORKED-.so  cache off  ns5   forked FA2 .so, no cache
#   A  = nativemtp5_exseed  forked  FLASH  FORKED-.so  cache ON   ns5   forked FA2 .so + cache
#   D0 = chain5             forked  TREE   FORKED-.so  cache off  ns5   tree kernel, no cache
#   D  = chain5 + APC env   forked  TREE   FORKED-.so  cache ON   ns5   tree kernel + cache
#   B  = cat8  + APC env    forked  TREE   FORKED-.so  cache ON   ns8   full forked (branching)
#   C  = flash_ns8_exseed   forked  FLASH  FORKED-.so  cache ON   ns8   depth control (optional)
#
# THREE DECISIVE ONE-VARIABLE READS:
#   forked-FA2 carrier :  A0 - N   (forked vs stock .so, both FLASH/cache-off/ns5)
#   tree-kernel carrier:  D0 - A0  (TREE vs FLASH, both forked-.so/cache-off/ns5)
#   cache-on-tree carr.:  D  - D0  (cache on vs off, both TREE/ns5)   control off-tree: A - A0 (FLASH)
#
# SERIAL only. Run ONLY when the DGX is free (no parallel testing / no side load during a measurement).
set -uo pipefail
REPO=/home/mark/shared/lumoFlyWheel; cd "$REPO"
source "$REPO/.lumo.local.env" 2>/dev/null || true   # LUMO_SUDO_PASSWORD for the container restart machinery

CELLS=${CELLS:-"B B0 D0 D A0 N A"}              # B/B0 = cat8 cache on/off (branching-tree penalty); N A0 D0 D = 3-carrier
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
RUNROOT=${RUNROOT:-output/fr13_carrier_locator/run_$STAMP}
SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_b4_sixteen.json}   # parsed by variant; PROBE_ONLY ignores tasks
PROBE_PROMPTS=${PROBE_PROMPTS:-$REPO/output/fr13_tree_cache_matrix/run_20260702T092119Z/m_cat8on/proxy_request_dumps}
PROBE_SAMPLES=${PROBE_SAMPLES:-16}
PROBE_PROMPT_LIMIT=${PROBE_PROMPT_LIMIT:-16}
mkdir -p "$RUNROOT"

declare -A KINDOF=( [N]=nativemtp5 [A0]=flash_ns5_nocache [A]=nativemtp5_exseed
                    [D0]=chain5 [D]=chain5 [B]=cat8 [B0]=cat8 [C]=flash_ns8_exseed )
# cache ON via env for the tree/branching cache cells (empty XFLAGS); A/C bake it in their KIND; N/A0/D0/B0 off.
declare -A APCENV=( [N]="" [A0]="" [A]="" [D0]="" [B0]=""
                    [D]="FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1"
                    [B]="FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1" [C]="" )

echo "=== CARRIER LOCATOR GRID $STAMP cells=[$CELLS] prompts=$PROBE_PROMPTS samples=$PROBE_SAMPLES limit=$PROBE_PROMPT_LIMIT ==="
echo "    corpus: $(ls "$PROBE_PROMPTS"/chatreq_*.json 2>/dev/null | wc -l) banked boundary prompts"
for c in $CELLS; do
  KIND=${KINDOF[$c]:?unknown cell $c}; ARM="cell_${c}_${KIND}"
  echo "--- CELL $c KIND=$KIND cache=[${APCENV[$c]:-off}] @ $(date -u +%H:%M:%S) ---"
  if [[ -n "$(docker ps -q)" ]]; then echo "FAIL: docker not empty before cell $c"; docker ps; exit 2; fi
  env RUNROOT="$RUNROOT" PROBE_ONLY=1 \
      PROBE_PROMPTS="$PROBE_PROMPTS" PROBE_SAMPLES="$PROBE_SAMPLES" PROBE_PROMPT_LIMIT="$PROBE_PROMPT_LIMIT" \
      ${APCENV[$c]} \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" "$KIND" "$SUBSET" \
      > "$RUNROOT/${ARM}.log" 2>&1
  RC=$?
  echo "    cell $c rc=$RC -> $RUNROOT/$ARM/carrier_probe.json"
  if (( RC != 0 )); then echo "    WARN cell $c failed rc=$RC; tail:"; tail -25 "$RUNROOT/${ARM}.log"; fi
  if [[ -n "$(docker ps -q --filter name=fr13-bigdenom-$ARM)" ]]; then
    echo "    WARN: container survived; forcing rm"; docker rm -f "fr13-bigdenom-$ARM" >/dev/null 2>&1 || true
  fi
done

echo "=== GRID DONE — per-cell malformed-markup rate + the three carrier reads ==="
.venv/bin/python - "$RUNROOT" $CELLS <<'PY'
import json, os, sys
runroot = sys.argv[1]; cells = sys.argv[2:]
kindof = {"N":"nativemtp5","A0":"flash_ns5_nocache","A":"nativemtp5_exseed",
          "D0":"chain5","D":"chain5","B":"cat8","B0":"cat8","C":"flash_ns8_exseed"}
rows = {}
for c in cells:
    f = os.path.join(runroot, f"cell_{c}_{kindof[c]}", "carrier_probe.json")
    if not os.path.isfile(f):
        print(f"cell {c}: NO RESULT"); continue
    s = json.load(open(f))["summary"]; rows[c] = s
    sp = s.get("spec") or {}
    print(f"cell {c:<2}({kindof[c]:<18}) malformed_rate={s['malformed_rate']}  "
          f"clean_toolcall_rate={s['clean_toolcall_rate']}  n_ok={s['n_ok']}  labels={s['labels']}")
    print(f"      spec(same-prompt): accepted/step={sp.get('accepted_per_step')} accept_rate={sp.get('accept_rate')} "
          f"draft_width={sp.get('draft_width')}")
def r(c): return rows.get(c, {}).get("malformed_rate")
def read(name, hi, lo, verdict_hi):
    if hi in rows and lo in rows:
        d = (r(hi) or 0) - (r(lo) or 0)
        print(f"  {name:<22}: {hi}={r(hi)} vs {lo}={r(lo)}  delta={d:+.4f}  -> "
              f"{verdict_hi if d > 0 else 'NOT a carrier here'}")
print("\nTHREE CARRIER READS (one variable each):")
read("forked-FA2 .so", "A0", "N",  "forked FA2 .so IS a carrier")
read("TREE_ATTN kernel", "D0", "A0", "TREE_ATTN kernel IS a carrier")
read("cache-on-chain(ns5)", "D",  "D0", "APC/EXACT_SEED x chain-tree IS a carrier")
read("cache-on-BRANCH(cat8)", "B", "B0", "APC/EXACT_SEED x BRANCHING cat8 tree IS a carrier (spine-5 proof does NOT cover it)")
read("  (cache off-tree ctrl)", "A", "A0", "cache hurts off-tree too (not tree-specific)")
read("branching vs chain", "B0", "D0", "cat8 branching (cache off) worse than chain5 (cache off)")
PY
echo "CARRIER LOCATOR GRID COMPLETE run=$RUNROOT"

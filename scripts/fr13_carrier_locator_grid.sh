#!/bin/bash
# fr13_carrier_locator_grid.sh — FAST generation-level carrier localization ("tree or FA2?").
#
# Boots each kernel cell SERIALLY (via fr13_bigdenom_swe_serve_variant.sh PROBE_ONLY=1), replays the
# banked tool-call-boundary prompts N times DIRECTLY against local vLLM at temp 0.6, and measures the
# malformed-markup emission rate per cell. No agent, no nudge net, no offload tunnel, no eval => breaks
# all four confounds (kernel/num_spec/topology by the grid; nudge+trajectory+infra by single-turn direct).
#
# CELLS (all EXACT_SEED cache ON, temp 0.6, SAME prompt corpus):
#   A = nativemtp5_exseed  FLASH_ATTN naive  ns5   (native baseline — expect clean)
#   C = flash_ns8_exseed   FLASH_ATTN naive  ns8   (isolate draft DEPTH under clean kernel)
#   D = chain5             TREE_ATTN  tree   ns5    (isolate the TREE_ATTN tree apparatus at native depth)
#   B = cat8               TREE_ATTN  tree   ns8    (full forked — carrier confirmed agentically)
#
# DECISIVE READ: D vs A (same depth, tree apparatus on/off) answers "tree or FA2":
#   D corrupts & A clean            -> the TREE_ATTN tree apparatus is the carrier (answer: "tree")
#   D clean, C corrupts             -> draft depth/num_spec is the driver (answer: "not the tree kernel")
#   D clean, C clean, only B corrupts-> the cat8 BRANCHING specifically (rank-2 siblings), not the kernel
#
# SERIAL only. Run ONLY when the DGX is free (no parallel testing / no side load during a measurement).
set -uo pipefail
REPO=/home/mark/shared/lumoFlyWheel; cd "$REPO"
# LUMO_SUDO_PASSWORD for the container restart machinery inside the variant script.
source "$REPO/.lumo.local.env" 2>/dev/null || true

CELLS=${CELLS:-"A D C B"}                       # "A D" alone gives the kernel answer in ~1 boot-pair
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
RUNROOT=${RUNROOT:-output/fr13_carrier_locator/run_$STAMP}
SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_b4_sixteen.json}   # parsed by variant; PROBE_ONLY ignores tasks
PROBE_PROMPTS=${PROBE_PROMPTS:-$REPO/output/fr13_tree_cache_matrix/run_20260702T092119Z/m_cat8on/proxy_request_dumps}
PROBE_SAMPLES=${PROBE_SAMPLES:-16}
PROBE_PROMPT_LIMIT=${PROBE_PROMPT_LIMIT:-16}
mkdir -p "$RUNROOT"

declare -A KINDOF=( [A]=nativemtp5_exseed [C]=flash_ns8_exseed [D]=chain5 [B]=cat8 )
# A and C carry EXACT_SEED cache in their KIND XFLAGS; D and B (empty XFLAGS) need it via env to
# hold cache ON across all cells (matches the banked cat8on / nat16 deployment config).
declare -A APCENV=( [A]="" [C]="" [D]="FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1" [B]="FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1" )

echo "=== CARRIER LOCATOR GRID $STAMP cells=[$CELLS] prompts=$PROBE_PROMPTS samples=$PROBE_SAMPLES limit=$PROBE_PROMPT_LIMIT ==="
echo "    corpus size: $(ls "$PROBE_PROMPTS"/chatreq_*.json 2>/dev/null | wc -l) banked boundary prompts"
for c in $CELLS; do
  KIND=${KINDOF[$c]:?unknown cell $c}; ARM="cell_${c}_${KIND}"
  echo "--- CELL $c KIND=$KIND @ $(date -u +%H:%M:%S) ---"
  # pre-cell guard: DGX must be free (variant does its own hygiene, but fail fast here too)
  if [[ -n "$(docker ps -q)" ]]; then echo "FAIL: docker not empty before cell $c"; docker ps; exit 2; fi
  env RUNROOT="$RUNROOT" PROBE_ONLY=1 \
      PROBE_PROMPTS="$PROBE_PROMPTS" PROBE_SAMPLES="$PROBE_SAMPLES" PROBE_PROMPT_LIMIT="$PROBE_PROMPT_LIMIT" \
      ${APCENV[$c]} \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" "$KIND" "$SUBSET" \
      > "$RUNROOT/${ARM}.log" 2>&1
  RC=$?
  echo "    cell $c rc=$RC -> $RUNROOT/$ARM/carrier_probe.json"
  if (( RC != 0 )); then echo "    WARN cell $c failed rc=$RC; tail:"; tail -25 "$RUNROOT/${ARM}.log"; fi
  # belt-and-suspenders teardown (variant trap handles it; verify no container leaks into next cell)
  if [[ -n "$(docker ps -q --filter name=fr13-bigdenom-$ARM)" ]]; then
    echo "    WARN: container survived; forcing rm"; docker rm -f "fr13-bigdenom-$ARM" >/dev/null 2>&1 || true
  fi
done

echo "=== GRID DONE — per-cell malformed-markup rate ==="
.venv/bin/python - "$RUNROOT" $CELLS <<'PY'
import json, os, sys
runroot = sys.argv[1]; cells = sys.argv[2:]
kindof = {"A":"nativemtp5_exseed","C":"flash_ns8_exseed","D":"chain5","B":"cat8"}
rows = {}
for c in cells:
    f = os.path.join(runroot, f"cell_{c}_{kindof[c]}", "carrier_probe.json")
    if not os.path.isfile(f):
        print(f"cell {c}: NO RESULT"); continue
    s = json.load(open(f))["summary"]; rows[c] = s
    print(f"cell {c:<2}({kindof[c]:<18}) malformed_rate={s['malformed_rate']}  "
          f"clean_toolcall_rate={s['clean_toolcall_rate']}  n_ok={s['n_ok']}  labels={s['labels']}")
# decisive reads
def rate(c): return rows.get(c,{}).get("malformed_rate")
if "A" in rows and "D" in rows:
    print(f"\nKERNEL READ (D vs A, same depth): D(TREE ns5)={rate('D')}  A(FLASH ns5)={rate('A')} "
          f"-> {'TREE apparatus IS carrier' if (rate('D') or 0)>(rate('A') or 0) else 'kernel NOT the carrier'}")
if "A" in rows and "C" in rows:
    print(f"DEPTH READ  (C vs A, same kernel): C(FLASH ns8)={rate('C')}  A(FLASH ns5)={rate('A')} "
          f"-> {'depth contributes' if (rate('C') or 0)>(rate('A') or 0) else 'depth NOT the driver'}")
PY
echo "CARRIER LOCATOR GRID COMPLETE run=$RUNROOT"

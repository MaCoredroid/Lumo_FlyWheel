#!/bin/bash
# fr13_g1_kernel_confirm.sh — G1 KERNEL-CONFIRM gate (the cheapest decisive carrier/fix gate).
#
# Captures the tree GDN kernel's per-node gdn_scan_out on cat8 under {default(body), recompute}, then
# offline-diffs each vs the NATIVE per-path FLA replay (fr12_branch_path_oracle_probe.py), split spine vs
# branch nodes. Deterministic (seed-fixed), NO temp 0.6, NO SWE agent.
#   PASS: default(body) scan max_abs > ~1e-3 on spine AND branch nodes (=> _gdn_node_step diverges from native
#         = the carrier locus) AND recompute scan max_abs == 0.0 bit-exact on ALL nodes (=> recompute is the
#         fix). That combination also EXONERATES FA2/TREE_ATTN by elimination (byte-identical across arms).
# Requires the un-baked FR13_REPLAY_ROUTE (=0 keeps the per-node tree_state scratch alive for the capture).
# Cache-independent: G1 tests the KERNEL arithmetic vs native, so it boots cat8 WITHOUT APC.
set -uo pipefail
REPO=/home/mark/shared/lumoFlyWheel; cd "$REPO"
source "$REPO/.lumo.local.env" 2>/dev/null || true
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
RUNROOT=${RUNROOT:-output/fr13_g1_kernel_confirm/run_$STAMP}
SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_carrier_four.json}   # parsed by variant; CAPTURE_ONLY exits before use
MODEL_DIR=${MODEL_DIR:-/models/qwen3.6-27b-fp8}
mkdir -p "$RUNROOT"
echo "=== G1 KERNEL-CONFIRM $STAMP  (cat8 tree; capture body + recompute; diff vs native FLA) ==="
[[ -d "$MODEL_DIR" ]] || echo "WARN: model dir $MODEL_DIR not found on host — probe o_proj/norm may fail (set MODEL_DIR)"

for MODE in body recompute; do
  ARM="cap_$MODE"; SA=$([ "$MODE" = recompute ] && echo 1 || echo 0)
  echo "--- CAPTURE $MODE (FR13_SCAN_ALIGN=$SA) @ $(date -u +%H:%M:%S) ---"
  if [[ -n "$(docker ps -q)" ]]; then echo "FAIL: docker not empty before $MODE"; docker ps; exit 2; fi
  env RUNROOT="$RUNROOT" CAPTURE_ONLY=1 \
      FR13_REPLAY_ROUTE=0 FR13_SCAN_ALIGN="$SA" FR13_SCAN_ALIGN_MODE="$MODE" \
      FR10_TREE_GDN_CAPTURE_PAYLOAD=/logs/g1_payload.pt \
      FR12_SUBKERNEL_CAPTURE=/logs/g1_subkernel.pt \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" cat8 "$SUBSET" \
      > "$RUNROOT/${ARM}.log" 2>&1
  RC=$?
  echo "  $MODE capture rc=$RC ; .pt in $RUNROOT/$ARM/logs:"
  ls "$RUNROOT/$ARM/logs"/*.pt 2>/dev/null | sed 's|.*/logs/|    |' || echo "    (none)"
  if (( RC != 0 )); then echo "  CAPTURE FAILED — tail:"; tail -25 "$RUNROOT/${ARM}.log"; fi
  # belt-and-suspenders teardown
  [[ -n "$(docker ps -q)" ]] && { docker rm -f "fr13-bigdenom-$ARM" >/dev/null 2>&1 || true; }
done

echo "=== PROBE: default-vs-native, recompute-vs-native (offline, boot-free) ==="
for MODE in body recompute; do
  L="$RUNROOT/cap_$MODE/logs"
  PAY=$(ls "$L"/*payload*.pt 2>/dev/null | head -1); SUB=$(ls "$L"/*subkernel*.pt 2>/dev/null | head -1)
  if [[ -z "$PAY" || -z "$SUB" ]]; then echo "  $MODE: MISSING capture (payload=$PAY subkernel=$SUB) — skip"; continue; fi
  .venv/bin/python scripts/fr12_branch_path_oracle_probe.py \
    --payload "$PAY" --tree-subkernel "$SUB" --model-dir "$MODEL_DIR" --layer 0 \
    --out "$RUNROOT/probe_$MODE.json" > "$RUNROOT/probe_$MODE.log" 2>&1
  echo "  $MODE probe rc=$? -> $RUNROOT/probe_$MODE.json"
  tail -3 "$RUNROOT/probe_$MODE.log" 2>/dev/null | sed 's/^/    /'
done

echo "=== G1 VERDICT ==="
.venv/bin/python - "$RUNROOT" <<'PY'
import json, os, sys
rr = sys.argv[1]
def load(m):
    f = os.path.join(rr, f"probe_{m}.json")
    return json.load(open(f)) if os.path.isfile(f) else None
for m in ("body", "recompute"):
    d = load(m)
    if d is None: print(f"  {m}: NO PROBE OUTPUT"); continue
    print(f"  {m}: {json.dumps(d)[:600]}")
print("\nREAD: PASS if body(default) scan max_abs > 1e-3 on spine AND branch (=> _gdn_node_step is the carrier)")
print("      AND recompute scan max_abs == 0.0 bit-exact (=> recompute is the fix; FA2/TREE_ATTN exonerated).")
PY
echo "G1 KERNEL-CONFIRM DONE run=$RUNROOT"

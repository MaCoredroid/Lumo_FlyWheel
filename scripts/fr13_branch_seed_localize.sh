#!/bin/bash
# fr13_branch_seed_localize.sh — pin the BRANCH-node seed on the CURRENT served build.
#
# The garble is forward-drift born at layer-0 GDN on a bit-identical INPUT (recurrent seed),
# amplified x~14800 over 64 layers. Spine is clean (chain5 kills garble + spine conv fixed 3a9039cc);
# the seed lives on BRANCH nodes. This captures the served cat8 tree GDN per-node stages, then
# diffs each node vs the NATIVE per-path FLA replay (fr12_branch_path_oracle_probe.py = the
# SpecInfer/STree path-rerun oracle = the RIGHT reference for a branch), split spine vs branch.
#
#   READ: if BRANCH nodes drift (scan/gate/o_proj max_abs > 0) while SPINE nodes are ~0.0
#         => the seed is a branch-path forward defect (untested vs the path-rerun oracle),
#         and it is compute-only (which state a branch reads/replays), NOT the scan arithmetic
#         (proven bit-exact both geometries) NOR inter-request batching (native+realB8=0%).
#
# Body-only (scan/recompute modes DELETED this session). Served flags (replay + stateless col-0,
# all baked default-on). Deterministic capture; NO temp-0.6, NO SWE agent. Fail-loud on no-capture.
set -uo pipefail
REPO=/home/mark/shared/lumoFlyWheel; cd "$REPO"
source "$REPO/.lumo.local.env" 2>/dev/null || true
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
RUNROOT=${RUNROOT:-output/fr13_branch_seed_loc/run_$STAMP}
SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_carrier_four.json}
MODEL_DIR=${MODEL_DIR:-/models/qwen3.6-27b-fp8}
ARM=cap_body
mkdir -p "$RUNROOT"
echo "=== BRANCH-SEED LOCALIZE $STAMP (served cat8; capture body; diff vs native per-path FLA) ==="
[[ -d "$MODEL_DIR" ]] || echo "WARN: model dir $MODEL_DIR not found on host — o_proj/norm probe may fail"
if [[ -n "$(docker ps -q)" ]]; then echo "FAIL: docker not empty before capture"; docker ps; exit 2; fi

echo "--- CAPTURE body (served replay path) @ $(date -u +%H:%M:%S) ---"
env RUNROOT="$RUNROOT" CAPTURE_ONLY=1 \
    FR13_REPLAY_ROUTE=1 \
    FR12_SUBKERNEL_CAPTURE_LAYER_PREFIX=0 \
    FR10_TREE_GDN_CAPTURE_PAYLOAD=/logs/bseed_payload.pt \
    FR12_SUBKERNEL_CAPTURE=/logs/bseed_subkernel.pt \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" cat8 "$SUBSET" \
    > "$RUNROOT/${ARM}.log" 2>&1
RC=$?
echo "  capture rc=$RC ; .pt in $RUNROOT/$ARM/logs:"
ls "$RUNROOT/$ARM/logs"/*.pt 2>/dev/null | sed 's|.*/logs/|    |' || echo "    (NONE — capture did not fire)"
[[ -n "$(docker ps -q)" ]] && { docker rm -f "fr13-bigdenom-$ARM" >/dev/null 2>&1 || true; }

PAY=$(ls "$RUNROOT/$ARM/logs"/*payload*.pt 2>/dev/null | head -1)
SUB=$(ls "$RUNROOT/$ARM/logs"/*subkernel*.pt 2>/dev/null | head -1)
if [[ -z "$PAY" || -z "$SUB" ]]; then
  echo "FAIL: missing capture (payload=$PAY subkernel=$SUB) — capture hook did not fire on served path. tail:"
  tail -30 "$RUNROOT/${ARM}.log"; exit 3
fi

echo "=== PROBE: served-vs-native per-path FLA (boot-free), spine vs branch ==="
.venv/bin/python scripts/fr12_branch_path_oracle_probe.py \
  --payload "$PAY" --tree-subkernel "$SUB" --model-dir "$MODEL_DIR" --layer 0 \
  --branch-nodes 3,5,7,9 --spine-nodes 0,1,2,4,6 \
  --out "$RUNROOT/probe_body.json" > "$RUNROOT/probe_body.log" 2>&1
echo "  probe rc=$? -> $RUNROOT/probe_body.json"
tail -5 "$RUNROOT/probe_body.log" 2>/dev/null | sed 's/^/    /'
echo "=== VERDICT ==="; cat "$RUNROOT/probe_body.json" 2>/dev/null | head -c 1400
echo; echo "BRANCH-SEED LOCALIZE DONE run=$RUNROOT"

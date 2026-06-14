#!/usr/bin/env bash
# FR13 LOCKED LAUNCHER (2026-06-13) — boots the EXACT B=1 SWE-Verified gold-gate cat9 pipeline
# (== main HEAD default-ON serving path == b7887c89). Every flag pinned explicitly so a run is
# NOT env-default-dependent. Diagnostics are forced OFF unless armed via `--arm FR13_<FLAG>`.
# See FR13_PIPELINE_LOCK.md. Native baseline = scripts/fr10_launch_speed_server.sh num_spec=5 FLASH.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- diagnostics: OFF by default; `--arm FR13_COMMIT_ARGMAX_GATE` etc. flips one to 1 ---
declare -A ARMED=()
while [[ "${1:-}" == "--arm" ]]; do ARMED["$2"]=1; shift 2; done
diag() { local f="$1"; if [[ "${ARMED[$f]:-0}" == "1" ]]; then echo 1; else echo 0; fi; }

# --- LOCKED cat9 tree (9-node: 5-spine + top-2 leaf on depths 1-4; NO root leaf) ---
export TREE="[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1), (0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 0, 1)]"
export FR10_DECODE_MODE_DEFAULT=tree_mtp

# --- PIPELINE-ON (the gold-gate serving path) ---
export FR13_DRAFTER_SINGLE_LOGITS=1   # FIX-1
export FR13_EAGER_PACK=1              # FIX-2  (replay-coupled)
export FR13_TREE_CONV_FUSED=1         # FIX-3  (requires REPLAY_ROUTE=1)
export FR13_TREE_SAMPLE_ROW=1         # FIX-A
export FR13_REPLAY_ROUTE=1            # replay route (ALWAYS ON)
export FR13_FA2_TREE_BIAS=1
export FR13_FA2_PREFILL_NATIVE=1
export FR13_TREE_ATTN_EXP2_SOFTMAX=1
export FR13_CONV_COMMITTED_PATH=1
export BATCH_INVARIANT=0
# --- BAKED FIX (2026-06-14): in_proj_ba pad-to-fixed-M batch-invariance (FR13_WIDTH_
# CARRIER_INPROJ_BA_BIND.md, H1). Pads bf16 in_proj_ba (+ out_proj) to a tree_n-
# independent M => M-invariant a/b => removes ~8 of the +17 leaf co-residency flips.
# Same-boot OFF=26 vs ON=18 (-8, lossless: det [T,T,T,T], CPU max_abs=0.0, flips DOWN).
# Spec-path-only (gate num_spec_decodes>1); regular decode byte-unaffected.
export LUMO_FB_KERNEL_ROWS=1
export LUMO_FB_PROJ_PAD_ROWS=16

# --- DIAGNOSTIC-OFF (armable for the chase) ---
export FR13_COMMIT_ARGMAX_GATE="$(diag FR13_COMMIT_ARGMAX_GATE)"
export FR13_FORCE_SPINE_COMMIT="$(diag FR13_FORCE_SPINE_COMMIT)"
export FR13_FIX1_SELFCHECK="$(diag FR13_FIX1_SELFCHECK)"
export FR13_CHASE_DIAG="$(diag FR13_CHASE_DIAG)"
export FR13_BI_TREE_ATTN="$(diag FR13_BI_TREE_ATTN)"
export FR10_METRICS="$(diag FR10_METRICS)"

echo "[locked] cat9 num_spec=9 TREE_ATTN | pipeline ON | diagnostics armed: ${!ARMED[*]:-none}"
exec "$HERE/fr13_launch_forked_fa2_tree_server.sh" "$@"

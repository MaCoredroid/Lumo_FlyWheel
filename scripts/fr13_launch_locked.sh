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
export FR13_EAGER_PACK=${FR13_EAGER_PACK:-1}   # FIX-2 (replay-coupled); env-overridable for committer-native test
export FR13_TREE_CONV_FUSED=1         # FIX-3  (requires REPLAY_ROUTE=1)
export FR13_TREE_SAMPLE_ROW=1         # FIX-A
export FR13_REPLAY_ROUTE=1            # replay route (ALWAYS ON)
export FR13_APC_COMMIT_TO_RUNNING_ROW=1  # stateless col-0 lifecycle (write leaf->col0)
export FR13_TREE_RUNROW_INIT=1           #   ... read col-0
export FR13_APC_BURN_NODE_BANK=1         #   ... burn node cols (all 3 = one lifecycle, now baked default-on)
export FR13_FA2_TREE_BIAS=1
export FR13_FA2_PREFILL_NATIVE=1
export FR13_TREE_ATTN_EXP2_SOFTMAX=1
export FR13_CONV_COMMITTED_PATH=1
# --- BAKED FIX (2026-07-13): attention-KV re-linearization = THE garble fix. GDN/conv
# state gets re-linearized post-accept (launch_tree_state_linear_remap) but attn KV
# never did => accepted non-contiguous BRANCH path read a sibling near-neighbor's foreign
# KV => garble. Copies each committed node's K/V flat verify slot -> linear committed slot
# per full-attn layer (launch_attn_kv_linear_remap, in sample_tokens = graph-safe).
# GATE: temp-0.6 matrix_build 15/15 -> 0/15 (eager+graph+cache-cold), engaged foreign>0,
# clean; live SWE astropy-12907 RESOLVED (cache hits 68-74%); speed tax -0.7% s/fwd (noise).
export FR13_ATTN_KV_REMAP=1
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
# FR13_FORK_MARGIN_DUMP: READ-ONLY committer-fork classifier (default OFF).
# Armable via `--arm FR13_FORK_MARGIN_DUMP`; ALSO honors a pre-set env (the
# K1 BootCapture harness exports it directly, like FR13_SCAN_ALIGN). Default
# OFF = byte-identical locked serving path.
export FR13_FORK_MARGIN_DUMP="${FR13_FORK_MARGIN_DUMP:-$(diag FR13_FORK_MARGIN_DUMP)}"
export FR13_FORCE_SPINE_COMMIT="$(diag FR13_FORCE_SPINE_COMMIT)"
export FR13_FIX1_SELFCHECK="$(diag FR13_FIX1_SELFCHECK)"
export FR13_CHASE_DIAG="$(diag FR13_CHASE_DIAG)"
export FR13_BI_TREE_ATTN="$(diag FR13_BI_TREE_ATTN)"
export FR10_METRICS="$(diag FR10_METRICS)"

_armed_list="${!ARMED[*]}"   # assign first: `${!ARMED[*]:-none}` mis-parses with >1 key ("invalid variable name")
echo "[locked] cat9 num_spec=9 TREE_ATTN | pipeline ON | diagnostics armed: ${_armed_list:-none}"
exec "$HERE/fr13_launch_forked_fa2_tree_server.sh" "$@"

#!/usr/bin/env bash
# ============================================================================
# FR13 CANONICAL ENV REGISTRY — the ONE place baked flags and era-fix
# protections live. Created 2026-07-26 after the second lost-protection
# regression (LUMO_PROXY_SSE_HEARTBEAT_S=15 lived only in era seq files;
# helper defaulted 0; five emit-wedges across four arms before it was
# re-found — same class as the ATTN_KV_REMAP near-loss that created
# fr13_required_tree_flags.sh).
#
# RULES
#  - Every env here is BAKED (canonical value) or a PROTECTION (era fix).
#    Per-arm seq files may contain ONLY experiment deltas — a seq file that
#    exports a var listed here is overriding a baked value and must say why.
#  - Sourced by: fr13_b4_campaign_driver.sh (all campaign arms) and
#    fr13_launch_locked.sh (golden pipeline). New drivers MUST source it.
#  - Tree-side flags additionally have engagement ASSERTION via
#    fr13_required_tree_flags.sh (kept as the assertion layer; values there
#    must match here).
#  - When an era fix lands: add it HERE in the same commit, with the origin
#    commit hash. Never only in a seq/launch file.
# ============================================================================

# ---- tree/serving (values mirrored in fr13_required_tree_flags.sh) ----
export FR13_ATTN_KV_REMAP="${FR13_ATTN_KV_REMAP:-1}"          # garble fix (cat9 15/15->0/15)
export FR13_SLOT_REORDER="${FR13_SLOT_REORDER:-1}"            # FA2 accept M-dep fix
export FR13_COMMITTER_BATCHED="${FR13_COMMITTER_BATCHED:-1}"  # B2b batched committer
export FR13_KV_REMAP_SYNCFREE="${FR13_KV_REMAP_SYNCFREE:-1}"  # zero-host-sync remap
export FR13_INPUTPREP_GUARD="${FR13_INPUTPREP_GUARD:-1}"      # crash-fix stack
export FR13_RING_EXPORT="${FR13_RING_EXPORT:-1}"              # in-kernel ring staging
export FR13_CONV_WB_FUSED="${FR13_CONV_WB_FUSED:-1}"          # fused conv write-back
export FR13_DRAFT_VOCAB_K="${FR13_DRAFT_VOCAB_K:-65536}"      # DVK gather-64k (baked 2026-07-26; e3ca231ba)
export FR13_DRAFT_VOCAB_BLOCKS="${FR13_DRAFT_VOCAB_BLOCKS:-/workspace/scripts/fr13_dvk_subset_blocks.json}"
export FR13_DRAFTER_GRAPH="${FR13_DRAFTER_GRAPH:-1}"          # R4 drafter graph (-11ms; live-gated across all session arms)
export FR13_DM_DEPTHSYNC="${FR13_DM_DEPTHSYNC:-1}"            # depth-sync walk (-14..18ms; dscg PASS; byte gate 96/96)
export FR13_PARENT_GATHER="${FR13_PARENT_GATHER:-1}"          # lean stack (bsweep2 recomposition)
export FR13_CONV_PREGATHER="${FR13_CONV_PREGATHER:-1}"
export FR13_FLAGS_INKERNEL="${FR13_FLAGS_INKERNEL:-1}"
export FR13_SUBTREE_PARALLEL="${FR13_SUBTREE_PARALLEL:-1}"    # +4.7% B=1, byte-exact
export FR13_SUBTREE_PARALLEL_SELFCHECK="${FR13_SUBTREE_PARALLEL_SELFCHECK:-0}"
export FR13_HC_INTERNAL="${FR13_HC_INTERNAL:-0}"              # lean verdict: off
export FR13_ENABLE_APC="${FR13_ENABLE_APC:-1}"                # cache ON (goal = spec+cache)
export MAMBA_BLOCK_SIZE="${MAMBA_BLOCK_SIZE:-1024}"
export APC_BLOCK_SIZE="${APC_BLOCK_SIZE:-1024}"
export MAMBA_SSM_CACHE_DTYPE="${MAMBA_SSM_CACHE_DTYPE:-float32}"  # LOSSLESSNESS choice; change = behavior gate (PARKED)
# EXPERIMENT-ONLY (default off; own live gates): FR13_COMMITTER_GRAPH,
# FR13_TAW (benched: live FAIL, deployment-shape re-gate pending), FR13_STEP_GRAPH

# ---- proxy (offload, alienware) ----
export LUMO_PROXY_SSE_HEARTBEAT_S="${LUMO_PROXY_SSE_HEARTBEAT_S:-15}"  # EMIT-WEDGE protection (3a86c8a85; regressed to 0 until 4df608b75)
export LUMO_PROXY_SSE_LOG="${LUMO_PROXY_SSE_LOG:-1}"                   # per-request terminal-reason forensics
export LUMO_PROXY_SSE_CAPTURE_DIR="${LUMO_PROXY_SSE_CAPTURE_DIR:-/home/mark/lumo_proxy_offload/sse_capture}"  # wedge forensics (17d7b864b)
export LUMO_PROXY_QWEN_SAMPLING="${LUMO_PROXY_QWEN_SAMPLING:-1}"       # qwen sampling profile at the proxy
# DEPLOY_FORCE_TEMP -> LUMO_PROXY_FORCE_TEMPERATURE is set by the serve
# scripts from the driver's DEPLOY_FORCE_TEMP (canonical 0.6; never greedy).

# ---- agent (qwen-code in per-instance image) ----
# Set in run_swe_bench_q36_a.py docker-run assembly; listed here as the record:
#   QWEN_STREAM_IDLE_TIMEOUT_MS=600000   (stream-idle abort window)
#   QWEN_CODE_MAX_OUTPUT_TOKENS=32768    (turn cap)
# Nudge/retry: BANNED (nudge-free honesty gate; feedback_giveup_gate_nudge_and_agent)

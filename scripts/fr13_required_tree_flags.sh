# FR13_REQUIRED_TREE_FLAGS — SINGLE SOURCE OF TRUTH for env flags that MUST be ON
# for any branching-tree serving config (cat9/cat8/cat6/tail6/chain5/...).
#
# Why this file exists (2026-07-22): FR13_ATTN_KV_REMAP and FR13_SLOT_REORDER were
# both proven fixes ("BAKED" per project memory: project_fr13_garble_attn_kv_remap_fix.md,
# project_fr13_accept_mdep_fix_costgate.md) but were only ever hardcoded into
# fr13_launch_locked.sh and a dozen narrow one-off diagnostic scripts. The actual
# B4 agentic SWE-bench campaign path (fr13_launch_forked_fa2_tree_server.sh) never
# got them, so every tail6/cat8/cat6 campaign run through it -- weeks of runs --
# booted without the fixes. This file exists so that never happens again: update
# the list HERE ONLY, and every consumer (launcher default + assertion gate)
# picks it up automatically. Do not hardcode a copy of this list anywhere else.
#
# Consumers:
#   - fr13_launch_forked_fa2_tree_server.sh: sources this to set defaults
#   - fr13_bigdenom_swe_serve_variant.sh:    sources this to build its fail-loud
#                                             NEEDS assertion (tree-kind arms only)
#   - fr13_launch_locked.sh:                 sources this instead of its own
#                                             hardcoded `export FR13_ATTN_KV_REMAP=1`
#
# Format: "KEY=VALUE" strings, same shape docker -e / bash NEEDS arrays already use.
# Both flags are no-ops on non-branching configs (a linear chain's accepted path
# is already contiguous / M-independent), so defaulting them ON is behavior-
# preserving for every tree-launcher caller, not just branching kinds.
FR13_REQUIRED_TREE_FLAGS=(
  "FR13_ATTN_KV_REMAP=1"   # branching-tree foreign-KV garble fix (cat9 15/15->0/15)
  "FR13_SLOT_REORDER=1"    # FA2 accept M-dependence fix (superset +0.166 live-confirmed)
  "FR13_RING_EXPORT=1"     # in-kernel replay-ring staging (B1; rwb1 16-task gate 2026-07-23: accept 4.520, TPS 38.01 vs 32.85, pass 8/16 in-band)
  "FR13_CONV_WB_FUSED=1"   # fused conv-state write-back (B2a; same gate; both offline byte-gated + live A/B'd)
  "FR13_COMMITTER_BATCHED=1"  # B2b batched all-layers committer (tail6_batched_f70 16-task gate: span 47->36ms, verify FLAT, comb 3.592, 8P/8F; byte gate fr13_committer_graph_varying ALL-IDENTICAL)
  "FR13_KV_REMAP_SYNCFREE=1"  # zero-host-sync KV remap (patch-time baked; offline byte gate PASS; bv1 4-task gate 2026-07-24: accept 4.411, cfwd 12.33ms/event, 0 errors)
  "FR13_INPUTPREP_GUARD=1"    # draft-slot rescue + committed-slot async assert (crash-fix stack; r8 16-task survival + bv1 clean)
  "FR13_DRAFT_VOCAB_K=65536"  # BAKED 2026-07-26: gather-64k drafter head (measured 128-id-block subset, broad 3.14M-tok corpus; fp8 scale-aligned; verifier full-head=lossless). Live 4-task dvkg64L: accept 5.552 == full-head same-workload control 5.556 (dvkdump), drafter_gpu 94.9->56.3ms/step; exact draft-id coverage 97.1% == contig-128k at half the read. Set 0 for full head."
  # FR13_DRAFT_VOCAB_BLOCKS default = scripts/fr13_dvk_subset_blocks.json (launcher); unset BLOCKS with K>0 => contig slice (diagnostic only — NOT a valid frequency cap, see FR13_DVK_RESEARCH_BRIEF.md)
  # FR13_PARENT_GATHER REVERTED 2026-07-24: loop escalation common factor (bar16/bv4/bv5); bit-identity proven EAGER-only — graph-capture suspect. Loop epidemic later root-caused to HOST DRIVER degradation (reverts stand precautionary). Re-gate under capture (regate_queue.sh 2a) before any re-bake.
  # FR13_CONV_PREGATHER REVERTED 2026-07-24: loop escalation; token hole = col0 row change under stable req-id. FIX BUILT 2026-07-24: composite (req_ids, col0 page-ids) token — publish in _prepare_inputs + trigger/consume in lockstep; stage refuses without col0. Re-gate = regate_queue.sh 2b.
  # FR13_SSI_PREBUILD: unconditional kernel-lib code (no flag) — batched committer ssi broadcast; byte gate ALL-IDENTICAL + bv1 clean. Registry marker for drift awareness.
  # FR13_HC_INTERNAL QUEUED (built 2026-07-24, default OFF): scan h_cache compacted to internal-node rows (leaves never re-read; attacks scan +18.2 via register/local footprint). Gate = selfcheck eager + capture leg (regate_queue.sh 2d). Incompatible with PARENT_GATHER/PIGGYBACK (launcher raises).
  # FR13_CONV_NODEBANK QUEUED (built 2026-07-24, default OFF): conv node deposits -> private per-layer bank; pool keeps col0 only (spec-page surgery 1+2). Offline route byte gate CPU PASS 36/36 incl. ordinal-perm composition cases. Gate = regate_queue.sh 2e. Patch-time incompat raises: TCF_SELFCHECK / RUNROW_INIT!=1 / FULL_CAPTURE.
  # FR13_SPEC_BLOCKS_CAP QUEUED (built 2026-07-24, default 0=OFF): MambaSpec num_speculative_blocks min-cap (21->12 => 13 pages/request, ~+9 pages reclaimed) + gdn_attn page-col width caps. REQUIRES FR13_CONV_NODEBANK=1 (patcher main raises otherwise). Gate = regate_queue.sh 2f.
  # FR13_CONV_WB_BATCHED QUEUED (B2c, built 2026-07-24, default OFF): one batched conv writeback across requests (per-b launch loop -> 1-2 launches/layer; committer host-gap slice). Composes with NODEBANK. Gate = regate_queue.sh 2h.
)

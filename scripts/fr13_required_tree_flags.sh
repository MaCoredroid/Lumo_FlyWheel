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
  # --- cleanup+bake 2026-07-27 (FR13_CLEANUP_BAKE_PLAN.md; cleanbake1 gate PASS: 1P/3F band-in, tuple ON pooled line) ---
  # Registered HERE (not just launcher defaults) per the config-drift lesson in this file's header:
  # the registry is the single source AND the variant harness's fail-loud NEEDS assertion, so a
  # boot that loses any of these fails loud instead of silently drifting. Diagnostics that must
  # disable one (e.g. eager attribution probes with DRAFTER_GRAPH=0) boot via the launcher
  # directly, not the variant harness.
  "FR13_ENABLE_APC=1"         # cache ON (goal = spec+cache; APC lossless proven MISS==HIT)
  "FR13_TAW=1"                # ran clean in every S1/A-B arm
  "FR13_PARENT_GATHER=1"      # byte-identical selfcheck-gated (lean recomposition 2026-07-25)
  "FR13_COMMITTER_GRAPH=1"    # CG committer graph
  "FR13_CONV_PREGATHER=1"     # re-gated 2026-07-25 lean recomposition
  "FR13_FLAGS_INKERNEL=1"
  "FR13_SUBTREE_PARALLEL=1"   # #60: byte-exact + graph-safe + B=1 +4.7%
  "FR13_DRAFTER_GRAPH=1"      # R4 whole-spine drafter capture (the -40ms lever)
  # FR13_DRAFT_VOCAB_BLOCKS default = scripts/fr13_dvk_subset_blocks.json (launcher); unset BLOCKS with K>0 => contig slice (diagnostic only — NOT a valid frequency cap, see FR13_DVK_RESEARCH_BRIEF.md)
  # FR13_PARENT_GATHER RE-GATED 2026-07-25 via lean recomposition (bsweep2 29.02 winner; every arm since =1 incl bar19 8/16 band-pass). Old note: REVERTED 2026-07-24: loop escalation common factor (bar16/bv4/bv5); bit-identity proven EAGER-only — graph-capture suspect. Loop epidemic later root-caused to HOST DRIVER degradation (reverts stand precautionary). Re-gate under capture (regate_queue.sh 2a) before any re-bake.
  # FR13_CONV_PREGATHER RE-GATED 2026-07-25 (same lean recomposition). Old note: REVERTED 2026-07-24: loop escalation; token hole = col0 row change under stable req-id. FIX BUILT 2026-07-24: composite (req_ids, col0 page-ids) token — publish in _prepare_inputs + trigger/consume in lockstep; stage refuses without col0. Re-gate = regate_queue.sh 2b.
  # FR13_SSI_PREBUILD: unconditional kernel-lib code (no flag) — batched committer ssi broadcast; byte gate ALL-IDENTICAL + bv1 clean. Registry marker for drift awareness.
  # FR13_HC_INTERNAL RETIRED 2026-07-27 (cleanup+bake): never gated in, incompatible with the now-BAKED PARENT_GATHER. Mechanism hard-disabled (hc_internal_on()->False); env wiring removed. Kernel-body HC_MASK dead branches remain (excision = follow-up with its own bit-exact gate, FR13_CLEANUP_BAKE_PLAN.md).
  # FR13_CONV_NODEBANK DELETED 2026-07-25 (dce60d18c + f4f67f4e8, FR13_LEVER_REDESIGN.md): the isolation gate measured it BELOW the no-lever baseline -- 28.05 tps vs 32.14 -- so the bank tax outweighed the page reclaim. Full code deletion: kernel fns, patcher dual-arm writeback wrap, bank fetch, committer leaf bank read, builder preseeds; pool path collapsed unconditional. Env is NOT accepted -- it stays on the tcf fail-loud deleted-env list (patcher _fr13_tcf_env) so a =1 export raises instead of silently no-opping. Do not re-add without a fresh gate.
  # FR13_SPEC_BLOCKS_CAP DELETED 2026-07-25 (dce60d18c, 101 lines: env fn, mamba patch, consumer caps, preflight): isolation gate 29.62 tps, also BELOW the 32.14 no-lever baseline, and structurally tied to nodebank (capped pages needed bank storage for replay reads) so it deleted with it. The cache-hit-rate concern it addressed moved to the mamba_block_size 1024->8192 route (project_fr13_apc_blocksize_fix). Nothing reads this env any more.
  # FR13_MAMBA_SPEC_BLOCKS_CDIV QUEUED (built 2026-08-09, default 0=OFF): MambaSpec num_speculative_blocks -> cdiv(num_speculative_tokens, mamba_block_size) at the abstract.py construction site (31 -> 1 at 31/1024), i.e. 3*(1+1)=6 mamba pages/request instead of 3*(1+31)=96. Motivating measurement (20260809T064230Z salvage): at B4 the per-token reservation pins 384 of 692 pool pages (55%) before any caching and 89-93% of LRU evictions are mamba pops -- B4 APC hit 77% vs B1 89% on identical tasks. SOURCE-VERDICT 2026-08-09: DOES NOT SHIP AS-IS. num_speculative_blocks counts STATE SLOTS, not a token range (MambaSpec.page_size_bytes = sum(prod(shape)*itemsize), independent of block_size; block_size only sets prefix-cache token alignment), so ceil-dividing by mamba_block_size is a units error. Four live consumers index the slots PER DRAFT NODE: (1) stock FLA vllm/model_executor/layers/fla/ops/fused_recurrent.py stores one recurrent state per speculative token to ssm_state_indices[b, i_t] inside `for i_t in range(0, T)` and reloads column num_accepted_tokens-1; (2) gdn_linear_attn passes the whole [B, num_spec+1] window as ssm_state_indices; (3) FR13 tree conv writeback scatters tree_n node windows to spec_state_indices[b, :tree_n] (launch_conv_state_writeback raises when dst_rows.numel() < tree_n); (4) the accepted-path remap reads spec_state_indices[b, accepted_paths[b,k]]. gdn_attn slices block_table_tensor[mask, :num_spec+1] off a table mamba_get_block_table_tensor already narrowed to 1+num_speculative_blocks columns and copy_s it into a persistent (max_bs, num_spec+1) buffer, so a narrower reservation shape-mismatches or short-feeds the kernels. The patcher's main() preflight (_fr13_assert_mamba_spec_blocks_cdiv_slot_demand) therefore REFUSES =1 while 1+cdiv(...) < num_speculative_tokens+1 -- it opens by itself once a per-node scratch rehome lowers the demand. That rehome is the deleted FR13_CONV_NODEBANK + FR13_SPEC_BLOCKS_CAP pairing below (28.05 / 29.62 tps vs a 32.14 no-lever baseline), so re-landing it needs a fresh design, not a re-revert. Gate = boot diag + recurrent-oracle lossless + exact4 B4 pair, ALL blocked on that rehome.
  "FR13_CONV_WB_BATCHED=1"  # B2c BAKED 2026-07-27: offline byte gate PASS (07-24) + b2c1 band PASS (2P/2F garble-free) + subspan1 speed positive (sfwd fit -6.7 fixed/-2.4 per event; step -20 vs model at eps 2.7); batched conv writeback across requests.
)

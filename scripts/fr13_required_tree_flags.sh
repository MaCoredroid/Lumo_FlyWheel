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
  # FR13_MAMBA_SPEC_BLOCKS_CDIV QUEUED (built 2026-08-09, default 0=OFF): the B4 mamba page lever, ONE flag arming BOTH halves coherently. (1) PHYSICAL: MambaSpec num_speculative_blocks -> cdiv(num_speculative_tokens, mamba_block_size) at the abstract.py construction site (31 -> 1 at 31/1024), so MambaManager's align first-prefill branch allocates 1+1=2 pages/group instead of 1+31=32, i.e. 3*2=6 mamba pages/request instead of 3*32=96. (2) LOGICAL: _patch_gdn_attn_mamba_spec_scratch_table rewrites both gdn_attn spec-window slice sites to republish the single align spare page across logical columns 1..num_spec, so every consumer still sees a num_spec+1 window and NOTHING downstream is narrowed. Motivating measurement (20260809T064230Z salvage): at B4 the per-token reservation pins 384 of 692 pool pages (55%) before any caching and 89-93% of LRU evictions are mamba pops -- B4 APC hit 77% vs B1 89% on identical tasks. SOURCE-VERDICT 2026-08-09 (SUPERSEDES the 2026-08-09 'DOES NOT SHIP AS-IS' verdict, which was correct only for the physical half alone): the units-error objection stands -- num_speculative_blocks counts STATE SLOTS, not a token range -- but the four per-draft-node consumers it named are now satisfied by construction rather than by reservation. (1) stock FLA fused_recurrent.py/fused_sigmoid_gating.py store per i_t to ssm_state_indices[b,i_t]: absorbed by the shared scratch column; h0 is loaded to registers ONCE before the i_t loop opens, and the served committer omits num_accepted_tokens so the load is i_t=0. (2) gdn_linear_attn still receives the full [B,num_spec+1] window. (3) the tree conv writeback guard (launch_conv_state_writeback raises when dst_rows.numel()<tree_n) never observes the narrowing because the logical table keeps its width -- and it is unreachable under fixed32 anyway: both call sites are gated off by FR13_CONV_WB_BATCHED=1 (hard-required '1' by the patcher preflight) and by a literal `not _FR13_FIXED32_MODE`. (4) the accepted-path remap is off under fixed32 (full_node_writebacks==0 and conv_remaps==0, asserted every event by _fr13_fixed32_conv_runtime_contract / _fr13_fixed32_observed_commit). Every served reader is col0: conv commit route 'fixed32_direct_source_col0' (+0*ssi_stride_s), conv prior gather col0, committer h0 col0 (RUNROW_INIT), patched get_temporal_copy_spec reads block_ids[cur_block_idx] under FR13_APC_COMMIT_TO_RUNNING_ROW=1 (fail-loud if commit/init disagree). The deployed CUDA-graph committer ALREADY hands stock FLA a fully aliased 16-column table of one repeated page id, so column aliasing is the established production pattern, not a new one; the committer's RUNROW_COMMIT deposit makes col0 authoritative regardless of what the per-t stores did. Safety of the shared scratch: it is not a new entity but the align spare itself, held ONCE in the scheduler's req_to_blocks (duplication exists only in the downstream logical tensor), so ref-counts, free_blocks and the APC hash path keep stock semantics -- no double-free, leak, or double-hash. Garbage in it is inert: the only way it is read is by becoming the running state at a block rollover, and preprocess_mamba copies the live state into it first. It must be a real page (>0): the conv row guard loads all num_spec+1 columns and requires each strictly inside (0,BANK_ROWS), which is why the preflight now enforces a floor of 2 rather than opening unconditionally. KNOWN CONSTRAINT: do not combine with a diagnostic arm that re-enables the eager per-request conv writeback (FR13_CONV_WB_FUSED!=1, FR13_TCF_SELFCHECK=1, FR10_METRICS=1 with tree conv diag, or the commit handoff logs) -- that path falls through to a raw conv_state.index_copy_ over spec_state_indices[b,:tree_n], which with aliased columns rewrites one page instead of raising. This is NOT a re-revert of FR13_CONV_NODEBANK + FR13_SPEC_BLOCKS_CAP (28.05 / 29.62 tps vs a 32.14 no-lever baseline, deleted 2026-07-25 dce60d18c): that pairing narrowed the CONSUMER widths and so short-fed the per-node kernels, which is exactly why it needed a node bank. Here consumer widths are untouched. Preflight _fr13_assert_mamba_spec_blocks_cdiv_slot_demand now enforces 1+cdiv(...) >= 2 (col0 + one scratch) and main() adds _fr13_assert_mamba_spec_blocks_cdiv_coherent so a half-applied pair (anchor drift on either site) fails loud instead of silently short-feeding. Gate = boot diag + recurrent-oracle lossless + exact4 B4 pair.
  "FR13_CONV_WB_BATCHED=1"  # B2c BAKED 2026-07-27: offline byte gate PASS (07-24) + b2c1 band PASS (2P/2F garble-free) + subspan1 speed positive (sfwd fit -6.7 fixed/-2.4 per event; step -20 vs model at eps 2.7); batched conv writeback across requests.
)

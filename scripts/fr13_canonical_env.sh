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
export FR13_ENABLE_APC="${FR13_ENABLE_APC:-1}"                # cache ON (goal = spec+cache)
export MAMBA_BLOCK_SIZE="${MAMBA_BLOCK_SIZE:-1024}"
export APC_BLOCK_SIZE="${APC_BLOCK_SIZE:-1024}"
export MAMBA_SSM_CACHE_DTYPE="${MAMBA_SSM_CACHE_DTYPE:-float32}"  # LOSSLESSNESS choice; change = behavior gate (PARKED)
export FR13_MAMBA_SPEC_BLOCKS_CDIV="${FR13_MAMBA_SPEC_BLOCKS_CDIV:-1}"  # PROMOTED 2026-08-10 (Mark: "just flip it"), default ON, fixed32-only (fail-loud guard 4b3c7f8d4; 2-slot floor preflight). Evidence: within-run only-arm-delta pair output/fr13_mamba_narrow_withinrun_abc49506a_20260810T002124Z (+9.7pp APC, per-request TPS 15.07->18.00, KV peak 74->18%). B4 mamba page lever. ONE flag arms BOTH halves: MambaSpec num_speculative_blocks -> cdiv(num_spec_tokens, MAMBA_BLOCK_SIZE) (2 physical pages/group/request, not 32) AND the gdn_attn scratch-window rehome that keeps the logical spec window num_spec+1 wide over those 2 pages. Preflight now enforces the 2-slot floor (col0 + one real scratch page) instead of refusing; see fr13_required_tree_flags.sh
export FR13_FA2_QROW32_B1_PRODUCTION_ARM_DEFAULT="${FR13_FA2_QROW32_B1_PRODUCTION_ARM_DEFAULT:-gqa_pair}"  # PROMOTED 2026-08-13 (Mark: "B1 flip Yes"). The arm B1 SERVING takes when a launch does not name one. Evidence: Tier-A byte gate PASS + re-seals for the GQA-pair B1 unit (.so 3560cdc0c1ebbe3d912858ea447b350edefc0d6749950d6353e5f763185da6ae, 299815552 B, closure 172b5e7131841ce45650bb8eea35f0b427ca660ce8f145bd39b55b00a336ebf4, pinned by 3120b3765) and the exact4 real-traffic timing pair output/fr13_fa2_qrow32_gqa_pair_b1_timing_20260812T073429Z: step_wall 232.360 ms vs the qrow16 incumbent 236.765 ms (-4.405 ms, -1.86%), per-request TPS 22.769 -> 23.155, promotion_eligible=true.
# WHY THIS IS *_DEFAULT AND NOT FR13_FA2_QROW32_B1_PRODUCTION_ARM ITSELF: the B1
# production selector is batch-1-only -- the launcher refuses ANY B1 selector
# unless MAX_NUM_SEQS==1 && SWE_CONCURRENCY==1 (_FR13_FA2_QROW32_B1_CANDIDATE_MODE)
# -- and this registry is sourced by fr13_b4_campaign_driver.sh at line 29, seven
# lines BEFORE BSIZE is read. Exporting the selector here would hand every B=4
# campaign arm a B1 selector it must then refuse at boot. So the registry owns the
# VALUE and the launcher applies it in the one shape where it is legal; the two
# must never disagree (tests/test_fr13_qrow32_b1_production_default.py parses both
# files and compares, exactly as the mamba-narrowing promotion is pinned).
# A caller that NAMES FR13_FA2_QROW32_B1_PRODUCTION_ARM -- including naming it
# empty -- is always obeyed; the default applies only when it is unset.
export FR13_FA2_QROW32_B4_PRODUCTION_ARM_DEFAULT="${FR13_FA2_QROW32_B4_PRODUCTION_ARM_DEFAULT:-gqa_pair}"  # PROMOTED 2026-08-14 (Mark's B4 production default flip ruling, the B4 analogue of the B1 flip 99a511319). The arm B4 SERVING takes when a credentialed launch does not name one. The unit is the sealed PADDED GQA-pair kernel: .so af9e9f24335db899468032f5b5a3eba100febe294932533cb9b87163ce2b3fdb, 299813360 B, closure 9c3f9e751da7b783e9d07d8e40d5bc2234b99e719a1048668bd6c82244ed2d81, candidate_scope final_fixed32_b34_full_graph_only, production_widths 3,4. Evidence: four-pass Hydra27 sealing campaign output/fr13_b4_hydra27_sealing_campaign_20260814T011514Z verdict SEALED_HYDRA27_GAIN -- batch-conditioned width-4 improvement mean 27.03 ms/step, one-sided 95% lower bound 10.82 ms/step, n=4 with balanced SC/CS arm order, placebo width clean in every pass, sealed MDE 4.20 ms; the padded pair output/fr13_b4_width4_timing_padded_20260813T201426Z, width-3 treated 25.16 ms/step against the pre-registered 25.6 +/- 3.5 ms/step; the b34 dual raw-byte gates 0/0 output+LSE mismatches clean AND poisoned-shadow at widths 3 and 4 at two successive commits (output/fr13_fa2_gqa_pair_b34_dual_byte_gate_20260813T173106Z and ...T234051Z); exact16 agent QC PASS results/fr13_b4_exact16_qc_20260814 -- 9/16 resolved = the 21-arm historical median, band 8-11, zero giveups, zero always-resolves regressions.
# WHY THIS IS *_DEFAULT AND NOT FR13_FA2_QROW32_B4_PRODUCTION_ARM ITSELF: the same
# reason as the B1 entry above, from the other side. This registry is sourced by
# fr13_b4_campaign_driver.sh before BSIZE is read, so it cannot see whether the
# launch about to happen is a B4 serve at all -- and the B4 production selector is
# a CREDENTIALED selector: exporting it here would hand every arm the registry
# reaches (including the B=1 arms the same driver runs) a selector the launcher
# must then refuse at boot for want of a sealed dual gate. So the registry owns
# the VALUE and the launcher applies it in the one shape where it is legal:
# fixed32 B=4 / concurrency 4 serving that PRESENTS the sealed b34 dual byte gate.
# That last clause is the B4-specific part of the scope and it is load-bearing.
# B=1/concurrency 1 is a rare shape, so the B1 promotion could key on batch alone;
# B=4/concurrency 4 fixed32 is the shape of the ENTIRE campaign, so batch alone
# would retarget every floor gate, timing pair and live gate in the tree. The
# credential is what separates a production serve from a campaign arm: only a
# launch holding the sealed gate can serve this kernel at all, so only such a
# launch is promoted. Everything else is left exactly where it was.
# A caller that NAMES FR13_FA2_QROW32_B4_PRODUCTION_ARM -- including naming it
# empty -- is always obeyed; the default applies only when it is unset. The two
# files must never disagree (tests/test_fr13_qrow32_b4_production_default.py
# parses both and compares, exactly as the B1 flip and the mamba narrowing are).
export FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_DEFAULT="${FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_DEFAULT:-0}"  # BUILT 2026-08-14, DELIBERATELY OFF. Whether a credentialed fixed32 serve that does not name the GDN single-launch arm gets it. The unit is the folded GDN scan kernel fixed32_gdn_single_launch_tree_v2 -- one physical launch per layer for the whole batch against the deployed two-launch reference's two per request-layer, eliminating the L0->L1 state export/parent-read handoff entirely (state_export_writes 0, state_parent_reads 0).
# WHY THIS SHIPS AS 0 WHILE THE TWO FA2 ENTRIES ABOVE SHIP AS gqa_pair: those two
# were promoted on SEALED gain. This one is not sealed. What it has is legality
# and a price, and those are different things:
#   * LEGALITY (phase 1, output/fr13_gdn_single_launch_b4_gate_20260814T191621Z):
#     the b4 Hydra27 live byte gate returned PASS on real SWE-Verified traffic --
#     1,584 records (33 comparator events x 48 GDN layers) compared across all 7
#     surfaces (output, ring_k, ring_v, ring_a, ring_b, flags, counter) with
#     raw_byte_equal true, every event at request-tuple width 4. The B1 arm was
#     sealed earlier at hydra27:b1. So the kernel is byte-identical to the
#     incumbent at the width-4 operating point. That is a licence to SERVE, not a
#     reason to serve by DEFAULT.
#   * PRICE (phase 0, results/fr13_gdn_scan_b4_probe_20260814): directly measured
#     at b=4, not transferred -- two_launch 861.504 us/layer-batch vs
#     single_launch 674.336, a saving of 8.984 ms/step over 48 layers = 2.14x the
#     4.20 ms sealed MDE. Super-linear in width: 4.668x the b=1 saving.
# What is MISSING is the thing the FA2 entries have and this one does not: a
# sealing campaign -- paired passes at the Hydra27 width-4 operating point with a
# one-sided 95% lower bound on the pre-registered batch-conditioned width-4
# improvement, balanced arm order and a clean placebo width. Phase 3's n=1 screen
# is a SCREEN; it can halt the lever but it cannot seal it. Until that campaign
# runs and Mark rules on it, an unsealed kernel must not become what the tree
# serves when nobody asked. Flipping this value to 1 is the whole of the
# promotion: every guard, pin, credential check and engagement needle below is
# already built and already exercised by the tests, so the flip is a one-token
# change to this line and nothing else.
# WHY THIS IS *_DEFAULT AND NOT FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION ITSELF:
# the same reason as the two entries above. This registry is sourced before BSIZE
# is known, and the GDN single-launch production selector is a CREDENTIALED
# selector the launcher must refuse wherever no HEAD-bound gate credential was
# presented. So the registry owns the VALUE and the launcher applies it in the one
# shape where it is legal: fixed32 B1/B4 at matching concurrency, K64/root1, BV=8,
# FULL_AND_PIECEWISE, presenting a PASS credential bound to the exact serving
# HEAD. A caller that NAMES FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION -- including
# naming it 0 -- is always obeyed; the default applies only when it is unset. The
# two files must never disagree (tests/test_fr13_gdn_single_launch_production_default.py
# parses both and compares, exactly as the two FA2 flips are).
# BAKED 2026-07-27 (cleanup+bake, launcher defaults now 1): FR13_COMMITTER_GRAPH,
# FR13_TAW, FR13_DRAFTER_GRAPH, FR13_PARENT_GATHER (all ran clean in every S1/A-B arm).
# EXPERIMENT-ONLY (default off): FR13_STEP_GRAPH (=1/=2/=3 capture modes; A/B verdict:
# no speed effect vs staged — see FR13_CLEANUP_BAKE_PLAN.md #72 closure).
# DELETED 2026-07-27: FR13_HC_INTERNAL (retired mechanism), FR13_APC_BURN_NODE_BANK.
# DELETED 2026-07-25: FR13_CONV_NODEBANK, FR13_SPEC_BLOCKS_CAP (both measured BELOW
# the 32.14 no-lever baseline -- 28.05 / 29.62 tps; FR13_LEVER_REDESIGN.md). Neither
# env is read anywhere; do not re-export them. The launcher forwarded both until
# 2026-08-09 even though the implementations went at 2026-07-25.

# ---- proxy (offload, alienware) ----
export LUMO_PROXY_SSE_HEARTBEAT_S="${LUMO_PROXY_SSE_HEARTBEAT_S:-15}"  # EMIT-WEDGE protection (3a86c8a85; regressed to 0 until 4df608b75)
export LUMO_PROXY_SSE_LOG="${LUMO_PROXY_SSE_LOG:-1}"                   # per-request terminal-reason forensics
export LUMO_PROXY_SSE_CAPTURE_DIR="${LUMO_PROXY_SSE_CAPTURE_DIR:-/home/mark/lumo_proxy_offload/sse_capture}"  # wedge forensics (17d7b864b)
export LUMO_PROXY_QWEN_SAMPLING="${LUMO_PROXY_QWEN_SAMPLING:-1}"       # qwen sampling profile at the proxy
# DEPLOY_FORCE_TEMP -> LUMO_PROXY_FORCE_TEMPERATURE is set by the serve
# scripts from the driver's DEPLOY_FORCE_TEMP (canonical 0.6; never greedy).

# ---- runner (GB10 host process) ----
export LUMO_SWE_STALL_KILL_S="${LUMO_SWE_STALL_KILL_S:-900}"  # trace-growth stall watchdog (AUDIT #2: env was read by run_swe_bench_q36_a.py but SET NOWHERE => watchdog silently OFF; the session wedges would have been auto-killed by it. 900s > the 600s agent stream-idle window so it never races a healthy long turn)

# ---- agent (qwen-code in per-instance image) ----
# Set in run_swe_bench_q36_a.py docker-run assembly; listed here as the record:
#   QWEN_STREAM_IDLE_TIMEOUT_MS=600000   (stream-idle abort window)
#   QWEN_CODE_MAX_OUTPUT_TOKENS=32768    (turn cap)
# Nudge/retry: BANNED (nudge-free honesty gate; feedback_giveup_gate_nudge_and_agent)

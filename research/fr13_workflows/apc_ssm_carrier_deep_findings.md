# APC SSM align-snapshot carrier — deep findings (2026-06-20)

Marathon debug of the SSM/temporal align-snapshot poison carrier for APC + the GDN tree committer.
Branch fr13-prefix-cache. ~13 GPU drills + 4 CPU workflows. **NOT yet fixed.** Banking the full chain so
we don't re-derive it.

## The carrier (confirmed)
APC's align cache snapshots/restores recurrent SSM state at block boundaries (preprocess_mamba=restore,
postprocess_mamba=snapshot, both -> collect_mamba_copy_meta -> get_temporal_copy_spec). Stock reads
`state[block_ids[cur_block_idx + num_accepted - 1]]`. The TREE committer keeps its accepted recurrent
state in a NODE BANK (spec_state_indices / layer._fr13_replay_spec_idx), NOT that block row -> the
snapshot saves a STALE row -> GDN state poisoned on the next cache-hit -> garble -> SWE agent gives up
empty (cache-OFF runs full wall + real patch). White-box: Tap-C postprocess stale_read on
get_temporal_copy_spec. Black-box (GROUND TRUTH): garble scan + agent gave_up_empty on astropy-13033.

## Fixes attempted (chain — each fixed, each exposed the next)
1. NameError os in rejection_sampler injection -> local import (f7f950fc).
2. req_id keying: keyed leaf map by _LUMO_FA_SAMPLER_ROW_REQ_IDS (full sampler) vs spec rows -> key by
   _LUMO_FA_SPEC_ROW_REQ_IDS (ecb6bc43). Tap-A producer had same bug -> fixed (0bf77911).
3. Override get_temporal_copy_spec reads module-global _FR13_CUR_SSM_LEAF_ROW: **PROVEN never lands** —
   FR13_OV_DIAG showed bare_leaf=None for ALL override calls even with the leaf set right before the call;
   same_module=True (not a module-identity bug). Cause: the align batches ALL reqs' bias-chokepoint calls
   then ALL copies; the single global is clobbered by the last (found=False) bias before the override reads.
4. Direct copy_spec pointer substitution in collect (state has the leaf + the tensor) (634bace0): the
   substitution MECHANISM works (FR13_SUB_DIAG fires, executes BEFORE the Tap-C read so the tap sees it).
   BUT it fires in PREPROCESS, not the POSTPROCESS snapshot. Black-box: agent STILL gave up empty.
5. Leaf off-by-one: _FR13_APC_SSM_LEAF_BY_REQ holds +1 node ids (81 vs committed 80). CORRECT leaf =
   _FR13_BOUNDARY_LAST_WRITTEN_BY_REQ[req].rows[-1] (the Tap-A producer's actual written bank rows).
   Switched to it (523cd816) — still postprocess didn't fire.

## ROOT of the remaining blocker (workflow w7xypeplq)
preprocess_mamba and postprocess_mamba call collect for **overlapping but NON-identical request cohorts**:
 - preprocess: `if prev_state_idx != -1 and prev_state_idx != curr_state_idx` (state-index transition), bias
   = num_accepted-1 (>0).
 - postprocess: `if aligned_new_computed_tokens >= num_tokens_running_state` (block alignment), bias can be 0.
So some postprocess snapshot reqs have NO _FR13_BOUNDARY_LAST_WRITTEN_BY_REQ entry -> _fr13_apc_leaf None ->
no substitution -> stock. **UNEXPLAINED GAP**: the Tap-C stale records (postprocess, src_row=stock 40,
last_written=[70,37]) DO have a map entry (Tap-C computed stale, needs it) yet still read stock — so even
committed postprocess reqs aren't substituting. Code-reads can't close this; needs a unified per-copy runtime
diagnostic (phase + req_id + _fr13_apc_leaf + condition-passed + resulting src_row, both branches).

## Key equivalence (important)
Pointer-substitution and "write-through" produce the SAME end state (state[dest]=state[leaf]) and face the
SAME cohort/timing issue. write-through at COLLECT is not fundamentally different. A genuinely different fix
would write the leaf into the req's mamba state slot at the COMMIT site (forward), so the regular decode AND
the stock align both read correct state — but that needs the block layout at commit time.

## Diagnostics / flags added (gated, default off)
FR13_APC_SSM_SNAPSHOT (the fix), FR13_APC_SSM_DIAG (FR13_SUB_DIAG / FR13_CMM_DIAG / FR13_OV_DIAG probes).
Forwarded in the launcher -e list. Tap-A producer + Tap-C are FR13_REPLAY_BOUNDARY_LOG-gated, LAYER 0 only.

## Operational notes
- Killing a drill leaves swap that trips pre-boot hygiene; loop recover_host_memory until swap<1GiB before relaunch.
- The stale_read white-box is necessary-not-sufficient; gate on the BLACK-BOX (garble + agent full wall) — the
  white-box mis-led repeatedly (contaminated Tap-A keying; substitution-vs-tap order; cohort coverage).

## Status: OPEN. The snapshot-side approaches are confirmed not fixing the black-box. Next decision: (A) one
unified per-copy diagnostic drill to close the committed-postprocess gap, or (B) commit-site write-through, or
(C) reconsider APC scope for tree-spec. Awaiting user steer on resource investment.

## UPDATE: cohort mismatch CONFIRMED (unidiag2, periodic diag) — snapshot-side approach is a dead end
Steady-state per-copy diag: leaf=None for 48/50 temporal copies; leaf=VAL only 2/50, and when found the
substitution fires CORRECTLY (leaf=80 in last_written, vs stock 87). So the collect-level substitution is
correct when it fires but the leaf source covers almost none of the align's snapshot cohort. The align
snapshots a request cohort that is mostly NOT in the committed-leaf map -> fundamentally cannot be fixed at
the snapshot/collect site. PIVOT: commit-site write-through (ensure the committed leaf recurrent state sits
in the exact slot the STOCK align snapshot reads, for ALL committed reqs, at commit time) — needs the GDN
regular-decode state-slot addressing + how it relates to the align's block_ids source. CPU workflow next.

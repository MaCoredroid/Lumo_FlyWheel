# APC flag cleanup + new-road plan (2026-06-22)

NEW ROAD (user-chosen, option **a**): port SGLang's **verbatim snapshot** discipline onto
vLLM `align` — snapshot the EXACT committed conv+recurrent (SSM) state at the cache boundary
and restore it VERBATIM on a cache-hit, instead of letting align RECONSTRUCT it. All the
prior APC carrier theories are dead ends and their flags are dead code to remove.

## Why the priors are dead (one line each)
- chunk-vs-recurrent "153x carrier" = CONFOUND: it's a universal harmless kernel constant
  (genuine spine shows 158x too, and spine ships); measured vs a recurrent PROXY, not the
  cache-OFF incumbent. (audit wsh8imqxe + spine_probe FLASH_ATTN num_spec=5).
- cache-ON vs cache-OFF greedy byte-AB DIVERGES for BOTH spine (char23) AND tree (char11)
  => byte-AB too coarse (chunk-noise forks greedy both); spine NOT byte-lossless, only
  within-floor. So lossless is a per-token flip-RATE question, not byte.
- TTFT-WIN is REAL and banked: APC 2.0-2.2x (prefill ~5.5s -> ~2.5s).

## CLEANUP CONSTRAINT — KEEP the publisher + leaf-map (FR13_APC_VERBATIM reuses them)
FR13_APC_VERBATIM (the IN-FLIGHT SSM-axis tree fix, evidence in `FR13_APC_STATUS.md`) REUSES
the publisher `_fr13_publish_apc_ssm_leaf` and the `_FR13_APC_SSM_LEAF_BY_REQ` leaf-map. The
decouple agent made these FIRE under VERBATIM (the publisher early-returns unless SSM_SNAPSHOT
OR VERBATIM is set), INCLUDING the FR13_EAGER_PACK committer guard. So the dead-code cleanup
MUST KEEP:
- `_fr13_publish_apc_ssm_leaf` (read-only leaf-row publisher; safe under VERBATIM)
- `_FR13_APC_SSM_LEAF_BY_REQ` (the leaf-map it populates)
- the FR13_EAGER_PACK committer guard that lets it fire
ONLY the WRONG-ROW WRITERS get removed: the `get_temporal_copy_spec` SSM_SNAPSHOT redirect,
`FR13_APC_SSM_WRITE_THROUGH`, `FR13_APC_COMMIT_SITE_WT`, `FR13_APC_ALIGN_TREE_AWARE`.
(NOTE: items in REMOVE below that touch the publisher/leaf-map — e.g. the leaf-sub line —
must be re-scoped to "remove the writer, keep the publisher+map" per this constraint.)

## REMOVE (dead, all default-off so removal is byte-neutral; remove flag + patch-fn + registration + launcher -e + harness forward)
- FR13_APC_VALUE_VS_ORACLE (+_LOG/_FH)            — confounded probe (chunk-vs-recur red herring)
- FR13_APC_CACHEHIT_VALUE_PROBE  / fn _patch_apc_cachehit_value_probe  — confounded (vs leaf, row-index)
- FR13_APC_ALIGN_TREE_AWARE       / fn _patch_mamba_utils_apc_align_tree_aware — every-step harmful, failed
- FR13_APC_COMMIT_SITE_WT          — wrong-row write-through, failed
- FR13_APC_SSM_WRITE_THROUGH (+ _SSM_SNAPSHOT_SUB) — substitution, "CONFIRMED NOT FIXING"
- _patch_mamba_utils_collect_apc_leaf + _SSM_COLLECT_LEAF — leaf-sub WRITER (dead). NOTE: KEEP
  FR13_APC_SSM_LEAF_BY_REQ + the publisher `_fr13_publish_apc_ssm_leaf` (VERBATIM reuses them —
  see CLEANUP CONSTRAINT above); remove only the substitution writer, not the map/publisher.
- FR13_APC_HIT_RECURRENT_SUFFIX + FR13_APC_HIT_SUFFIX_CAP — uncapped recurrent (treated non-cause, slow)
- FR13_APC_MROPE_TAIL_ZERO         — captured-op, ruled out (garbles in eager too)
- FR13_APC_DROP_FINAL_BLOCK        / fn _patch_mamba_drop_final_block_43650 — #43650 REFUTED
- FR13_APC_POS_PROBE (+_FH/_STEP/_LOG) / fn _patch_gpu_model_runner_apc_pos_probe — dead diag
- FR13_APC_STATE_PROBE             / fn _patch_apc_state_probe — dead diag
- FR13_APC_GRAPH_REPLAY_BARRIER (+_DEBUG/_FIRED) — fired, no help
- FR13_APC_INDEX_RERESOLVE         — dead
- FR13_APC_SSM_DIAG / FR13_APC_DIAG / _DIAG_N — diagnostics tied to the dead WT/snapshot family

## KEEP
- FR13_ENABLE_APC (master) + base SGLang disciplines that WORK: --mamba-block-size,
  --mamba-ssm-cache-dtype float32, --max-num-batched-tokens, --enable-prefix-caching,
  --enable-chunked-prefill  (give the 2x TTFT win + spine within-floor)
- FR13_APC_CACHE_AB (+_LOG/_BLOCK/_SEEN/_FH) — the CORRECT cache-ON vs cache-OFF instrument
  (validation tool for the new road; has a matching bug to fix: align ON boundary block-aligned w/ OFF)
- FR13_APC_BLOCK_ALIGN_45477 — real block-align backport
- CUDAGRAPH_MODE knob (PIECEWISE)
- FR13_FULL_ATTN_KV_FP8 (gated discriminator, off)

## PENDING (the design wbppd4r72 decides extend-vs-replace)
- FR13_APC_CONV_SNAPSHOT / FR13_APC_SSM_SNAPSHOT / FR13_APC_CONV_FIX — the PARTIAL verbatim
  attempts. The new complete verbatim snapshot either completes or supersedes these.

## SEQUENCING
1. WAIT for design wbppd4r72 (do NOT edit patcher mid-workflow — corrupts its reads).
2. ONE clean pass: remove the DEAD set above + implement the new verbatim snapshot per the
   design (extend/replace the PENDING flags as the design says).
3. Verify APC default-OFF path byte-identical (ast.parse + test-apply + idempotent).
4. Validate new road: cache-AB |h_on-h_off|->0 (fix matching bug) + per-token cache-ON vs
   cache-OFF within-floor, BOTH spine + tree; agentic 12907 solves; TTFT-win holds -> bake.

Branch fr13-prefix-cache. Patcher = scripts/fr10_phase4_patch_vllm_tree_gdn.py.

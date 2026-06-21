# APC (GDN prefix-cache) fix state — cleaned up + baked (2026-06-21)

Branch `fr13-prefix-cache`. Current, honest state of the APC work after the carrier was isolated.
Supersedes the over-optimistic `apc_ssm_carrier_FIXED.md` (the SSM write-through is a PARTIAL sub-fix,
not full losslessness).

## BAKED (default-ON whenever APC is enabled; non-APC path byte-identical)
Launcher `fr13_launch_forked_fa2_tree_server.sh`, inside the `FR13_ENABLE_APC==1` block:
- `FR13_APC_SSM_WRITE_THROUGH=1` — the proven sub-fix. In `collect_mamba_copy_meta`, writes the
  committed accepted-leaf SSM value into the exact row the stock align snapshot reads (dest from
  `copy_spec.start_addr`, leaf from the publish map). Byte-lossless for the single-hit A/B + the
  spec-decode-boundary (num_accepted>1) cohort. Validated: `fr13_apc_lossless_ab.sh` MATCH.
- `FR13_APC_SSM_SNAPSHOT=1` — populates the committed-leaf map the WT consumes.
- `FR13_APC_CONV_FIX=1` / `FR13_APC_CONV_SNAPSHOT=1` — conv-window override on `get_conv_copy_spec`.
  NOTE: for OUR GDN the conv is FUSED (`FR13_TREE_CONV_FUSED`), so `get_conv_copy_spec` isn't on the
  copy path (state-probe saw only the 4-D SSM) → this override is ~a no-op for cat9; kept gated for
  generality/other layouts.
- `FR13_APC_BLOCK_ALIGN_45477=1` (in the `-e` list, correctness/always-on) — backport of vLLM PR
  #45477 (chunk END block-alignment in `_mamba_block_aligned_split`). NO-OP in our B=1/aligned-budget
  config (`num_computed_tokens` already block-aligned; `block_size`=832=13×64 is chunk-phase-aligned),
  but a valid correctness fix for the budget-fragmented mid-block-snapshot case.

Non-APC safety: with APC off, `--enable-prefix-caching` is absent → mamba `align` mode is off →
`preprocess/postprocess_mamba`, `get_*_copy_spec`, the collect injects, and `_mamba_block_aligned_split`
are not invoked. So the locked cat9 path is byte-identical regardless of these defaults.

## DIAGNOSTICS (default-OFF; do NOT enable in deployment)
- `FR13_APC_STATE_PROBE` — conv/ssm src/dst checksum per copy (isolation probe; heavy CPU sync).
- `FR13_APC_SSM_DIAG` — `FR13_WT_DIAG`/`FR13_SUB_DIAG` prints. **Misbehaves under CUDA-graph** — eager only.
- `FR13_REPLAY_BOUNDARY_LOG` — Tap-A producer (layer-0, eager, torch.cuda.synchronize).

## DEAD / superseded (gated, never runs in deployment)
- The option-1 pointer-SUB in `_patch_mamba_utils_collect_apc_leaf` (`#..._SUB`): only fires when
  `FR13_APC_SSM_WRITE_THROUGH != 1`, but WT is baked default-on → dead. Kept as a no-WT A/B fallback.

## OPEN — the real remaining carrier (NOT fixed by the baked sub-fixes)
APC is **not yet fully lossless**. Probe-isolated carrier (`42df9e89`): the GDN recurrent SSM state at a
cache boundary is **chunk-vs-recurrent dependent** — a cache-hit's chunked suffix-prefill (FLA
`chunk_gated_delta_rule`, CHUNK_SIZE=64) computes a *different* SSM than a contiguous full-prefill (probe:
SSM checksum varies at abs=3328/4992, stable at 1664). Phase is fine (832=13×64); the divergence is the
**fp-numerics / reduction order** of the chunked kernel. This is vLLM **#43559**. Fix in progress = route
the cache-hit suffix-prefill through our **bit-exact sequential rank-1 scan** (the allowed, non-WY path;
`fr10_gdn_tree_kernel.py`) instead of FLA's chunked kernel (workflow wvbmcp1ye). It is option-2a
(recompute-from-spine, exact but token-sequential) — NOT a WY revival.

## Measured wins (banked, independent of the open carrier)
- **TTFT 2.09×** (cache-hit prefill 2.54s vs miss 5.31s, A/B) + **78.7% prompt-token cache reuse** on the
  live 12907 workload. Decode-TPS unchanged (APC is prefill-only).

## Reusable gates / instruments
- `scripts/fr13_apc_lossless_ab.sh` — same-boot cache-miss vs cache-hit byte compare + TTFT.
- `scripts/fr13_apc_prefill_after_hit.sh` — the prefill-after-hit reproduction (the carrier; ~10 min).
- `FR13_APC_STATE_PROBE` — the conv/ssm state-checksum isolation probe.

Cross-refs: [[apc_ssm_carrier_FIXED]] (partial), [[why_option1_snapshot_side_failed]],
[[sglang_mamba_radix_cache_design]].

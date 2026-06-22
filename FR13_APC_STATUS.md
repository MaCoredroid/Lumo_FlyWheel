# FR13 APC (Automatic Prefix Caching) — evidence record (2026-06-22)

This is an EVIDENCE RECORD, not a ship declaration. Numbers are recorded here because
`output/` is gitignored and would otherwise be lost. Read it with the caveats stated —
several figures are CROSS-BOOT N=1 and the tree fix is IN-FLIGHT (not yet proven).

Branch `fr13-prefix-cache`. Patcher = `scripts/fr10_phase4_patch_vllm_tree_gdn.py`.
Launcher = `scripts/fr13_launch_forked_fa2_tree_server.sh`.

---

## PROVEN: spine (native MTP-5) + APC is lossless AND fast

**Lossless.** `nativeapc_spine` config — `enable_prefix_caching=True`, mamba `align` mode,
`block_size=1024`, SSM cache dtype `float32`, FLASH_ATTN backend, cuda-graph capture,
`num_spec=5` — SOLVES `astropy-12907` (504-byte patch, task resolved). The produced patch is
identical to the cache-OFF run. So on the SPINE (linear MTP-5, no off-spine tree branches),
turning APC on does not change the answer. This is the linear/spine path that vLLM `align`
mode keeps one block-boundary checkpoint for; the spine's committed recurrent state lands
exactly on that boundary, so the cache-hit restore is correct.

**Fast** (astropy-12907, both arms solve, per-task `elapsed_s`). CAVEAT: **CROSS-BOOT N=1** —
the two arms ran in different boots with different agent trajectories, so these are indicative,
not a clean A/B. A SAME-BOOT spine A/B is still pending for the headline speed claim.

| metric | cache-ON | cache-OFF |
|---|---|---|
| wall (s) | 1463 | 1921 |
| prefix-reuse | 92.8% | 0% |
| decode tokens | 24638 | 22946 |
| decode-tokens / 1800s | ~30.3k | ~21.5k (~1.4x) |
| conversation-tokens-serviced / wall | 1704 | 357 (~4.8x) |

**Honest framing of the speed numbers:**
- The ~4.8x "conversation-tokens-serviced/wall" number COUNTS cache-reused tokens and the two
  arms had different agent trajectories — it is NOT a clean speedup, do not quote it as one.
- The real, repeatable mechanism is the **per-turn TTFT win of 2.0-3.9x** (prefill ~5.5s ->
  ~2.5s, banked separately). That win COMPOUNDS across many agentic turns.
- End-to-end TASK throughput is only ~1.4x because of the offload architecture: codex runs on
  the alienware box, so most of the wall-clock is NOT GB10 prefill — APC can only accelerate
  the GB10-prefill fraction. The clean headline number is pending a SAME-BOOT spine A/B.

---

## SGLang disciplines we ported

Source: `research/fr13_workflows/sglang_mamba_radix_cache_design.md` (SGLang
`mem_cache/mamba_radix_cache.py`, `mamba_checkpoint_pool.py`, `Int8CheckpointStore`) and the
launcher `scripts/fr13_launch_forked_fa2_tree_server.sh`.

| SGLang discipline | PORTS to vLLM align? | shared spine+tree? | how it lands on our path |
|---|---|---|---|
| fp32 working SSM state | YES | shared | `--mamba-ssm-cache-dtype float32` (computation accuracy; vLLM #26807) |
| block-aligned snapshots (page-aligned in SGLang) | YES | shared | `--mamba-block-size` multiple of 8, the boundary align caches at |
| #45238 `max-num-batched-tokens == block_size` | YES | shared | one block boundary crossed per chunked-prefill step (avoids silent 0-hit trap) |
| #43650 BLOCK_ALIGN backport | YES | shared | `find_longest_cache_hit` off-by-one-block fix |
| verify-the-conv-window-don't-reconstruct | YES | shared | re-prove `conv1d_out row-0 = 0.0` under whatever window vLLM reconstructs |
| int8 quantized cached store (`Int8CheckpointStore`) | NO | n/a | vLLM align has no separate quantized checkpoint pool — capacity trick, no hook |
| per-radix-node checkpoint tree | NO | n/a | vLLM align keeps ONE checkpoint at the last block boundary |
| ping-pong DONATE-don't-copy active slot | NO | n/a | SGLang-internal slot architecture; vLLM uses a single block pool |
| dual-LRU (separate mamba + KV budgets) | NO | n/a | vLLM align has one block pool, no analog |

---

## DIAGNOSIS: the tree gap is the ONE SGLang piece that does not port

vLLM `align` keeps exactly ONE checkpoint, at the block boundary.

- **Spine (linear MTP-5):** the committed recurrent state IS at that block boundary -> align
  restores the right row on a cache-hit -> correct. (Proven above: 12907 solves, identical
  to cache-OFF.)
- **Tree (e.g. cat6root):** the tree committer writes the per-NODE accepted state into the
  node-bank (one row per accepted tree node). On a cache-hit, vLLM align restores the
  BLOCK-ALIGNED row, which is NOT the committed accepted-leaf row -> the restored state is
  STALE -> recurrent poison -> the agent emits an EMPTY patch.

**Confirmed by experiment:**
- spine + APC solves 12907;
- tree + APC empties 12907 (reproduced under BOTH cuda-graph AND eager — so it is not a
  capture artifact);
- cache-OFF solves BOTH spine and tree.

`FR13_APC_CONV_FIX=1` handles the CONV axis of this (tree-node conv restore in preprocess,
stock invertible snapshot in postprocess). The **SSM (recurrent state) axis is the remaining
defect** — align still restores the block-aligned SSM row, not the committed accepted-leaf row.

---

## IN-FLIGHT (NOT yet proven): tree fix `FR13_APC_VERBATIM`

Ports SGLang's "snapshot the committed state, restore it VERBATIM, never reconstruct"
discipline onto vLLM align. At commit, copy the committed accepted-leaf conv+SSM state INTO
the exact block-aligned row that align reads on a hit
(`block_ids[aligned_new_computed // block_size - 1]`, at the post-commit seqlen). So when
align "restores the block-aligned row," that row already holds the correct committed state.

- Flag default `0`; **byte-neutral when off** (early-return -> leaf map never created ->
  override never fires -> byte-identical to pre-fix path).
- **EAGER-only for now** — the publisher does a GPU->CPU sync that is not yet cuda-graph-safe.
- **Confirmed FIRING**: 48 `did=True` DIAG lines observed.
- **Validation on astropy-12907 is IN PROGRESS** (not yet a verdict).
- Next steps if it solves 12907 like the spine does: (1) make the publisher cuda-graph-safe;
  (2) 4-task SWE gate; (3) same-boot `tokens/1800s` table for the clean speed headline.

---

## STATUS of flags

**PROVEN-shared (KEEP, default-on under APC):**
- `FR13_ENABLE_APC` (master)
- base disciplines: `--enable-prefix-caching`, `--enable-chunked-prefill`,
  `--mamba-block-size`, `--mamba-ssm-cache-dtype float32`, `--max-num-batched-tokens`
- `FR13_APC_BLOCK_ALIGN_45477` (real block-align backport)
- `FR13_APC_CONV_FIX` (conv axis of the tree fix; default 1)
- `FR13_APC_CACHE_AB` (the correct cache-ON vs cache-OFF instrument)

**IN-FLIGHT (flag-gated, default off):**
- `FR13_APC_VERBATIM` — the SSM-axis tree fix above. Validation in progress.

**DEAD (default-off cruft, cleanup pending — all confounded/refuted prior carrier theories):**
- `FR13_APC_VALUE_VS_ORACLE` (+`_LOG`/`_FH`), `FR13_APC_CACHEHIT_VALUE_PROBE`,
  `FR13_APC_ALIGN_TREE_AWARE`, `FR13_APC_COMMIT_SITE_WT`, `FR13_APC_SSM_WRITE_THROUGH`
  (+`_SSM_SNAPSHOT_SUB`), `FR13_APC_HIT_RECURRENT_SUFFIX` (+`_HIT_SUFFIX_CAP`),
  `FR13_APC_MROPE_TAIL_ZERO`, `FR13_APC_DROP_FINAL_BLOCK`, `FR13_APC_POS_PROBE`,
  `FR13_APC_STATE_PROBE`, `FR13_APC_GRAPH_REPLAY_BARRIER`, `FR13_APC_INDEX_RERESOLVE`,
  the dead SSM/APC `*_DIAG` family.

**CLEANUP CONSTRAINT (important):** `FR13_APC_VERBATIM` REUSES the publisher
`_fr13_publish_apc_ssm_leaf` and the `_FR13_APC_SSM_LEAF_BY_REQ` leaf-map (the decouple agent
made these fire under VERBATIM, including the `FR13_EAGER_PACK` committer guard). The eventual
dead-code cleanup MUST KEEP the publisher + leaf-map; only the WRONG-ROW writers may be
removed (the `get_temporal_copy_spec` `SSM_SNAPSHOT` redirect, `SSM_WRITE_THROUGH`,
`COMMIT_SITE_WT`, `ALIGN_TREE_AWARE`). See `research/fr13_workflows/apc_flag_cleanup_plan.md`.

---

## Pointers
- SGLang design reference: `research/fr13_workflows/sglang_mamba_radix_cache_design.md`
- Flag cleanup + sequencing plan: `research/fr13_workflows/apc_flag_cleanup_plan.md`
- Launcher (flag wiring): `scripts/fr13_launch_forked_fa2_tree_server.sh`
- Patcher (publisher + leaf-map + VERBATIM): `scripts/fr10_phase4_patch_vllm_tree_gdn.py`
  (`_fr13_publish_apc_ssm_leaf` ~L7572, `_FR13_APC_SSM_LEAF_BY_REQ`)

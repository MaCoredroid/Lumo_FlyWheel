# FR13 num_splits static check + native same-seed floor

Date: 2026-06-09

## Scope

This binds the two cheap cost-gate items after the TREE batch-invariant backend blocker:

- Static read of the forked-FA2 tree decode `num_splits` path.
- Same-seed native B=4 floor probe, seed `1313` vs `1313`, with `GPU_UTIL=0.82`.

No fallback TREE batch-invariant run is bound here. GDN scan was not re-investigated.

## num_splits static check

The FR13 tree decode replacement at `scripts/fr13_patch_fa2_tree_bias.py:570` calls `flash_attn_varlen_func(..., fa_version=2, tree_bias=tree_bias)` without an explicit `num_splits`, so the Python wrapper default is `0`. The broader flash-attn backend replacement at `scripts/fr13_patch_fa2_tree_bias.py:651-706` preserves `num_splits=attn_metadata.max_num_splits`; for FA2 the wrapper rejects `num_splits > 1`.

The C++ FA2 varlen path only calls `set_params_splitkv(...)` under `seqlenq_ngroups_swapped`, which requires `max_seqlen_q == 1`. Tree verify uses paged KV with `max_seqlen_q = tree_len > 1`, so that branch is not taken and `params.num_splits` remains `0`. Paged KV still forces the splitkv dispatch, but the kernel template treats `params.num_splits <= 1` identically: grid uses batch/head axes directly, `Split=false`, and no combine kernel runs.

Conclusion: the proposed `num_splits=1` probe is inert by construction for this tree decode shape. It would not change the attention reduction carrier, so no TREE GPU boot was run for this item.

Evidence:

- `scripts/fr13_patch_fa2_tree_bias.py:570` tree decode replacement.
- `/tmp/vllm_live_019/vllm/vllm_flash_attn/flash_attn_interface.py:298-323` FA2 wrapper rejects `>1` and passes `num_splits`.
- `.../vllm-flash-attn-src/csrc/flash_attn/flash_api.cpp:632-749` `set_params_splitkv` is gated by `max_seqlen_q == 1`.
- `.../vllm-flash-attn-src/csrc/flash_attn/src/flash_fwd_launch_template.h:107-136` `0` and `1` both select the non-split path with no combine.

## Native same-seed floor

Run root: `output/fr13_native_same_seed_floor_082_20260609T234548Z`

Both arms used the same probe config:

- Launcher: `scripts/fr13_launch_forked_fa2_tree_server.sh`
- Backend/mode: `FLASH_ATTN` / `naive_mtp`
- `FR13_FA2_PREFILL_NATIVE=1`
- B=4, `MAX_NUM_SEQS=4`, `max_tokens=64`
- `temperature=0.6`, `top_p=0.95`, seed `1313`
- `GPU_UTIL=0.82`
- One container at a time, host memory recovered between arms.

Regime proof:

- Logs show `attention_backend: FLASH_ATTN`.
- Logs show `gpu_memory_utilization: 0.82`.
- Logs show `Using FlashAttention version 2`.
- Logs show `enforce_eager=False`.
- Logs show FULL CUDA graph capture completed for B=4.

Arm A summary:

- accept/event: `3.203125`
- accept/token: `0.3559027777777778`
- accepted/draft tokens: `205/576`
- draft events: `64`
- returned tokens: `256`
- wall seconds: `13.280279874801636`
- returned tokens / wall second: `19.276702178976013`
- request TPS mean: `5.166110644237123`
- warm decode TPS: `9.834429859774906`

Arm B summary:

- accept/event: `3.125`
- accept/token: `0.3472222222222222`
- accepted/draft tokens: `200/576`
- draft events: `64`
- returned tokens: `256`
- wall seconds: `13.201187133789062`
- returned tokens / wall second: `19.392195368911626`
- request TPS mean: `5.17061963786305`
- warm decode TPS: `9.9476437823192`

Same-seed comparison:

- Artifact: `output/fr13_native_same_seed_floor_082_20260609T234548Z/fr13_native_same_seed_floor_compare.json`
- Exact records: `1/4`
- Mismatched records: `3/4`
- Compared positions: `256`
- Mask/token-mismatch positions: `139/256`
- Bag-TV: `0.11328125`
- First diffs:
  - prompt `0`, sample `0`, pos `11`: A `26622`, B `12182`
  - prompt `1`, sample `0`, pos `15`: A `5759`, B `1970`
  - prompt `2`, sample `0`, pos `25`: A `44675`, B `13766`

Conclusion: same-seed native B=4 is not deterministic under this deployed probe shape. The seed-robust native floor for this pair is approximately bag-TV `0.1133` with `139/256` raw positional token mismatches.

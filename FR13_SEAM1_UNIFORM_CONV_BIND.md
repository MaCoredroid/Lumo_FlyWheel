# FR13 Seam 1 uniform conv write-back validation

Date: 2026-06-09
Run: `output/fr13_decisive_seam1_20260609T173330Z`

## Scope

Seam 1 asked whether the earlier layer-conditioned rolled-tail conv band-aid still existed and whether the tree conv write-back is uniformly native-tail clean. The layer-conditioned band-aid is already absent from HEAD:

- no `_fr10_use_rolled_tail_prior`
- no `layer_idx >= 4` / `int(_fr13_layer_idx) >= 4`
- no `rolled_tail_remapped`

The live code uses the uniform accepted-path write-back:

- read slot: accepted length clamped to `accepted_len - 1`
- write-back source: `cat(prior_window, node_path_x)`
- per-node store index: `node_path_len + arange(conv_state_len)`
- write-back target: every speculative node row via `conv_state.index_copy_`

This stage only normalized the cast ordering in the accepted-length clamp so the wiring guard checks the accepted-path contract directly. It is not a new tail-read or layer-conditioned conv behavior change.

## CPU Gates

Passed:

```text
pytest -q tests/test_fr10_phase4_sampled_committer_wiring.py tests/test_fr10_tree_commit_gates.py tests/test_fr10_tree_conv.py
22 passed, 1 skipped

python3 -m py_compile scripts/fr10_phase4_patch_vllm_tree_gdn.py scripts/fr13_gdn_subop_diff.py scripts/fr13_gdn_subop_table.py scripts/fr12_compare_gdn_subkernel_spine.py
```

## Live Substate Gate

Capture shape:

- Tree arm: `TREE_ATTN`, forked FA2, `tree_mtp`, eager, `MAX_MODEL_LEN=65536`, `MAX_NUM_SEQS=1`, `FR12_SUBKERNEL_CAPTURE_NUM_TOKENS=10`.
- Native arm: `FLASH_ATTN`, `naive_mtp`, eager, `MAX_MODEL_LEN=65536`, `MAX_NUM_SEQS=1`, `FR12_SUBKERNEL_CAPTURE_NUM_TOKENS=6`.
- GDN layers captured: all 48 GDN layers (`0,1,2,4,5,6,...,62`).
- Captures: `144` tree and `144` native files = `48` layers x `3` verify calls.
- Capture filter excludes prefill/profile passes; all tree captures have `num_tokens=10`, all native captures have `num_tokens=6`, and `conv1d_out` is present.

Reducer artifact:
`output/fr13_decisive_seam1_20260609T173330Z/fr13_seam1_uniform_conv_validation.json`

Summary:

- Compared layer/call pairs: `144`.
- Row0 clean-input cases: `99`.
- Row0 clean-input cases with nonzero `conv1d_out`: `0`.
- Spine clean-input cases with nonzero `conv1d_out`: `0`.
- `all_row0_clean_input_conv1d_out_zero=true`.
- `all_spine_clean_input_conv1d_out_zero=true`.

## Verdict

Seam 1 is validated. The fresh gate did not expose a new clean-input conv divergence, so there is no further conv write-back work to do before Seam 2. Remaining divergences in non-clean rows are upstream input/branch-state issues, not a conv read/write-back seam.

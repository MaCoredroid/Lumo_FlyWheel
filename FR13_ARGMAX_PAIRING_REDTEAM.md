# FR13 argmax pairing red-team - prompt-0 guarded localization

Run: `output/fr13_argmax_lcp_prompt0_20260609T052640Z`

Commit under test: `e9d2a701` plus measurement-tool worktree changes.

## Pairing guard

- Tree and native `prompt_token_ids` are byte-identical.
- Compared rows: `1`.
- Prompt token count: tree `15`, native `15`.
- This invalidates the earlier `lcp=0` / token-0 artifact as a capture-pairing bug, not a model result.

## Served-token result

- Served outputs share an exact prefix of `8` tokens.
- First served mismatch: position `8`.
- Tree token: `727`.
- Native token: `1005`.
- First-token TV: `0.0`.
- Prompt-0 single-sample bag-TV: `0.375`.

This is consistent with the prior e2e binding's "first paired mismatch at position 8" shape and contradicts the rejected token-0 divergence.

## Authoritative argmax flip

Reducer: `scripts/fr13_argmax_lcp_localize.py`

- First target-argmax stream flip: stream position `7`, completion position `8`.
- Tree location: `tree_path_lcp_max` call `2`, row `0`, local emitted index `0`, token `727`.
- Native location: `native_final_logits.call2.pt`, row `0`, local emitted index `0`, token `1005`.
- Tree event had `accepted_len=0`; native event had `accepted_len=5`.

Layer localization for that exact flip row:

- `input_hidden` max_abs: `0.0`.
- First nonzero layer: layer `0`, `linear_attention`, max_abs `0.0625`.
- Final norm max_abs: `9.03125`.

Interpretation: the first valid guarded flip is not full-attention. It is already introduced by the layer-0 GDN/linear-attention path on the third verify event after the earlier accepted path. The next root is the layer-0 recurrent-state/current-event GDN handling for call 2 row 0, not prompt pairing and not the rejected lcp=0 token-0 result.

## num_warps=8 red-team gate

Script: `scripts/fr13_gdn_scan_warp_gate.py`

Artifact: `output/fr13_numwarps8_gdn_scan_gate_20260609T043954Z.json`

- `root_npad1` output vs native max_abs: `0.0`.
- `tree_npad16` output vs native max_abs: `0.0`.
- State deltas were tiny post-output (`3.7e-9` / `1.49e-8`), but the authoritative raw output gate is bit-exact.

Verdict: no live regression from `num_warps=8` at `N_PAD=1` or deployed `N_PAD=16`; do not revert it based on the red-team gate.

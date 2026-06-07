# FR13 GDN Conv Offline Replay Findings

Date: 2026-06-07
Commit under test: cbaff1b0
Capture: `output/fr13_gdn_conv_multilayer_capture_20260607T193044Z`

## Setup

- TREE arm: `ATTENTION_BACKEND=TREE_ATTN`, `FR13_FA2_TREE_BIAS=1`, `FR10_ALLOW_LINEAR_FALLBACK=0`, eager, `GPU_UTIL=0.4`.
- Native arm: `ATTENTION_BACKEND=FLASH_ATTN`, `FR10_DECODE_MODE_DEFAULT=naive_mtp`, `FR10_ALLOW_LINEAR_FALLBACK=0`, eager, `GPU_UTIL=0.4`.
- Capture layers: GDN `0`, `24`, `45`, `62`.
- Captured rows: spine path plus first branch row (`selected_nodes=[0,1,2,4,6,8,3]`).
- Replay output: `conv_replay_multilayer.json`.

## Superseded Result

This multilayer result is contaminated for kernel tuning. Layers 24, 45, and
62 have nonzero `pre_conv_path0`, so their `conv1d_out` diffs include upstream
tree/native drift and must not be used to tune causal-conv arithmetic. The
replay tool now supports `--require-clean-input` to exclude these layers from
aggregate variant selection.

## Original Result

Offline replay did not produce a valid conv fix. The failure is not isolated to the causal-conv arithmetic: for layers 24, 45, and 62 the tree and native spine rows already differ at `pre_conv_path0` and `window_path0`, before the manual conv op is applied.

| layer | pre_conv max_abs | window max_abs | bf16 tap max_abs | conv1d_out max_abs | best replay variant |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 0.0 | 0.0 | 0.0 | 0.0 | bf16 taps, 0123, torch silu, torch bf16 |
| 24 | 0.15234375 | 0.15234375 | 0.0546875 | 0.0234375 | bf16 taps, 0123, torch silu, torch bf16 |
| 45 | 0.2421875 | 0.2421875 | 0.078125 | 0.046875 | bf16 taps, 0123, torch silu, torch bf16 |
| 62 | 0.375 | 0.375 | 0.125 | 0.125 | bf16 taps, 0123, torch silu, tie_all_down |

Aggregate best replay variant:

```text
tap_product=bf16, order=0123, silu=torch, store=torch_bf16
max_abs=0.125, mean_abs=0.000930948939640075, nonzero=149912
```

## Interpretation

The cached native `causal_conv1d_update` PTX uses bf16 tap multiplies, converts tap products to fp32, accumulates in 0->1->2->3 order, applies an `ex2.approx` sigmoid path, and stores bf16. The current replay matches layer 0 exactly under that shape. The deeper failures cannot be fixed by changing silu rounding or tap dtype in the manual conv, because the operands entering conv are already different.

No live full-ladder run was performed after this replay, because the offline prerequisite `conv1d_out=0.0` for every captured layer was not met.

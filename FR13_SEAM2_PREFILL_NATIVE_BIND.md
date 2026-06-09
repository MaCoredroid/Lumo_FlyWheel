# FR13 Seam 2 Prefill-Native Binding

Date: 2026-06-09

Commit target: FR13 decisive test Seam 2.

## Scope

Seam 2 is validation of the already-committed `FR13_FA2_PREFILL_NATIVE`
route in `scripts/fr13_patch_fa2_tree_bias.py`, not a new kernel fix.
The launcher default remains `FR13_FA2_PREFILL_NATIVE=1`.

## Artifacts

Run root: `output/fr13_decisive_seam2_20260609T175913Z`

- Extended gate-2 stock-vs-fork no-bias compare:
  `gate2/no_bias_compare.json`
- Live tree L7 prefill full-attn capture:
  `tree_prefill_l7_live/logs/full_attn_tree_l7.call0.pt`
- Live native L7 prefill full-attn capture:
  `native_prefill_live/logs/full_attn_native_l7.call0.pt`
- L7 replay reducer:
  `prefill_full_attn_l7_replay.json`
- Live tree all-layer prefill-GDN capture:
  `tree_prefill_live/logs/prefill_gdn_tree.call*.pt`
- Live native L8 prefill-GDN capture:
  `native_prefill_live/logs/prefill_gdn_native.call0.pt`
- L8 replay reducer:
  `prefill_gdn_l8_state_replay.json`

Both live arms used the same text prompt:

`Write a Python function to reverse a linked list, then explain it.`

Both completions reported `prompt_tokens=14`.

## Results

Extended gate-2 passed for all no-bias stock-vs-fork cases:

| case | torch_equal | max_abs | nonzero |
| --- | ---: | ---: | ---: |
| `float16_decode` | true | 0.0 | 0 |
| `float16_prefill` | true | 0.0 | 0 |
| `bfloat16_decode` | true | 0.0 | 0 |
| `bfloat16_prefill` | true | 0.0 | 0 |

Live L7 prefill full-attn replay:

| stage | max_abs | nonzero |
| --- | ---: | ---: |
| `input_hidden` | 0.0 | 0 |
| `qkv_proj` | 0.0 | 0 |
| `q_after_rope` | 0.0 | 0 |
| `k_after_rope` | 0.0 | 0 |
| `attn_out_raw` | 0.0 | 0 |
| `o_proj_out` | 0.0 | 0 |

Live L8 prefill-GDN replay:

| stage | max_abs | nonzero |
| --- | ---: | ---: |
| `pre_conv` | 0.0 | 0 |
| `conv_out` | 0.0 | 0 |
| `initial_state` | 0.0 | 0 |
| `core_out` | 0.0 | 0 |
| `final_state` | 0.0 | 0 |

## Verdict

Seam 2 is validated for HEAD: forked FA2 no-bias decode and prefill are
byte-identical to pristine stock FA2, live TREE_ATTN prefill L7
`attn_out_raw` is byte-identical to native FLASH_ATTN, and the downstream
GDN L8 prefill recurrent seed is byte-identical.

Next front is the FR13 decisive all-8 branch-oracle plus self-noise-corrected
B=4 superset e2e comparator. The known served-token branch flips are a
comparator/oracle front, not a new kernel-fix target.

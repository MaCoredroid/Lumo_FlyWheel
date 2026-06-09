# FR13 Conv Prior Slot Fix Bind

Date: 2026-06-09
Fix commit: `3a9039cc`
Run: `output/fr13_conv_slot_fix_prompt0_20260609T063933Z`

## Patch

Changed the tree GDN conv prior read in
`scripts/fr10_phase4_patch_vllm_tree_gdn.py` so the compact bank read uses the
last committed accepted slot (`accepted_len - 1`) instead of the next slot
(`accepted_len`).

This aligns conv with the already-clean h0 read convention and fixes the root
row/slot selection. It does not introduce a tail-column band-aid.

## Substate Gate

Fresh paired prompt0 capture with call0..2 L0 subkernel taps:

| Call | input_hidden | pre_conv | conv1d_out | h0_state_in | gdn_scan_out | gate_out | o_proj_out |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1 | 0.0 | 0.0 | 0.0 | 3.725e-09 | 0.0 | 0.0 | 0.0 |
| 2 | 0.0 | 0.0 | 0.0 | 9.537e-07 | 0.0 | 0.0 | 0.0 |

The previous failing metric was call2 row0 `conv1d_out=18.375036`; after the
fix it is `0.0`.

Conv-detail alignment at call2:

- Tree prior bank rows: `[[5]]`
- Tree read cols: `[[4]]`
- Tree compact prior cols: `[0, 1, 2]`
- Native prior bank row: `1`
- Native rolled-tail prior cols: `[5, 6, 7]`
- `prior_window_max_abs=0.0`
- `window_row0_max_abs=0.0`

Substate summary artifact:
`output/fr13_conv_slot_fix_prompt0_20260609T063933Z/fr13_conv_substate_compare.json`

## E2E Gate

Reducer:
`scripts/fr13_e2e_measure.py --skip-capture`

Result:

- `valid=true`
- prompt pairing: byte-identical
- `first_mismatch=None`
- `bag_tv=0.0`
- native accept/event: `2.6`
- tree accept/event: `2.8`
- native TPS: `10.320005524789147`
- tree TPS: `6.149089367549517`
- `first_flip_layer=None`

The argmax localizer still reports a tail event at completion position 15 where
the native stream has no paired row left; the deliverable comparison is the
authoritative e2e gate here and reports no token mismatch plus zero bag-TV.

E2E artifact:
`output/fr13_conv_slot_fix_prompt0_20260609T063933Z/fr13_e2e_measure.json`

## Clean Per-Forward Profile

Profile surface:

- `scripts/fr10_quick_decode_tps_probe.py`
- B=4, CUDA graphs enabled, `FR10_METRICS=0`
- prompt0, `max_tokens=64`, `temperature=0.6`, `top_p=0.95`
- `GPU_UTIL=0.86` because `0.88` tripped vLLM startup free-memory guard

Profile artifact:
`output/fr13_conv_slot_fix_prompt0_20260609T063933Z/profile/fr13_profile_summary.json`

Per-request decode-forward timing from vLLM `/metrics`:

| Arm | Nodes | Decode iters | Decode seconds sum | ms / request-forward | Accept/event |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tree | 9 | 20 | 21.937230 | 274.215 | 2.173 |
| Native | 6 | 15 | 11.654068 | 194.234 | 3.714 |

Measured tree/native per-forward ratio: `1.412x`.

Attribution from the supported non-nsys surface:

- Node count is the measurable dominant factor: `9 / 6 = 1.50x`, close to the
  measured `1.412x`.
- Residual versus native scaled by node count is `-17.136 ms`, so this surface
  does not show a positive extra penalty beyond node-count scaling.
- GDN tree-scan state traffic / num_warps spill and forked-FA2 whole-tree cost
  are not separately isolatable from the `/metrics` profile alone; separating
  them still requires a component ablation or a valid server-side kernel trace.

Do not use returned-token TPS alone for this profile: native returned 256 tokens
while tree returned 163 under stochastic sampling, so per-forward decode timing
is the comparable profile metric.

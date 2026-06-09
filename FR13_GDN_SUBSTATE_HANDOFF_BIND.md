# FR13 GDN Substate Handoff Bind

Date: 2026-06-09
Base commit: `a3c5806a`
Run: `output/fr13_gdn_substate_prompt0_20260609T061732Z`

## Scope

One paired GPU capture on guarded prompt0, using the FR13 orchestrator guard and existing
GDN subkernel taps. No server re-pairing or extra GPU iteration was done after the capture.

Captured artifacts:

- Tree: `tree/logs/gdn_l0_subkernel.call0.pt` through `call2.pt`
- Native: `native/logs/gdn_l0_subkernel.call0.pt` through `call2.pt`
- Paired reducer: `fr13_e2e_measure.json`

## Pairing / E2E Guard

The orchestrator prompt guard passed:

- `prompt_pairing=byte-identical`
- prompt ids: `[7734, 264, 12654, 709, 310, 9637, 264, 82546, 10278, 1103, 11, 1179, 10033, 424, 13]`
- first completion mismatch: position 8, native token `1005`, tree token `727`
- first argmax flip: `tree_call=2/tree_row=0` vs `native_call=2/native_row=0`
- first flip layer: layer 0 `linear_attention`, `hidden_max_abs=0.0625`
- final norm row0 at call2: `9.40625`
- bag-TV: `0.375`
- native accept/event: `2.6`
- tree accept/event: `2.1666666666666665`
- native TPS: `10.257913894991908`
- tree TPS: `4.577720760093666`

## Seed Row L0 Sub-Op Comparison

For row0, calls 0 and 1 are bit-exact through the layer0 GDN subkernel. Call2 is the
first captured early-GDN divergence.

| Call | input_hidden | pre_conv | conv1d_out | h0_state_in | gdn_scan_out | gate_out | o_proj_out |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1 | 0.0 | 0.0 | 0.0 | 3.725e-09 | 0.0 | 0.0 | 0.0 |
| 2 | 0.0 | 0.0 | 18.375036 | 9.537e-07 | 0.453125 | 0.290283 | 0.1875 |

Interpretation: the call2 row0 divergence is present before scan/gate, at conv-state
read/use. It is not caused by divergent input embeddings, QKV projection, or h0
recurrent state entering the scan.

## Conv-State Handoff Pin

Call2 conv detail is the decisive mismatch:

- Tree prior bank row: `[[6]]`
- Tree read col: `[[5]]`
- Tree prior cols used after compact-head read: `[0, 1, 2]`
- Native prior bank row: `1`
- Native prior cols used from clean pre-update ref: `[5, 6, 7]`
- Tree/native source indices for row0 are identical: `[0, 1, 2, 3]`
- `pre_conv_row0` max diff: `0.0`
- `prior_window` max diff: `58.234375`
- `window_row0` max diff: `58.234375`
- `tap_bf16_row0` max diff: `31.15625`

The native call2 prior window is present inside the tree candidate bank, but not at
the row/cols read by the live tree path:

- `candidate_conv_state_post_remap` bank 5 cols `[0, 1, 2]` matches native prior
  window exactly.
- Live tree reads bank 6 cols `[0, 1, 2]`, which mismatches native by `58.234375`.

## Code Site

The alignable bug is at the tree prior conv-state read selection in
`scripts/fr10_phase4_patch_vllm_tree_gdn.py:797-818`:

- `_fr10_conv_read_cols` is derived directly from `_fr10_accepted_lens_tensor`.
- The live call2 accepted path has accepted length 5, which gathers
  `spec_state_indices_tensor[..., 5] == 6`.
- The native-equivalent post-remap state for this call is at bank 5, not bank 6.

The likely patch point is the non-native-prior-read branch that computes
`_fr10_conv_read_cols`; for accepted_len > 0 the compact tree bank appears to need
the previous accepted slot (`accepted_len - 1`) rather than the next slot. This was
not changed in this commit because the turn allowed one GPU boot and this note binds
the capture-backed diagnosis only.

## Verdict

Exact sub-op pinned: cross-call conv-state prior row/slot selection at the tree GDN
handoff. The scan, gate, h0 recurrent input, and input embedding are downstream or
clean for the seed row at the first flip.

Next validation target: patch the compact prior read column/row selection and re-run
the same guarded prompt0 orchestrator capture plus substate comparison.

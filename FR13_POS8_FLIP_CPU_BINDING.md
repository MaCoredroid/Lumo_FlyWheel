# FR13 pos-8 flip CPU binding

Run: `output/fr13_argmax_lcp_prompt0_20260609T052640Z`

Reducer: `scripts/fr13_pos8_flip_cpu_characterize.py`

Artifact: `output/fr13_argmax_lcp_prompt0_20260609T052640Z/pos8_flip_cpu_characterization.json`

## Guarded flip

The prompt pairing remains valid: this analysis uses the guarded prompt-0 capture whose tree/native prompt token IDs are byte-identical.

The first **authoritative comparable emitted-stream** argmax flip is:

- Stream position `7`, completion position `8`.
- Tree: call `2`, row `0`, token `727`.
- Native: call `2`, row `0`, token `1005`.

This is the same position-8 miss from the e2e binding. The earlier rejected token-0/lcp-0 result remains invalid.

## Raw row-index caveat

A raw same-index table finds an earlier call-0 row-2 mismatch:

- Tree parent target `12305`.
- Native row-2 argmax `198`.

That is **not** bound as the first path flip because raw tree branch row indices are not necessarily native sequential-path comparable. The bound result is the emitted-stream flip above.

## Layer and floor characterization

For the bound call-2 row-0 flip:

- `input_hidden max_abs = 0.0`.
- First nonzero model layer: layer `0`, `linear_attention`, `max_abs = 0.0625`.
- Final norm max_abs: `9.03125`.

Native logits at call `2`, row `0`:

- Native argmax token `1005`: logit `29.375`.
- Tree-chosen token `727` in native logits: `22.625`.
- Native argmax minus tree token: `6.75`.
- Native top1 minus top2: `4.875`.

This is not a tiny tie flip in the native reference. From the existing captures it classifies as an above-floor model-path divergence at/through layer 0 for the first comparable stream flip, not a within-floor one-argmax wobble.

## Non-monotone and downstream-path evidence

Row-0 first nonzero by call is non-monotone:

- call `0`: first nonzero layer `10`, max_abs `0.005126953125`.
- call `1`: first nonzero layer `9`, max_abs `0.00152587890625`.
- call `2`: first nonzero layer `0`, max_abs `0.0625`.

At call `2`, four of six same-index rows already have divergent inputs: rows `[1, 2, 3, 5]`. That supports the adversarial finding that call-2 row-0 is downstream of earlier path/acceptance divergence. It does **not** support a monotone GDN recurrent-state writeback chase.

## Capture inventory

Checked 20 `.pt` files in the prompt-0 run. The existing captures contain model-level `input_hidden`, per-layer hidden/residual, final norm, native logits, and tree LCP target IDs.

They do **not** contain GDN substate tensors for `h_recurrent`, `h0`, `conv`, or `conv_state`.

No GPU boot was performed for this binding. The missing substates mean the exact layer-0 GDN subcomponent cause is not determined from this artifact; a future targeted substate capture would be needed only if the next front requires splitting layer-0 GDN into carried recurrent state versus conv/current-event arithmetic.

## Verdict

- Pos-8 flip localization: **valid**.
- Cause: **undetermined**.
- Not supported: **GDN recurrent-state writeback root cause**.
- Current characterization: **above-floor layer-0 model-path divergence for the first comparable stream flip**, with exact GDN substate cause not captured.

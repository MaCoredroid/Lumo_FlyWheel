# FR13 Eager B4 Bisection Bind

Date: 2026-06-09

Run root: `output/fr13_b4_eager_bisect_20260609T203718Z`

## Question

Does the B=4 corruption gate reproduce under eager execution, or is the 47.5% captured-run loss a CUDA-graph-only bug?

## Regime

Three one-GPU sequential arms, `MAX_NUM_SEQS=4`, `MAX_MODEL_LEN=131072`, `FR10_METRICS=0`, `max_tokens=64`, `temperature=0.6`, `top_p=0.95`, same four SWE prompts.

- TREE: `TREE_ATTN/tree_mtp`, seed `1313`, `ENFORCE_EAGER=1`.
- Native E5: `FLASH_ATTN/naive_mtp`, seed `1313`, `ENFORCE_EAGER=1`.
- Native-noise: `FLASH_ATTN/naive_mtp`, seed `2313`, `ENFORCE_EAGER=1`.

Eager proof files:

- `tree/eager_proof.txt`
- `native/eager_proof.txt`
- `native_noise/eager_proof.txt`

All three logs contain `Enforce eager set, disabling torch.compile and CUDAGraphs` and `Cudagraph is disabled under eager mode`; no CUDA graphs were used for this bisection.

Prompt-token identity passed. Prompt token counts matched across arms: `681`, `1080`, `829`, `1614`.

## Result

Comparator: `scripts/fr13_corruption_gate.py`

Output: `fr13_eager_b4_corruption_gate.json`, exit code `2`, `valid=true`, `passed=false`, `verdict=FAIL`.

- Self-noise mask: `137` positions; native self bag-TV `0.15234375`.
- Compared positions: `256`.
- Self-noise eligible positions: `119`.
- Outside-self-noise losses: `44/119 = 0.3697478992`.
- Raw argmax matches: `75/256 = 0.29296875`.
- First real loss: prompt `1`, position `11`, tree token `12182`, native token `26622`.
- Depth collapse: prompt `2`, run ending at position `26`, run length `6`.
- Longest real-loss run: `31`.
- Bag-TV: tree/native emitted-token bag-TV `0.19140625` > budget `0.15234375`.
- Accept/event: tree `1.8021978022` vs native `3.1875`, delta `-1.3853021978`.

Per-prompt outside-self-noise losses:

- prompt `0`: `0/11`
- prompt `1`: `5/16`
- prompt `2`: `7/28`
- prompt `3`: `32/64`

## Bisection Verdict

The B=4 real-loss reproduces under eager execution. The captured-run failure is not CUDA-graph-only.

The eager rate is lower than the previous captured bind (`36.97%` here vs `47.51%` captured), and the first real-loss row differs because this quick bisection used one sample per prompt, but the failure class remains large, self-noise-corrected, and fail-closed.

Localization should proceed as an eager B=4 co-residency / batched-verify bug. An eager substate hook is not a false-negative route for this bisection question.

## Capture-Hook Caveat

A direct captured-regime substate run was attempted first and failed before serving:

- `output/fr13_b4_verify_localize_20260609T202608Z/tree/docker_failed_layer_capture.log`: layer-hidden capture specialized dynamic prefill shape via `int(hidden_states.shape[0])`, causing a Dynamo `ConstraintViolationError`.
- `output/fr13_b4_verify_localize_20260609T202608Z/tree_gdn/docker_failed_gdn_capture.log`: GDN subkernel capture hit `torch._dynamo.exc.Unsupported: torch.cuda.is_current_stream_capturing`.

Those failures explain why the bisection was necessary before trusting eager substate localization.


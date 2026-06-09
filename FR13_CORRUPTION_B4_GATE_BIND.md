# FR13 Corruption B4 Gate Bind

Date: 2026-06-09
Worker: codex_fr13

## Scope

Executed the user-approved three-arm corruption gate on the SWE-Verified B=4 CUDA-captured diagnostic prompts:

- TREE: forked FA2 `TREE_ATTN`, `tree_mtp`, `MAX_NUM_SEQS=4`, `FR10_METRICS=0`, seed `1313`.
- Native E5: `FLASH_ATTN`, `naive_mtp`, `MAX_NUM_SEQS=4`, `FR10_METRICS=0`, seed `1313`.
- Native-noise: `FLASH_ATTN`, `naive_mtp`, `MAX_NUM_SEQS=4`, `FR10_METRICS=0`, seed `2313`.

Run root: `output/fr13_corruption_b4_gate_20260609T194841Z`.

All three arms were run sequentially on one GPU with `recover_host_memory()` between arms. All three logs confirm `enforce_eager=False`, CUDA graph memory profile `FULL=4`, and `Capturing CUDA graphs (decode, FULL): 4/4`.

## Pairing

The gate prompt-token identity guard passed:

- tree/native prompt token IDs byte-identical: `true`
- prompt token counts: prompt0 `681`, prompt1 `1080`, prompt2 `829`, prompt3 `1614`
- records per arm: `16`

## Gate Result

Reducer: `scripts/fr13_corruption_gate.py`

Output: `output/fr13_corruption_b4_gate_20260609T194841Z/fr13_corruption_gate.json`

Exit code: `2`

Result:

- `valid=true`
- `passed=false`
- `verdict=FAIL`

Self-noise baseline:

- native self-noise mask positions: `277`
- native self bag-TV: `0.10986328125`

Self-noise-corrected token surface:

- compared positions: `467`
- self-noise eligible positions: `221`
- raw token/argmax match rate: `126/467 = 0.2698`
- outside-self-noise real losses: `105/221 = 0.4751`
- first outside-self-noise real loss: prompt `0`, position `16`, tree token `369`, native token `3051`
- longest outside-self-noise loss run: `44`
- depth-collapse detector fired at prompt `0`, run end position `62`, run length `6`

Per-prompt outside-self-noise losses:

- prompt0: `46/101 = 0.4554`
- prompt1: `0/21 = 0.0`
- prompt2: `6/27 = 0.2222`
- prompt3: `53/72 = 0.7361`

Bag-TV:

- tree vs native emitted-token bag-TV: `0.2335064444`
- budget: `max(E5 floor 0.0593, native-self 0.10986328125) = 0.10986328125`
- result: above budget

Accept/event:

- tree: `2.1016042781`
- native: `3.6191536748`
- delta: `-1.5175493968`
- native-noise: `3.8139534884`

## Depth-0 Root Signal

CPU reducer output: `output/fr13_corruption_b4_gate_20260609T194841Z/fr13_depth0_root_gate.json`

Measured tree events from summary: `561`; rows with a step-0 trace in the measured tail: `549`.

- depth-0/root accept count: `348/549`
- depth-0/root accept rate: `0.6338797814`
- depth-0/root reject count: `201/549`
- `target_argmax != draft0`: `201`

This is the depth-0 contamination signature the user asked to test: the tree root has one candidate, yet the tree verifier rejects that root draft in `36.6%` of measured step-0 rows. The three-arm gate shows those rejects coexist with outside-self-noise served-token losses and the accept/event drop versus native.

Limitation: this run did not emit a native `per_req_spec_trace.jsonl`, so the reducer could not do a per-event native target-argmax comparison for each root row. The served-token, self-noise-corrected gate is valid and paired; the per-event native trace remains absent.

## Conclusion

The decisive three-arm test does not support the "drafter-quality" stop. The topology is present, pairing is clean, and native self-noise does not explain the tree output losses. This is a real tree-verify contamination / target-row bug surface, with a strong depth-0 root-reject signature.

Speed/forward-cost analysis remains deferred per user instruction.

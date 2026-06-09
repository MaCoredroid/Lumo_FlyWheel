# FR13 B4 Nondeterminism + Native Batch-Invariant Diagnostic

Date: 2026-06-09

Commit: this bind

Run root: `output/fr13_b4_nondet_bi_native_20260609T220457Z`

## Purpose

After `e9f9267c`, the direct batch-invariant TREE gate is blocked because vLLM rejects
`VLLM_BATCH_INVARIANT=1` with `TREE_ATTN`. This diagnostic pivots to:

1. confirm whether the same deployed B=4 TREE config is run-to-run nondeterministic;
2. test whether native `naive_mtp + FLASH_ATTN` self-noise drops under
   `VLLM_BATCH_INVARIANT=1`;
3. avoid speed claims and avoid assigning the carrier to GDN-scan vs TREE_ATTN without
   an isolating measurement.

## Arms

All arms used one GPU sequentially, B=4, `MAX_NUM_SEQS=4`, seed-controlled probes, and
host-memory recovery between boots.

- TREE repeat: `TREE_ATTN/tree_mtp`, seed `1313`, `FR10_METRICS=0`,
  `VLLM_BATCH_INVARIANT=0`, CUDA graph captured, output
  `tree_repeat/tree_swe4_probe.json`.
- Native BI: `FLASH_ATTN/naive_mtp`, seed `1313`, `VLLM_BATCH_INVARIANT=1`,
  `LUMO_BATCH_INVARIANT_VLLM=1`, output `native_bi/native_swe4_probe.json`.
- Native BI noise: `FLASH_ATTN/naive_mtp`, seed `2313`, `VLLM_BATCH_INVARIANT=1`,
  `LUMO_BATCH_INVARIANT_VLLM=1`, output
  `native_bi_noise/native_noise_swe4_probe.json`.

Comparison JSON:
`output/fr13_b4_nondet_bi_native_20260609T220457Z/fr13_b4_nondet_native_bi_compare.json`.

## Result A - TREE Nondeterminism Confirmed

Compared prior captured no-BI TREE run
`output/fr13_corruption_b4_gate_20260609T194841Z/tree/tree_greedy_probe.json`
against the fresh same-config TREE repeat on overlapping `(prompt_id, sample_index)`
records.

- Overlap: `4` records.
- Exact records: `0/4`.
- Token positions compared: `256`.
- Token mismatches: `189/256`.
- First diffs moved:
  - prompt0/sample0: pos `16`, prior `369`, repeat `5759`;
  - prompt1/sample0: pos `11`, prior `26622`, repeat `12182`;
  - prompt2/sample0: pos `21`, prior `1970`, repeat `3425`;
  - prompt3/sample0: pos `11`, prior `12182`, repeat `26622`.

Verdict: same seed/config TREE output is not stable run-to-run. The failing rows/tokens
move, matching the batch-invariance/race signature and explaining why fixed-row
substate localization can miss.

## Result B - Native Batch-Invariant Self-Noise

Native no-BI eager baseline from `output/fr13_b4_eager_bisect_20260609T203718Z`:

- Overlap: `4` records.
- Compared positions: `256`.
- Self-noise mask positions: `137`.
- Bag-TV: `0.15234375`.

Native BI fresh pair:

- Overlap: `4` records.
- Compared positions: `256`.
- Self-noise mask positions: `139`.
- Bag-TV: `0.0859375`.

Captured no-BI reference from the earlier 16-record three-arm gate:

- Compared positions: `2048`.
- Self-noise mask positions: `1319`.
- Bag-TV: `0.10986328125`.

Verdict: batch-invariant native lowers the emitted-token bag-TV in this B=4 seed-pair
comparison, but it does not reduce the raw positional self-noise mask (`137 -> 139`).
This does not support binding the TREE excess solely to shared fp8 GEMM
batch-dependence from this diagnostic.

## Result C - Carrier Not Isolated

The TREE nondeterminism is confirmed, but this run does not distinguish GDN tree-scan
from forked `TREE_ATTN` as the tree-specific carrier. Because `VLLM_BATCH_INVARIANT=1`
cannot boot with `TREE_ATTN`, the direct lossless-drop gate remains blocked. A
GDN-vs-TREE_ATTN carrier split requires a separate isolating measurement.

## Bound Conclusion

- Direct TREE run-to-run nondeterminism: **confirmed**.
- Native BI self-noise raw-mask drop: **not observed** in this B=4 seed-pair
  measurement; bag-TV improved but positional flips did not.
- Tree-specific carrier: **unresolved**; do not assign to GDN-scan or TREE_ATTN yet.
- Speed remains deferred.

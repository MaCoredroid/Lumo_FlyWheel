# Sampling correctness of LumoTree's deployed tree sampler (2026-10-02)

**Deployment under test:** `hydra27_fixed32` with the deployed sampling settings:

| Setting | Value |
|---|---|
| temperature | 0.6 |
| top_p | 0.95 |
| top_k | 20 |
| min_p | 0 |
| presence_penalty | 1.0 |
| per-request seed | none (server seed 0) |
| draft_probs | None (target-only acceptance) |

**Code path, found by reading the source.** vLLM's `RejectionSampler.apply_logits_processors` applies penalties to the 31 target rows and the 31 self rows. The rows then go through temperature and top-k/top-p, and the fixed32 tree accept walk runs on the result. The deployed walk route is `fixed32_pytorch_exact_float_triton_integer_commit` with `bulk_device_generator`. Its float math is the `_fr13_fixed32_taw_execute_torch` oracle, which the CUDA path matches byte for byte.

## 1. Penalty history: the defect, confirmed

vLLM's stock `_combine_outputs_with_spec_tokens` assumes the drafts form a chain. It gives spec row j the history output + spec[:j], where spec[:j] is the draft tokens in flat node order. For a tree, the correct history for a row is output + the tokens on that row's own root-to-node path:
- for a target row, the path to the parent;
- for a self row, the path to the node, including the node itself.

**Capture.** A record-only runtime wrapper (`scripts/v2exp/pendiag/`) recorded the deployed vehicle on 4 corpus requests. It covered the first 40 tree steps, 2,480 target and self rows in all. For each row it recorded the token set whose logits the processors changed.

| Prediction | Rows matched |
|---|---|
| Flattened-chain history | **2,480 / 2,480** |
| Correct per-path history | 63 / 2,480 (only rows where the two coincide) |

With presence_penalty = 1.0, tokens from siblings and other branches are penalised by 1.0 (1.67 after dividing by the 0.6 temperature) at every decision past the root. A leaf's self row also misses its own token.

**Impact.** Per tree node, the deviation is measured as the total variation between the row the sampler used and the correct autoregressive row. Both are built from the captured top-256 pre-processing logits. The used row applies the observed penalties, /T, Triton exact-k top-k, then top-p. The correct row applies per-path penalties, /T, tie-keeping top-k, then top-p. Over 1,120 node-steps:

| Decision depth | Node-steps | Share with TV > 0.01 | TV, 90th percentile | TV, max |
|---|---|---|---|---|
| root | 40 | 0.00 | 0.000 | 0.000 |
| depth 1 | 120 | 0.25 | 0.259 | 0.418 |
| depth 2 | 200 | 0.23 | 0.201 | 0.408 |
| depth 3 | 200 | 0.28 | 0.216 | 0.421 |
| depth 4 | 160 | 0.17 | 0.114 | 0.410 |
| depth 5 | 160 | 0.23 | 0.267 | 0.403 |
| depths 6–11 | 240 | 0.10–0.28 | 0.04–0.20 | 0.09–0.38 |

- **The first token emitted per step is exact.** About 21% of deeper node decisions are distorted.
- **Total distortion per step:** weighted by how often the walk visits each node, it is 0.00–0.20 (6 steps, 200,000 walks each).

## 2. The accept/correction walk itself

**Monte Carlo test** (`pendiag/taw_mc.py`, `pendiag/analyze_pen.py`):
- **Walks:** the deployed walk ran 200,000 times on each of the first 6 captured steps.
- **Vocabulary:** compressed to the rows' supports, keeping token order, which leaves the walk's distribution unchanged.
- **Statistic:** a per-node G-test of the token emitted at each node against the exact per-node distribution (target row of the first child, or the self row for a leaf).
- **Impossible tokens:** any emitted token with zero probability rejects outright. The first version mishandled this; a planted fault exposed it, and it was fixed before the results below.
- **Multiple testing:** Bonferroni correction at α = 10⁻³ per step.

| Comparison | Result |
|---|---|
| Walk vs the rows it received | **0 rejections in 54 node tests**: the walk is lossless on its inputs |
| Walk vs the correct per-path rows | rejected in 5 of 6 steps (the 6th had no distorted node); per-node TV up to 0.41 |

**Fault sensitivity**, shown on the same data before the walk was scored:

| Planted fault | Steps detected (of 6) |
|---|---|
| double temperature | 6 |
| target rows shifted by one | 6 |
| fixed residual draw | 5 |
| always accept | 3 |
| correlated source and accept uniforms | 1 |

Every planted fault is detected in at least one step. The randomness faults matter only at nodes where acceptance is uncertain, which explains their lower per-step rates.

## Conclusion

The tree accept/correction walk is exact for the probability rows it receives. Under the deployed `presence_penalty=1.0` those rows are wrong past the root: vLLM's chain-history penalty construction is applied to a tree. LumoTree's deployed output distribution therefore departs from autoregressive sampling after the first token of each step.

Exactness holds only for history-independent sampling settings (presence, frequency and repetition penalties all off, together with min_p = 0, which the rejection path ignores). In that regime the processed rows equal the autoregressive rows. Separately, ties at the top-k boundary differ between the Triton exact-k path the tree uses and the single-row PyTorch tie-keeping path.

The fix is to build per-path penalty histories for tree rows from the tree's parent table. Until then, any "lossless" claim must be restricted to that penalty-free regime, and the deployed results described as using a non-exact penalty application.

The RNG for the walk is a single per-process generator seeded 0, so LumoTree boots with identical request order replay the same uniforms. Distribution protocol v3 treated its 3 tree boots as independent, and they were not.

# Attention-provenance correction red-team — 2026-09-24

**PASS.** The current wording correctly identifies patched attention implementations, distinguishes the recorded patched Triton qualification path from the engaged patched FA2 agent path, and does not assign kernels by precision format. The one transient selected-route scope regression identified below is fixed. No numerical result or qualification boundary changed. This was a read-only source/receipt review; no inference or manuscript edit was performed.

## Independent implementation checks

I read `experiments/out-20260922T213142Z-e8-single-logits-v2/qualification_on/loaded_backend/tree_attn.py` directly and rehashed it to `a6d4ae3000f248ec2b9fd37546a1f90b96145c3078e5a98e696b5bda49a1af97`. Line 30 imports `unified_attention` from `vllm.v1.attention.ops.triton_unified_attention`; the `TREE_ATTN` selector is at line 725. The active forward calls that function at lines 1132 and 1153, supplying the decode ancestry bias and flat block table at 1165 and 1167. The forked FA2 diagnostic helper instead returns at line 86 unless `FR13_FA2_MAB=1`; mere presence of its FA2 operator call does not establish active serving through FA2.

The saved qualification Docker command executes `fr10_phase4_patch_vllm_tree_gdn.py` and selects `TREE_ATTN`; it does not execute the separate `fr13_patch_fa2_tree_bias.py`. The frozen primary patcher hashes to `a7f8f0943d1a13ac4742e824ea4e126c6ff341342aba677db5169e712c7e7281`. Its live Triton patch function begins at 33303, pins launch metadata at 33416, changes softmax to exp2 at 33459, adjusts tree-bias masking at 33517, and is invoked at 44339. Its “FA2-style” softmax comment describes an algorithmic transformation, not a call to the forked FA2 operator. The upstream container base does not make the code executing after these patches stock.

I independently rehashed all eight E8 run copies of `loaded_backend/tree_attn.py` (two qualification arms and six timing cells), plus the timing campaign's two copied qualification receipts: all ten files have the same `a6d4ae...` hash. This is source identity, not an inference from a backend-name string.

For the agent route, I independently read the Cqc10 `logs/fr13_fa2_qrow32_b1_production_engagement.json`, SHA `c8f3c761e9cd3ee97a1c3c421f28488154ae13df2504bb42aa026966dad1da71`. It reports `ENGAGED`, `gqa_pair_splitk`, four splits, `candidate_served=true`, `fallback_allowed=false`, FA2 head `29210221863736a08f71a866459e368ad1ac4a95`, and candidate binary `28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857`. These are distinct positive FA2 engagement receipts.

The separately reviewed provenance correction note supplies the E2 B1/B4 matching loaded-module hashes and representative E7 command/patch receipts. It explicitly distinguishes older command/patch evidence from E8-style loaded-module snapshots; this review does not upgrade older E7 captures to newly recovered identical kernel bytes. The run-specific naming is warranted without transferring results across attention implementations.

## Wording and scope

The eight prior `stock` attention labels in `main.tex:57,120,204,221–222,231,243,274` are corrected. There are no remaining occurrences of `stock` in current `main.tex`. The design evidence map, baseline row for Cat10 qualification, and README agree: patched Triton unified attention selected through `TREE_ATTN` for the recorded qualification route; patched FA2 for the engaged agent deployment. `main.tex:57` and the supporting notes explicitly reject a precision-wide FP8/NVFP4 restriction.

During review, the draft of `main.tex:243` changed “selected Cat10 route” to “recorded Cat10 jobs,” which would have attributed accepted-path remapping and synchronous execution to earlier diagnostics that did not enable the remap flag. That sentence now reads “The selected Cat10 continuation-qualification route uses patched Triton unified attention through TREE_ATTN...” The narrower subject preserves the original scope and agrees with the explicit earlier-diagnostic limitation at `main.tex:298`. This issue is **closed**.

I compared current `main.tex` against the previously sealed `artifacts/paper-patch-producing-20260924T223358Z.tar.gz`, whose embedded prior main hash is `baee91032667889733aed5307b0d3a40fc0cc6a880df8df8aa17d8782b70066d`. The only changed source ranges are those eight attention descriptions. No empirical numeric values, sample counts, tolerances, pass/fail outcomes, or continuation-qualification limits changed. The abstract, workload case study, patch-producing rate audit, and unfiltered competitive audit are byte-identical to their files in that package. No extra experiment is necessary to correct these names.

## Reviewed source hashes

Paths are relative to paper v2.

| File | SHA-256 |
|---|---|
| `main.tex` | `6aab2a64f703b166e81a843b1bb6a408f3c66ebce97fa741e59cdceeb22fb8a6` |
| `notes/design/optimization-evidence-map.md` | `241b99d4827a365f196d157607414402bfac56da2ab85dbabdc114eaa73f5080` |
| `notes/design/baselines.csv` | `75eae46782cc678a750af590ddd0c97db1d454d32413c1808183c087b41c7fe8` |
| `README.md` | `15e84d30ce8ea3193d9ce20df495f259088ba65a22996eba2278fe4e59a66b38` |
| `notes/attention-backend-provenance-correction-2026-09-24.md` | `ffff898c6e0f0a7cdfb8aa7b46b8f49d64ea5c59e2c657c37a2691144c875179` |
| `abstract.tex` | `f160cd35a25f331d625bed688d7875372ca27234178727fb77a6bf23aeeb4b3b` |
| `results/agent-workload/case-study.tex` | `73b37d657faa261fba11eaed22bb914ec1fc61229a0b22280b204a5d07ac4b64` |

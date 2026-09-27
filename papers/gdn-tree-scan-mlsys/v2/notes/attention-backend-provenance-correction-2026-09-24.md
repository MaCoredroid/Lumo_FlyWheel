# Attention backend provenance correction — 2026-09-24

**Finding:** “stock TREE_ATTN” is inaccurate. The recorded FP8 qualification uses patched attention code. However, its saved active forward calls the **patched Triton unified-attention implementation**, not the separate forked FA2 operator. The NVFP4 agent runs do have the forked FA2 engagement receipts. Both are patched implementations; do not infer identical kernels from the `TREE_ATTN` selector or from the generic phrase “patched FlashAttention.” No experiment is needed to correct this naming.

Recommended wording:

> Our deployments use patched attention implementations. The NVFP4 agent runs engage the forked FlashAttention-2 path. The recorded FP8 qualification selects `TREE_ATTN` and executes our patched Triton unified-attention path with flat candidate slots. Numerical qualification applies to the recorded implementation and settings.

If “FlashAttention” is intended as the algorithm family rather than the FA2 library/operator, say that explicitly; it must not imply the FP8 capture used the same forked FA2 implementation as the NVFP4 run.

## Decisive loaded-source evidence

Let `Q` denote `experiments/out-20260922T213142Z-e8-single-logits-v2/qualification_on/`, relative to v2.

- [`Q/loaded_backend/tree_attn.py`](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/papers/gdn-tree-scan-mlsys/v2/experiments/out-20260922T213142Z-e8-single-logits-v2/qualification_on/loaded_backend/tree_attn.py:30) imports `unified_attention` from `vllm.v1.attention.ops.triton_unified_attention` at line30. `TreeAttentionBackend.get_name` returns `TREE_ATTN` at line725. The active `TreeAttentionImpl.forward` calls `unified_attention` for prefill at1132 and decode at1153; decode supplies ancestry `qq_bias` at1165 and the flat block table at1167. A capture hook follows at1173.
- This module contains a `flash_attn_varlen_func` call at226, but it belongs to `_fr13_fa2_mab_recall`, a diagnostic helper that immediately returns unless `FR13_FA2_MAB=1` (line86). Its presence is not evidence of active FA2 serving. The saved E8 command does not set that flag or invoke the FA2 replacement patcher.
- Both E8 qualification arms and all six timing cells have the identical loaded `tree_attn.py` hash `a6d4ae3000f248ec2b9fd37546a1f90b96145c3078e5a98e696b5bda49a1af97` (eight independently rehashed files). The qualification manifest's `unchanged_loaded_modules.tree_attn.py` pins this same hash. Thus the difference is not merely a selector-name guess.
- `Q/docker_logs.txt:3` reports `tree_attn.py=True`, `triton_unified_attention.py=True`, and `flash_attn.py=True` in the patcher's changed-file receipt. At43, the language-model backend selection is `TREE_ATTN`. The `FLASH_ATTN` lines39–40 concern ViT/MMEncoder attention, and do not identify the language-model tree decode.
- The exact frozen patcher is `experiments/e8-single-logits-qualification-v2/frozen/repository/scripts/fr10_phase4_patch_vllm_tree_gdn.py`, SHA `a7f8f0943d1a13ac4742e824ea4e126c6ff341342aba677db5169e712c7e7281`. `_patch_triton_unified_attention_fr13` begins at33303 and explicitly patches the live kernel. Its applied changes include the pinned Triton launch metadata at33416, exp2 softmax at33459 and tree-bias masking at33517; line44339 invokes it. The explanatory comment describes FA2-style log2 softmax, which is not the same thing as invoking the forked FA2 operator. `Q/docker_inspect.json` confirms the command executes this patcher before serving with `--attention-backend TREE_ATTN`; it does not execute `fr13_patch_fa2_tree_bias.py`.

## E2/E7 consistency and provenance limits

The selected-route archive `artifacts/selected-route-20260922T0930Z.tar.gz.manifest.json` binds the B1 and B4 `loaded_backend/tree_attn.py` files to the same `a6d4ae...` hash. I also rehashed those two original remote files read-only, with the same result:

- `experiments/out-20260922T082815Z-e2-b1-policyB-sync-p072/e7b_001_p072_arm1_none-B_fs_ieee-all/loaded_backend/tree_attn.py`
- `experiments/out-20260922T084319Z-e2-b4-policyB-sync-cohort4/e7b_001_p072_arm1_none-B_fs_ieee-all/loaded_backend/tree_attn.py`

Their boot receipts report all three attention Python files patched. The earlier independent B1 report records loaded `triton_unified_attention.py` hash `5dd83ac4fbd0a08054a5b7188f4937cdf33fce854b96d0a38e616a9e4c7789c0`; this is a previously captured hash, not a newly recovered kernel-source copy in this pass.

For representative E7 capture runs, I read the original saved command and patch receipt and cross-checked the log bytes against the local immutable archives. Each command invokes `fr10_phase4_patch_vllm_tree_gdn.py`, selects `TREE_ATTN`, and reports the attention files patched. These older roots do not carry the E8-style loaded-module snapshot, so retain that distinction in provenance rather than claiming a newly verified identical kernel hash:

| Run suffix under `experiments/` | Log selector line | `docker_logs.txt` SHA-256 |
|---|---:|---|
|`out-20260922T002242Z-e7a-step2-captures-v4/capture_01_p072`|43|`8ca1dd617851f7d9b884739f3d90214e7d1aae7cd933b36f31cd1b503385f8e5`|
|`out-20260922T063019Z-e7b-r13-p072-batch/e7b_001_p072_arm1_none-B_fs_ieee-all`|44|`decf107ff5c1f8e35e988bf2cff64edc2c50a9fc645553c900b95e9a68815a75`|
|`out-20260922T071945Z-e7b-r16-p072-commit-batch/e7b_001_p072_arm1_none-B_fs_ieee-all`|44|`44302705a8b8b168d6ee5eec135388db5a5768f88cfbb41462ac831448ebb0c2`|

The flat-slot/read-address and remap findings remain meaningful; removing “stock” does not erase their observed implementation contract. Likewise this correction does not transfer the numerical results automatically to the different FA2 kernel or NVFP4 settings.

## NVFP4 evidence and genuine “stock” scope

`results/agent-workload/raw/cqc10/hydra27_fixed32_promoab_Cqc10/logs/fr13_fa2_qrow32_b1_production_engagement.json`, SHA `c8f3c761e9cd3ee97a1c3c421f28488154ae13df2504bb42aa026966dad1da71`, reports `status=ENGAGED`, `arm=gqa_pair_splitk`, `num_splits=4`, `candidate_served=true`, `fallback_allowed=false`, 16 attention layers, FA2 head `29210221863736a08f71a866459e368ad1ac4a95`, and candidate binary SHA `28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857`. The Sr12/Cqc16/Cqc15 receipts are preserved in the shared-task engagement projection SHA `cd3de566e6f14d6dcd98414f240a17ec04bb497f4b16a70858efcc6b1108fae4`. These support the forked FA2 label for those agent deployments.

The pinned upstream container **base image** is genuine; it is patched before serving. The upstream enum/API names remain genuine names. Neither fact makes the executing backend pristine upstream code. A specifically unchanged helper/reference computation can be called upstream only where its own source comparison proves that. There is no support here for a broad “stock attention qualification” or “unmodified native serving stack” label. The receipt phrase “stock FA2 set_params_splitkv” identifies its scratch-allocation helper, not an unpatched whole attention kernel.

## Paper statements requiring correction

Reviewed source hashes: `main.tex` `baee91032667889733aed5307b0d3a40fc0cc6a880df8df8aa17d8782b70066d`; `abstract.tex` `f160cd35a25f331d625bed688d7875372ca27234178727fb77a6bf23aeeb4b3b`; `results/agent-workload/case-study.tex` `73b37d657faa261fba11eaed22bb914ec1fc61229a0b22280b204a5d07ac4b64`.

All eight offending passages are in `main.tex`: line57 (“FP8 ... stock TREE_ATTN”);120 (“stock TREE_ATTN qualification route”);204 (“stock-attention qualification”);221 (table “Qualified stock-attention route”);222 (table contrast against “stock TREE_ATTN”);231 (“qualified stock ... backend”);243 (numerical methods “uses stock TREE_ATTN”);274 (pilot description “with stock TREE_ATTN”). Replace with the appropriate patched flat-slot Triton qualification label, define it once, and keep the actual FA2 implementation distinction. No corresponding stock-attention claim was found in the reviewed abstract or case-study input. Older audit notes such as `notes/optimization-route-audit-2026-09-22.md:14,46,52` contain the old shorthand; preserve their history and point to this dated correction rather than rewriting immutable run evidence.

E8 primary receipts: `Q/docker_logs.txt` SHA `0529fe53528649aeed236cafc8c15e19ff7484c7afc3c262a1017b66d50242d7`; `Q/docker_inspect.json` SHA `5252cdca27c3f8659f259b1787b6d56cbb713428aa3c3165a6cb8249ab6fd459`. No source, manuscript, attempted experiment, or raw evidence was changed by this audit.

## Precision does not determine the attention implementation

The forked FA2 launcher also has an explicit FP8 serving lineage. At commit `de0d338ce2a3904efbee1b9f62544e60ba7f3b78` (the parent of the NVFP4 repoint commit `4bdf4dd14`), `scripts/fr13_launch_forked_fa2_tree_server.sh` has SHA `6ffd285412479d91b4a2b98cfcad8e8b09f7dec82482e25e6cacf692ef6c59c2`: line571 defaults `FR13_FA2_TREE_BIAS=1`, line6482 invokes `fr13_patch_fa2_tree_bias.py`, and line6741 serves `/models/qwen3.6-27b-fp8`. Thus forked FA2 is not an NVFP4-only implementation. The current launcher's lines595–604 explicitly pin the NVFP4 checkpoint as a deployment policy and disallow caller overrides; do not claim the current script is freely precision-switchable. The paper should say **“these recorded FP8 qualification runs”** and **“these NVFP4 agent runs”**, rather than generalizing a precision-to-backend mapping. This historical source check introduces no historical performance number.

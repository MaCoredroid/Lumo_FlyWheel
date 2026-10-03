# Native aligned A/B: bounded source diagnosis and fixed-tactic successor seam

2026-09-29. Read-only CPU/source inspection; no CUDA query, model execution, author-code import, source mutation, or new experiment. This diagnoses the two retained `q1-native-corpus-calibration-20260929T045821Z-aligned_nonpacked-{A,B}` runs. It does not replace their failed cross-process categorical check or alter the remaining frozen packed controls.

**Verdict:** the retained evidence narrows the earliest divergence to layer-0 input projection or its inputs/load/capture. Per-boot FlashInfer FP8/FP4 autotuning is an observed, source-supported dispatch candidate with an available explicit tactic replay interface. Winning tactics and loaded parameter digests were not retained, so neither different tactics nor a loading defect is established as the cause. There is no justified criteria relaxation.

## What the existing evidence resolves

- Parent/raw reviewer reports 168 valid captures per process, exact within-process repeats, 14/84 cross-process greedy mismatches, and all 84 output/logit comparisons different. This note uses that accepted raw audit instead of rerunning all tensors.
- The preserved source/config projection shows the same immutable image (`sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`), runtime patches and FA2 binary, model/config identity, seed 0, BF16 activation route, ModelOpt mixed precision, TP/PP/DP=1, max-sequences 1, aligned cache, FP32 SSM, nonpacked GDN decode, Triton GDN prefill, APC, chunking and graph settings. A/B identity/time fields differ as expected; no numerical route difference was found.
- Independently compared all 168 recorded `consumed_trace` arrays, removing only the global `seq` event counter: **zero differences**. Root-short r0 has 13 chunks of 1024 and a final 175-token chunk; its repeat has the final 175-token chunk. This is recorded scheduling evidence, not proof of every runtime tensor value.
- In the root-short-r0 case, all 48 GDN layers' conv/SSM hashes differ. Earliest layer-0 conv hashes: A `f840d4fb6b595a9e598980176bcb805fac9b9221b91f128e1b30d78fd543e50c`, B `fa545f40d5b67b1eb37f86d5c859b863ce380b80633cbb0c1708160f36a9fb45`. Layer-0 conv shape is `[3,10240]` BF16. Layer-0 SSM also differs; first actual attention layer 3's retained KV bytes differ with matching recorded maps. These are metadata-digest comparisons tied to the raw auditor's validity checks.

### Why layer-0 conv matters

Let `N = experiments/review-response-20260927/identity/native_source/` (all paths here are relative to v2). `N/vllm__model_executor__layers__mamba__gdn_linear_attn.py:534–560` computes `in_proj_qkvz` and splits its Q/K/V values before recurrence. `N/vllm__model_executor__layers__mamba__ops__causal_conv1d.py:197–236` directly copies the final `state_len` raw projection rows into conv state when `state_len <= seqlen`. Here 3 <= 175. Thus a correctly captured differing final layer-0 conv cache cannot be explained solely by convolution reduction order, GDN initial SSM, later FA2 attention, the LM head, or sampling. Remaining immediate candidates are embedding/input normalization, projection weights/scales or quantized GEMM dispatch, or an erroneous capture/address binding. The native GDN prefill additionally zeroes initial state for `~has_initial_state` at lines 984–986; no source evidence of uninitialized bootstrap was found.

The model config explicitly assigns `model.language_model.layers.0.linear_attn.in_proj_qkv` and `in_proj_z` to static **FP8**, not FP4. The retained `qwen3_5.py:463` packed mapping combines them into `in_proj_qkvz`; `modelopt.py:2134–2148,2175–2183` resolves the shared FP8 method. Therefore investigate the **FP8** projection at the first divergence, while retaining FP4 tactic evidence for downstream layers/head. Calling this only an “NVFP4 GEMM problem” is too narrow.

## Exact image source and observed autotuning

`S = p0/monitor/review-response-20260927/native-ab-source-inspection-20260929/`. Twelve small source files were copied unchanged from parent-created **stopped/nonstarted** inspection container `70cbbf3512551e4b3bb9b5c22453b7a23f3c51f6c725a59e76bfb3ab0f9b600a`, using `docker cp` stdout tar. No container start/import occurred. `S/MANIFEST.json` records every original `/usr/local/lib/python3.12/dist-packages/...` path, size and SHA. All 12 copies were rehashed successfully. Parent was notified that source copying is complete and may remove its exact owned inspection container.

The image embeds FlashInfer **0.6.8.post1**, git `8a49f9a242695fed9173cd5b19da45c9bd316503` (`S/flashinfer/_build_meta.py`).

| Exact source under S | Relevant contract |
| --- | --- |
| `vllm/model_executor/warmup/kernel_warmup.py:81–109` | `fi_utils.autotune()` at line 93 has **no cache argument**; profiles through a max-batched-token dummy forward. |
| `vllm/v1/worker/gpu_worker.py:578–590` | Compile warmups precede kernel autotuning; graph capture follows it. A replay policy must be installed before model profiling/warmup/capture, not patched after captured kernels exist. |
| `vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py:65–78`; `vllm/utils/flashinfer.py:518–528,688–711` | FP8 linear goes through `flashinfer.bmm_fp8(..., backend="auto")` with per-tensor FP32 scales and FP8 input/weight operands. |
| `flashinfer/gemm/gemm_base.py:5159–5187,5199–5287,919–948` | FP8 auto dispatch supplies suitable CUTLASS/cuBLAS/cuDNN runners to `AutoTuner.choose_one("fp8_gemm", ...)`. Actual available/selected runner is not attested in A/B. |
| `vllm/model_executor/kernels/linear/nvfp4/flashinfer.py:43–88`; `flashinfer/gemm/gemm_base.py:951–991,5093–5100` | FP4 uses CUTLASS backend with explicit selected tactic; runner enumerates tactic IDs and passes one into the compiled GEMM. |
| `flashinfer/autotuner.py:745–849` | Enumerates valid tactics, profiles them, picks the least measured time, and stores the winner in process-local `profiling_cache`. This is timing-sensitive selection, not evidence of randomized search or proven numerical inequality. |
| `flashinfer/autotuner.py:406–472,1173–1280,1282–1355` | Explicit JSON cache load/save API; metadata validation; no automatic persistent winner file when vLLM calls bare `autotune()`. |

A and B engine logs independently show tuning start at line 93, FP8/FP4 profiling in lines 95–126, and tuning end at 127 before capture. The retained projections carry original full-log hashes. No saved winning tactic JSON is in the reviewed run receipts, and the launch mounts do not bind a shared runtime cache directory. The compiled-code cache path alone does not establish equal GEMM tactics. Default source behavior therefore explains how different boot choices *could* persist throughout exact within-process repeats; it does not prove they did.

## Loading alternative: unresolved, not demonstrated

Retained `modelopt.py:474–499` allocates FP8 weight storage with `torch.empty`, initializes scale sentinels, then `:510–520` normalizes/requantizes merged scales and weights. NVFP4 allocation/postprocessing is at `:1118–1207`. Exact image `S/vllm/model_executor/model_loader/default_loader.py:380–395` runs strict missing-parameter validation only when `model_config.quantization is None`. Thus clean boot logs and identical checkpoint hashes are insufficient to prove equal effective loaded quantized tensors. This is a missing observation, **not** a finding that weights were missing. The common LM-head scalar reshape patch cannot alone cause layer-0 conv divergence.

## Profiling input alias refinement

`S/flashinfer/autotuner.py:1150–1171` generates new random tensors only when a profile tensor has a dynamic dimension; otherwise it appends the original tensor reference. `gemm_base.py:811–827` makes FP8 activation/output dimensions dynamic but leaves weight and scalar activation/weight scales static. FP4's 128x4 profile at `:4885–4905` makes activation, activation-block-scale and output dimensions dynamic; weight, weight-block-scale, global alpha and workspace stay referenced. `_prepare_input_tensors_with_batches:1357–1389` returns the same input list unless cold-L2 mode is enabled; the default is false (`autotuner.py:287`). Even when enabled, the first batch still aliases originals.

Thus autotuning is **not isolated from live weight/scale storage by a universal copy**. However, source-level aliasing is not proof of corruption: reviewed FP8 runners (`gemm_base.py:113–138,606–759,775–801,2644–2671`) pass weights/scales as GEMM inputs; their explicit Python stores affect outputs/new padding. The selected FP4 CUTLASS wrapper (`:955–991`) passes aliased weights/scales to the compiled kernel without an explicit Python in-place write. No reviewed source demonstrates mutation of live weights. Before/after-tuning digests of the same effective tensors are the smallest observation distinguishing a mutation from tactic selection; compiled-kernel behavior remains unmeasured here. Random profiling tensors also consume RNG, but no evidence connects that consumption to the deterministic projection mismatch, and greedy decoding does not make RNG a sufficient explanation.

## Minimum prospective route diagnostic and safe replay contract

Complete the unchanged frozen control set first. Any successor is separately identified, prospectively gated and must keep all model/precision/deployed settings and the categorical next-draft criterion. No extra GPU activity was performed or authorized by this review.

1. **Add discriminating receipts, not another broad sweep.** Bind the exact effective embedding, layer-0 norm and merged input-projection weights/scales after postprocessing and again after tuning; record the relevant normalized-input/quantized-input/projection digests for the existing first root-short capture. Export the effective FP8/FP4 runner+tactic table after tuning and before CUDA graph capture, with exact image/library/GPU/shape-key identity. A post-load digest difference routes investigation to loading/postprocessing; equal operands and different projection output localize dispatch/kernel behavior. Include graph-captured selections in the receipt.
2. **Available unchanged-library replay seam:** `AutoTuner.get().save_configs(path)` exports profile decisions; `load_configs(path)` imports them. An explicitly pinned `autotune(False, cache=path)` can replay without profiling, but is **not fail closed by itself**. Select/freeze the calibration table before successor comparison; never choose a table post hoc for favorable output.
3. **Required refusal hooks:** verify file hash and complete environment metadata (no wildcard/missing-metadata bypass); assert `load_configs` returns true; load into a fresh process before any competing in-memory profile entry; require every effective operation/profile key and runner name to be present; refuse **cache-miss fallback**, absent runner, uncovered shape, or retuning/changed table. Do not reject a saved tactic merely because its value is `-1`: the SM120 CUTLASS FP8 runner explicitly advertises `[-1]` at `gemm_base.py:607–614`; distinguish that bound, validated runner/tactic from an absent-key fallback. `search_cache:629–656` prioritizes live memory and silently maps an unknown saved runner name to runner 0; `:667–668,708–726` permits misses/fallback; `:714` explicitly lacks runtime tactic applicability validation. Validate the finite selected runner/tactic set for applicable profiles during the prospective qualification. Do not assume a single success log proves coverage.
4. **Minimal proof sketch:** with byte-equal effective operands and a fixed hash-bound tactic policy in two independent processes, reuse the existing short root plus continuation capture as a diagnostic locator. If first projection now agrees, trace the first later disagreement rather than claim all-model closure. If it still differs, retain the failure and investigate the exact selected implementation or load/capture path. The original full finite-corpus criteria remain necessary for later qualification. A successful short diagnostic does not retroactively repair A/B or qualify the corpus.

Blindly disabling autotuning is not the same as replaying a recorded winning tactic; it can activate fallback implementations and is not a proven deterministic fix. No existing evidence justifies attributing this failure to long-context recurrence or changing the numerical envelope.

## Evidence hashes

| File relative to v2 | SHA-256 |
| --- | --- |
| `p0/monitor/review-response-20260927/native-ab-source-diagnosis-20260929.json` | `ae972eb1a316b1aa806da525a83ce327b039edd5ed8c5ac70fe0075ece658b3d` |
| `p0/monitor/review-response-20260927/native-ab-prefill-trace-comparison-20260929.json` | `3121ec89ba15ce0ef7aed530c54a101051007fa55d63b44707651902207f5536` |
| `p0/monitor/review-response-20260927/native-ab-first-layer-metadata-20260929.json` | `351e9b0f432fce6ac355b29c7f78b081d760074eada93302bc3e73599b99dee9` |
| `p0/monitor/review-response-20260927/native-ab-source-inspection-20260929/MANIFEST.json` | `f13ff1bc3749efbb404ef297fc3624d9dcc3bb2ad850422705fcfb206baaa174` |
| `experiments/review-response-20260927/identity/native_source/vllm__model_executor__layers__mamba__gdn_linear_attn.py` | `c6c1a8ae5c9039f7a17c004761486013de58756a2988e25af78e4b6aa5316019` |
| `experiments/review-response-20260927/identity/native_source/vllm__model_executor__layers__mamba__ops__causal_conv1d.py` | `432ea450c4b2ad827918d209897566ab24fa4d1443a62a700c70b20485812d2c` |
| `experiments/review-response-20260927/workload-plan/inspections/cpu-vllm-reference-source-20260929T004505Z/result/sources/model_executor/layers/quantization/modelopt.py` | `48fa8f183589e91aa2a5c092b5c48b1c740f28b783fd6c09206b9aa0afd41626` |
| `experiments/review-response-20260927/workload-plan/inspections/cpu-vllm-reference-source-20260929T004505Z/result/sources/model_executor/models/qwen3_5.py` | `44cd9f88e43aae567d5a6f07e023f0f56980d330dd73444c5afb725a1c2aa198` |
| `experiments/review-response-20260927/workload-plan/inspections/codex-worker-binding-20260928T225042Z/config.json` | `08abd8e204ecae324108f41acc1bac10549a3cd0728737ecfa0fc94b21bbac73` |

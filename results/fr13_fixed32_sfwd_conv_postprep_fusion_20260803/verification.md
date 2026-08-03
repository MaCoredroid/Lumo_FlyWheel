# Verification

All commands ran from the isolated worktree. `CUDA_VISIBLE_DEVICES` was empty
for offline codegen.

```bash
python3 scripts/fr13_generate_sfwd_conv_postprep_fusion_kernel.py
python3 -m py_compile \
  scripts/fr13_generate_sfwd_conv_postprep_fusion_kernel.py \
  scripts/fr10_phase4_patch_vllm_tree_gdn.py \
  scripts/run_swe_bench_q36_a.py \
  src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion.py \
  src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion_kernel.py \
  tests/test_fr13_fixed32_sfwd_conv_postprep_fusion.py \
  tests/test_fr13_fixed32_sfwd_conv_postprep_wiring.py
```

```bash
bash -n \
  scripts/fr13_launch_forked_fa2_tree_server.sh \
  scripts/fr13_bigdenom_swe_serve_variant.sh
python3 -m pytest -q \
  tests/test_fr13_fixed32_sfwd_conv_postprep_wiring.py \
  tests/test_fr13_fixed32_sfwd_conv_postprep_fusion.py \
  tests/test_fr13_fixed32_conv_source_batch.py \
  tests/test_fr13_fixed32_sfwd_state_fusion.py
```

The CPU suite binds the exact physical32/K64 contract, generated frontier-5
arithmetic, BF16 product and activation boundaries, ordered FP32 adds,
adversarial rounding/cancellation and softplus inputs, distinct recurrence
storages, optional tap, fail-closed layouts/state-bank values/storage bounds,
default-off launcher, and exact B1/B4 byte/launch ledger.

The subsequent runtime-wiring suite also verifies default-off selection,
eager/FULL execution gating, final-FULL output preseed, exact capture binding,
absence of the eager SSI scalar read from the bound launch path, and that the
real-task wrapper classifies the candidate as eager only under
`ENFORCE_EAGER=1`. All 48 focused and related tests passed. Generated
`gdn_linear_attn.py` and `gpu_model_runner.py` patches were applied to pristine
vLLM sources and compiled as Python source.

Offline Triton 3.6 compilation targeted `GPUTarget("cuda", 121, 32)` for B1,
B4, B1+tap, and B4+tap. `cuobjdump --dump-resource-usage` reported 56 registers,
zero stack, zero local, and zero shared bytes for all four profiles. Cubins,
PTX, SASS, raw logs, and compiler caches were deleted/not checked in.

No device API, Docker, service, task, request, response, timing, or acceptance
path was used. Runtime qualification here means host-verified capture wiring,
not a GPU or real SWE acceptance result.

# Verification

All commands ran from the isolated worktree. `CUDA_VISIBLE_DEVICES` was empty
for offline codegen.

```bash
python3 scripts/fr13_generate_sfwd_conv_postprep_fusion_kernel.py
python3 -m py_compile \
  scripts/fr13_generate_sfwd_conv_postprep_fusion_kernel.py \
  src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion.py \
  src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion_kernel.py \
  tests/test_fr13_fixed32_sfwd_conv_postprep_fusion.py
python3 -m pytest -q \
  tests/test_fr13_fixed32_sfwd_conv_postprep_fusion.py
```

The CPU suite binds the exact physical32/K64 contract, generated frontier-5
arithmetic, BF16 product and activation boundaries, ordered FP32 adds,
adversarial rounding/cancellation and softplus inputs, distinct recurrence
storages, optional tap, fail-closed layouts/storage bounds, source-only
launcher, and exact B1/B4 byte/launch ledger.

Offline Triton 3.6 compilation targeted `GPUTarget("cuda", 121, 32)` for B1,
B4, B1+tap, and B4+tap. `cuobjdump --dump-resource-usage` reported 56 registers,
zero stack, zero local, and zero shared bytes for all four profiles. Cubins,
PTX, SASS, raw logs, and compiler caches were deleted/not checked in.

No device API, Docker, runtime patcher, served selector, service, task, request,
response, timing, or acceptance path was used.

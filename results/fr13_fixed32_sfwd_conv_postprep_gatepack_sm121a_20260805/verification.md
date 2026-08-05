# Verification

All commands ran in the isolated source worktree. Both offline builds used an
explicitly empty `CUDA_VISIBLE_DEVICES`, distinct empty Triton caches, Triton
3.6.0, and `ptxas-blackwell` 12.9 targeting `sm_121a`.

```bash
CUDA_VISIBLE_DEVICES='' TRITON_CACHE_DIR=/dev/shm/fr13_gatepack_cache_run1_20260805 \
  /home/mark/fr13_bf16_verifier_head_m32_n256k32s3_cea04fef_20260805/venv/bin/python \
  results/fr13_fixed32_sfwd_conv_postprep_gatepack_sm121a_20260805/offline_codegen_audit.py \
  --repo . \
  --incumbent e4dbf0a521e4b7c21c9ea4f5be0db1839aefc1ea \
  --candidate 0bf56d9d4d024129c2ff485c1802546dd518da30 \
  --output /dev/shm/fr13_gatepack_audit_run1_20260805
```

The command was repeated with `run2` output and cache paths. The two generated
`codegen_summary.json` files were byte-identical. Cubins, PTX, SASS, ELF dumps,
resource dumps, and compiler caches remain outside the repository.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /home/mark/shared/lumoFlyWheel/.venv/bin/python -m pytest -q \
  tests/test_fr13_fixed32_sfwd_conv_postprep_fusion.py \
  tests/test_fr13_fixed32_sfwd_conv_postprep_wiring.py \
  tests/test_fr13_b1_gate_b_m128_directgrid_pass_artifact.py
```

Result: 32 passed. The generated-kernel consistency check and `py_compile` for
the generator, launcher module, generated kernel, audit, and verifier passed.

No device API, Docker, service, real task, request, response, timing, or
acceptance path was used. The artifact makes no runtime performance claim.

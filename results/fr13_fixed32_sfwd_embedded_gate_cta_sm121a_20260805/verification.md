# Verification

Both codegen runs used an explicitly empty `CUDA_VISIBLE_DEVICES`, distinct
empty Triton caches, Triton 3.6.0, and `ptxas-blackwell` 12.9 for SM121a.

```bash
CUDA_VISIBLE_DEVICES='' \
TRITON_CACHE_DIR=/dev/shm/fr13_sfwd_embedded_gate_cache_run1_20260805 \
  /home/mark/fr13_bf16_verifier_head_m32_n256k32s3_cea04fef_20260805/venv/bin/python \
  results/fr13_fixed32_sfwd_embedded_gate_cta_sm121a_20260805/offline_codegen_audit.py \
  --repo . \
  --candidate 086da781207322601fc4876f9f6d69292a4a71a1 \
  --output /dev/shm/fr13_sfwd_embedded_gate_audit_run1_20260805
```

The command was repeated with `run2` cache and output paths. Both generated
`codegen_summary.json` files have SHA-256
`04635c84cde3d8bebdaff444530fb7614467dca0a8f77ae4ae0e3d11d65624a0`.
The B1/B4 baseline cubins are byte-identical across both runs, as are the B1/B4
candidate cubins. Raw cubins, SASS, PTX, ELF dumps, and compiler caches remain
outside the repository.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /home/mark/shared/lumoFlyWheel/.venv/bin/python -m pytest -q \
  tests/test_fr13_fixed32_sfwd_conv_postprep_fusion.py \
  tests/test_fr13_fixed32_sfwd_conv_postprep_wiring.py \
  tests/test_fr13_fixed32_sfwd_b1_block256_codegen_artifact.py \
  tests/test_fr13_fixed32_sfwd_gatepack2_codegen_artifact.py \
  tests/test_fr13_fixed32_sfwd_embedded_gate_codegen_artifact.py \
  tests/test_fr13_b1_gate_b_m128_directgrid_pass_artifact.py
```

Result: `45 passed in 1.67s`.

The generator parity check, Python compilation, checked-in summary verifier,
and SHA-256 verification also pass. No GPU API, Docker, service, real task,
request, response, timing, or acceptance path was used.

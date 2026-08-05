# Verification

Both runs used an explicitly empty `CUDA_VISIBLE_DEVICES`, distinct empty
Triton caches, Triton 3.6.0, and `ptxas-blackwell` 12.9 for SM121a.

```bash
CUDA_VISIBLE_DEVICES='' \
TRITON_CACHE_DIR=/dev/shm/fr13_sfwd_selector_cache_run1_20260805a \
  /home/mark/fr13_bf16_verifier_head_m32_n256k32s3_cea04fef_20260805/venv/bin/python \
  results/fr13_fixed32_sfwd_embedded_gate_selector_sm121a_20260805/offline_codegen_audit.py \
  --repo . \
  --candidate 7e99008327eb1b0609793277a10c282c3d85b7d8 \
  --output /dev/shm/fr13_sfwd_selector_audit_run1_20260805a
```

The command was repeated with `run2` cache and output paths. Both generated
`codegen_summary.json` files have SHA-256
`94c43af1a3c2c8d9035e5c6d0df5172f8078e23c79b2fec57443a2b7d759eff8`.
The standalone B1/B4 cubins are byte-identical across both runs, as are the
embedded B1/B4 cubins. Raw cubins, SASS, PTX, ELF dumps, and compiler caches
remain outside the repository.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /home/mark/lumoFlyWheel-sfwd-conv-postprep-livegate-20260803/.venv/bin/python \
  -m pytest -q \
  tests/test_fr13_fixed32_ingress_proxy.py \
  tests/test_fr13_fixed32_sfwd_conv_postprep_fusion.py \
  tests/test_fr13_fixed32_sfwd_conv_postprep_wiring.py \
  tests/test_fr13_fixed32_sfwd_embedded_gate_codegen_artifact.py \
  tests/test_fr13_fixed32_sfwd_embedded_gate_selector_codegen_artifact.py \
  tests/test_fr13_b1_composed_stack.py \
  tests/test_fr13_b1_u8_cfwd_sfwd_stack_timing.py \
  tests/test_fr13_k64_qrow16_sfwd_stack.py \
  tests/test_fr13_qrow16_production_sidecar.py \
  tests/test_fr13_fixed32_sfwd_state_fusion.py
```

The generator parity check, Python compilation, checked-in summary verifier,
and SHA-256 verification also pass. No GPU API, Docker, service, synthetic
probe, real task, request, response, timing, or acceptance path was used.

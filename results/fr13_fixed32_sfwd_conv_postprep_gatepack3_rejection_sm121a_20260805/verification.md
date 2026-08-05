# Verification

The experiment was applied in an isolated worktree based on
`ee72339c39a83282bbd86298ea4796f71020d334`. It changed the gate-row
multiplier from two to four consistently in the generator, generated kernel,
launcher/ledger, and CPU expectations. The following focused suite passed:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /home/mark/shared/lumoFlyWheel/.venv/bin/python -m pytest -q \
  tests/test_fr13_fixed32_sfwd_conv_postprep_fusion.py \
  tests/test_fr13_fixed32_sfwd_conv_postprep_wiring.py \
  tests/test_fr13_b1_gate_b_m128_directgrid_pass_artifact.py
```

Result: 34 passed. Generator parity passed. The checked-in audit and verifier
also passed `py_compile`.

The checked-in audit reconstructs the exact experimental kernel source from
the accepted source revision and compiles it without a GPU API:

```bash
CUDA_VISIBLE_DEVICES='' \
TRITON_CACHE_DIR=/dev/shm/fr13_gatepack3_reject_cache_run1_20260805 \
  /home/mark/fr13_bf16_verifier_head_m32_n256k32s3_cea04fef_20260805/venv/bin/python \
  results/fr13_fixed32_sfwd_conv_postprep_gatepack3_rejection_sm121a_20260805/offline_codegen_rejection_audit.py \
  --repo . \
  --output /dev/shm/fr13_gatepack3_reject_run1_20260805
```

The command was repeated with distinct `run2` output and Triton cache paths.
Both `summary.json` files were byte-identical. The experimental source changes
were then restored; only this rejection artifact is retained.

No device API, Docker, service, task, request, response, timing, or acceptance
path was used.

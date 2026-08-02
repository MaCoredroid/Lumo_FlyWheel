# Verification

Toolchain:

- Python: `/home/mark/fr13_streamk_build/venv/bin/python`
- PyTorch: 2.10.0+cu130
- Triton: 3.6.0
- backend producer: `ptxas-blackwell` 12.9.86
- target: `sm_121a`

Run two isolated builds from the repository root:

```bash
ART=results/fr13_fixed32_sfwd_v4_sm121a_codegen_20260802
PY=/home/mark/fr13_streamk_build/venv/bin/python
CANDIDATE=3295f4d38045486244b8cea1b1f647edc5617cc0
INCUMBENT=ac8d848b63278a9c956ebbb31b9b7836372816f1

CUDA_VISIBLE_DEVICES= \
TRITON_CACHE_DIR=/dev/shm/fr13_sfwd_v4_sm121a_cache_primary \
  "$PY" "$ART/offline_codegen_audit.py" \
  --repo . --candidate-revision "$CANDIDATE" \
  --incumbent-revision "$INCUMBENT" \
  --output /dev/shm/fr13_sfwd_v4_sm121a_primary

CUDA_VISIBLE_DEVICES= \
TRITON_CACHE_DIR=/dev/shm/fr13_sfwd_v4_sm121a_cache_rebuild \
  "$PY" "$ART/offline_codegen_audit.py" \
  --repo . --candidate-revision "$CANDIDATE" \
  --incumbent-revision "$INCUMBENT" \
  --output /dev/shm/fr13_sfwd_v4_sm121a_rebuild

CUDA_VISIBLE_DEVICES= "$PY" "$ART/verify_codegen_outputs.py" \
  --primary /dev/shm/fr13_sfwd_v4_sm121a_primary \
  --rebuild /dev/shm/fr13_sfwd_v4_sm121a_rebuild
```

Focused host tests:

```bash
python -m pytest -q \
  tests/test_fr13_fixed32_sfwd_v4_codegen_artifact.py \
  tests/test_fr13_fixed32_sfwd_prior_reuse_descriptorless.py \
  tests/test_fr13_fixed32_sfwd_prior_reuse.py
```

The source tests bind the fixed32 topology, exact load/activation order,
load-once invariant, two-activation window, B1/B4 selector geometry, layout
guards, and default-off authenticated selector. The artifact test binds the
offline-only SM121a compiler and spill-free verifier contracts.

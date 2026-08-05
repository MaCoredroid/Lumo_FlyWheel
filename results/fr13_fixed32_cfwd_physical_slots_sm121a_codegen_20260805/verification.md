# Verification

Toolchain:

- Python: `/home/mark/fr13_streamk_build/venv/bin/python`
- PyTorch: `2.10.0+cu130`
- Triton: `3.6.0`
- backend producer: `ptxas-blackwell` 12.9.86
- target: `sm_121a`

From repository root, build twice with independent empty Triton caches:

```bash
ART=results/fr13_fixed32_cfwd_physical_slots_sm121a_codegen_20260805
PY=/home/mark/fr13_streamk_build/venv/bin/python
REV=d2348ce9260292dcf6f9c687a774ed9966b92928

CUDA_VISIBLE_DEVICES= \
TRITON_CACHE_DIR=/dev/shm/fr13_cfwd_physical_slots_cache_a \
  "$PY" "$ART/offline_codegen_audit.py" \
  --repo . --revision "$REV" \
  --output /dev/shm/fr13_cfwd_physical_slots_a

CUDA_VISIBLE_DEVICES= \
TRITON_CACHE_DIR=/dev/shm/fr13_cfwd_physical_slots_cache_b \
  "$PY" "$ART/offline_codegen_audit.py" \
  --repo . --revision "$REV" \
  --output /dev/shm/fr13_cfwd_physical_slots_b

CUDA_VISIBLE_DEVICES= "$PY" "$ART/verify_codegen_outputs.py" \
  --primary /dev/shm/fr13_cfwd_physical_slots_a \
  --rebuild /dev/shm/fr13_cfwd_physical_slots_b
```

Focused host tests:

```bash
python3 -m pytest -q \
  tests/test_fr13_fixed32_cfwd_logit_direct_decision.py \
  tests/test_fr13_fixed32_cfwd_logit_direct_runners.py \
  tests/test_fr13_fixed32_cfwd_logit_direct_live_gate.py \
  tests/test_fr13_fixed32_cfwd_logit_direct_artifact.py \
  tests/test_fr13_fixed32_taw_exact_commit_kernel.py \
  tests/test_fr13_fixed32_taw_exact_commit_cuda.py \
  tests/test_fr13_fixed32_taw_native_precompute.py \
  -k 'not b1_runner_overrides_legacy_vocab_registry_for_full_vocab'
```

The one deselected assertion is already incompatible with the exact frozen
base: its runner lacks the literal it searches for. It is unrelated to this
candidate. The CUDA test skips because this audit intentionally has no GPU.


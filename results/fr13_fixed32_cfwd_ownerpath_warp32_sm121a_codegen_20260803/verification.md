# Verification

Toolchain:

- Python: `/home/mark/fr13_streamk_build/venv/bin/python`
- PyTorch: 2.10.0+cu130
- Triton: 3.6.0
- backend producer: `ptxas-blackwell` 12.9.86
- CUDA disassembler: 13.0.85
- target: `sm_121a`

Run two isolated builds from the repository root:

```bash
ART=results/fr13_fixed32_cfwd_ownerpath_warp32_sm121a_codegen_20260803
PY=/home/mark/fr13_streamk_build/venv/bin/python
CANDIDATE=47e411fb17c0e7f330399ef5698a06ef460c7401
SUPERSEDED=392c16929b40d527f5097eb198479f3370fae9f8
INCUMBENT=6deadde546ad9ee5fee845fabe016383c33f280c

CUDA_VISIBLE_DEVICES= \
TRITON_CACHE_DIR=/dev/shm/fr13_committer_warp32_cache_primary \
  "$PY" "$ART/offline_codegen_audit.py" \
  --repo . --candidate-revision "$CANDIDATE" \
  --superseded-revision "$SUPERSEDED" \
  --incumbent-revision "$INCUMBENT" \
  --output /dev/shm/fr13_committer_warp32_primary

CUDA_VISIBLE_DEVICES= \
TRITON_CACHE_DIR=/dev/shm/fr13_committer_warp32_cache_rebuild \
  "$PY" "$ART/offline_codegen_audit.py" \
  --repo . --candidate-revision "$CANDIDATE" \
  --superseded-revision "$SUPERSEDED" \
  --incumbent-revision "$INCUMBENT" \
  --output /dev/shm/fr13_committer_warp32_rebuild

CUDA_VISIBLE_DEVICES= "$PY" "$ART/verify_codegen_outputs.py" \
  --primary /dev/shm/fr13_committer_warp32_primary \
  --rebuild /dev/shm/fr13_committer_warp32_rebuild
```

Focused host tests:

```bash
PYTHONPATH=src python3 -m pytest -q \
  tests/test_fr13_fixed32_committer_warp32_codegen_artifact.py \
  tests/test_fr13_fixed32_committer_physical32_row_guard.py \
  tests/test_fr13_fixed32_conv_row_guard.py \
  tests/test_fr13_fixed32_conv_commit_wiring.py \
  tests/test_fr13_fixed32_committer_layer_batch.py \
  tests/test_fr13_fixed32_conv_commit_zero_tail.py

python3 scripts/fr13_fixed32_work_census.py --self-test
```

The verifier re-disassembles twelve cubins, checks producer and target,
recounts global/shared/local operations, enforces spill/call/resource gates,
and compares independent fresh-cache outputs byte for byte.

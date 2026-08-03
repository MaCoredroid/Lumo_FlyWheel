# Verification

Toolchain used for the checked result:

- Python: `/home/mark/fr13_streamk_build/venv/bin/python`
- PyTorch: 2.10.0+cu130
- Triton: 3.6.0
- backend producer: `ptxas-blackwell` 12.9.86
- target: `sm_121a`

Run two isolated builds from the repository root. Use new cache and output
paths on every run so stale outputs cannot satisfy the identity gate.

```bash
ART=results/fr13_fixed32_treeconv_zero_tail_sm121a_codegen_20260803
PY=/home/mark/fr13_streamk_build/venv/bin/python
CANDIDATE=0112ac7c49188baa6ab44bb9d9a832423520d8b7
INCUMBENT=b7fc9c594de58d5c38f7ad2da31a262d3cab7669

CUDA_VISIBLE_DEVICES= \
TRITON_CACHE_DIR=/dev/shm/fr13_treeconv_zero_tail_cache_primary \
  "$PY" "$ART/offline_codegen_audit.py" \
  --repo . --candidate-revision "$CANDIDATE" \
  --incumbent-revision "$INCUMBENT" \
  --output /dev/shm/fr13_treeconv_zero_tail_primary

CUDA_VISIBLE_DEVICES= \
TRITON_CACHE_DIR=/dev/shm/fr13_treeconv_zero_tail_cache_rebuild \
  "$PY" "$ART/offline_codegen_audit.py" \
  --repo . --candidate-revision "$CANDIDATE" \
  --incumbent-revision "$INCUMBENT" \
  --output /dev/shm/fr13_treeconv_zero_tail_rebuild

CUDA_VISIBLE_DEVICES= "$PY" "$ART/verify_codegen_outputs.py" \
  --primary /dev/shm/fr13_treeconv_zero_tail_primary \
  --rebuild /dev/shm/fr13_treeconv_zero_tail_rebuild \
  --report /dev/shm/fr13_treeconv_zero_tail_verification.json
```

Focused host tests:

```bash
python3 -m pytest -q \
  tests/test_fr13_fixed32_treeconv_zero_tail_codegen_artifact.py \
  tests/test_fr13_fixed32_conv_commit_zero_tail.py \
  tests/test_fr13_fixed32_conv_commit_wiring.py \
  tests/test_fr13_fixed32_conv_pregather_geometry.py \
  tests/test_fr13_fixed32_committer_layer_batch.py \
  tests/test_fr13_fixed32_conv_source_batch.py
```

The offline verifier re-disassembles every cubin, independently re-reads its
resource report, checks exact `.target sm_121a`, enforces zero spill/local/call
state, compares two isolated reports, and proves selector-off SASS/resource
identity plus the candidate's 34-to-3 source-read reduction.

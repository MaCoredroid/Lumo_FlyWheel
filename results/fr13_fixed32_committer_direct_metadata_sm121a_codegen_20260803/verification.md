# Verification

Toolchain:

- Python: `/home/mark/fr13_streamk_build/venv/bin/python`
- PyTorch: 2.10.0+cu130
- Triton: 3.6.0
- backend producer: `ptxas-blackwell` 12.9.86
- CUDA disassembler: 13.0.85
- target: `sm_121a`

From the repository root, run two builds with distinct empty cache/output
directories:

```bash
ART=results/fr13_fixed32_committer_direct_metadata_sm121a_codegen_20260803
PY=/home/mark/fr13_streamk_build/venv/bin/python
REV=0e2f3b940ee7076e7818da4e048206a978236f04

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/dev/shm/fr13_direct_meta_cache_a \
  "$PY" "$ART/offline_codegen_audit.py" \
  --repo . --revision "$REV" --output /dev/shm/fr13_direct_meta_a

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/dev/shm/fr13_direct_meta_cache_b \
  "$PY" "$ART/offline_codegen_audit.py" \
  --repo . --revision "$REV" --output /dev/shm/fr13_direct_meta_b

CUDA_VISIBLE_DEVICES= "$PY" "$ART/verify_codegen_outputs.py" \
  --primary /dev/shm/fr13_direct_meta_a \
  --rebuild /dev/shm/fr13_direct_meta_b
```

Focused host tests:

```bash
python3 -m pytest -q \
  tests/test_fr13_fixed32_committer_direct_metadata.py \
  tests/test_fr13_fixed32_committer_layer_batch.py \
  tests/test_fr13_fixed32_committer_physical32_row_guard.py \
  tests/test_fr13_fixed32_conv_commit_wiring.py \
  tests/test_fr13_fixed32_conv_commit_zero_tail.py \
  tests/test_fr13_fixed32_final_full_preseed.py
```

The verifier re-disassembles all eight cubins, checks source and binary hashes,
requires zero stack/local/spill/call use, rejects any candidate resource or SASS
regression, verifies the metadata round trip is zero, and compares independent
fresh-cache summaries exactly.

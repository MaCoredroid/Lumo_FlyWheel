# Verification

Source revision:

```text
5d15020c99aa58365096ee1c27a2c1afc4825644
```

The paired fresh-cache builds were produced without GPU visibility:

```bash
CUDA_VISIBLE_DEVICES= \
TRITON_CACHE_DIR=/dev/shm/fr13_committer_bv64_A_5d15020c99aa \
  /home/mark/fr13_streamk_build/venv/bin/python \
  scripts/fr13_codegen_committer_bv64_warp4.py \
  --repo . \
  --revision 5d15020c99aa \
  --output /home/mark/fr13_committer_bv64_codegen_20260805/A

CUDA_VISIBLE_DEVICES= \
TRITON_CACHE_DIR=/dev/shm/fr13_committer_bv64_B_5d15020c99aa \
  /home/mark/fr13_streamk_build/venv/bin/python \
  scripts/fr13_codegen_committer_bv64_warp4.py \
  --repo . \
  --revision 5d15020c99aa \
  --output /home/mark/fr13_committer_bv64_codegen_20260805/B
```

Every file in the two output trees was compared by path and SHA-256. The
result is recorded in `verification.json`; `codegen_summary.json` is the
byte-identical top-level summary from either build.

Focused source and routing suite:

```text
52 passed in 0.70s
```

Command:

```bash
python3 -m pytest -q \
  tests/test_fr13_fixed32_committer_bv64_warp4.py \
  tests/test_fr13_fixed32_committer_layer_batch.py \
  tests/test_fr13_fixed32_committer_gate_ring.py \
  tests/test_fr13_fixed32_committer_knorm_ring.py \
  tests/test_fr13_fixed32_committer_decay_ring.py
```

Additional checks: runner `bash -n`, codegen `py_compile`, and `git diff
--check` all passed.

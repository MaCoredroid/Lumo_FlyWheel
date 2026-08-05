# Verification

All commands ran in the isolated source worktree. The offline builds used an
explicitly empty `CUDA_VISIBLE_DEVICES`, distinct fresh Triton caches, Triton
3.6.0, and `ptxas-blackwell` 12.9 targeting `sm_121a`.

```bash
CUDA_VISIBLE_DEVICES='' \
TRITON_CACHE_DIR=/dev/shm/fr13_walk_trust_A_ed66 \
  /home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_cfwd_packed_walk_node_trust_sm121a_codegen_20260805/offline_codegen_audit.py \
  --repo . \
  --revision ed66c077bd543f90ad18a78ea974325227a21d7d \
  --output /tmp/fr13_walk_trust_A_ed66
```

The command was repeated with cache `fr13_walk_trust_B_ed66` and output
`/tmp/fr13_walk_trust_B_ed66`. The generated `codegen_summary.json` files were
byte-identical. Cubins, PTX, SASS, ELF dumps, resource dumps, and compiler
caches remain outside the repository.

```bash
CUDA_VISIBLE_DEVICES='' \
  /home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_cfwd_packed_walk_node_trust_sm121a_codegen_20260805/verify_codegen_outputs.py \
  --primary /tmp/fr13_walk_trust_A_ed66 \
  --rebuild /tmp/fr13_walk_trust_B_ed66
```

Result: four builds verified, fresh-cache byte identity, B1/B4 static gates
passed, resource-clean gate passed, and no GPU execution.

```bash
PYTHONPATH=$PWD pytest -q \
  tests/test_fr13_fixed32_cfwd_packed_walk_node_trust.py \
  tests/test_fr13_fixed32_cfwd_logit_direct_decision.py -x
```

Result: 55 passed. `py_compile` for the candidate, tests, audit, and verifier
also passed. The checked-in summary verifier and `SHA256SUMS` verification
passed.

No GPU API, Docker, service, real task, request, response, timing, or acceptance
path was used. The artifact makes no runtime performance claim.

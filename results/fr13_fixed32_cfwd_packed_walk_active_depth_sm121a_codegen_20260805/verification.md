# Verification

All commands ran in the isolated CFWD worktree. CUDA visibility was explicitly
empty and compiler caches were kept outside the repository.

```bash
CUDA_VISIBLE_DEVICES='' \
TRITON_CACHE_DIR=/dev/shm/fr13_cfwd_active_depth_A2_cd139 \
  /home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_cfwd_packed_walk_active_depth_sm121a_codegen_20260805/offline_codegen_audit.py \
  --repo . --revision cd1398aee \
  --output /tmp/fr13_cfwd_active_depth_A2_cd139
```

The command was repeated with cache `fr13_cfwd_active_depth_B_cd139` and output
`/tmp/fr13_cfwd_active_depth_B_cd139`.

```bash
CUDA_VISIBLE_DEVICES='' \
  /home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_cfwd_packed_walk_active_depth_sm121a_codegen_20260805/verify_codegen_outputs.py \
  --primary /tmp/fr13_cfwd_active_depth_A2_cd139 \
  --rebuild /tmp/fr13_cfwd_active_depth_B_cd139
```

The summaries were byte-identical and all B1/B4 static resource gates passed.

```bash
PYTHONPATH=$PWD pytest -q \
  tests/test_fr13_fixed32_cfwd_packed_walk_active_depth.py \
  tests/test_fr13_fixed32_cfwd_packed_walk_node_trust.py \
  tests/test_fr13_fixed32_cfwd_logit_direct_decision.py -x
```

Result: `69 passed`. No GPU API, Docker, service, request, task, timing, or
acceptance run was launched by this lane.

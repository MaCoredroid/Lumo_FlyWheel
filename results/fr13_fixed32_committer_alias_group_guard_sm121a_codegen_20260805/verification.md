# Verification

- Source revision: `ea5e32442a68e901a153ba14855708bab247b44e`
- Source parent: `97a0e596f81ca5cb4ae8946e44138f87636c4646`
- Target: CUDA `sm_121a`
- Torch: `2.10.0+cu130`
- Triton: `3.6.0`
- CUDA toolkit producer: `12.9`, `ptxas-blackwell` `V12.9.86`
- Compiled variants: incumbent owner sticky and candidate alias-group sticky
- Compiled batches: B1 and B4
- Independent fresh caches: two
- Builds verified: eight
- Fresh-cache binary identity: pass
- Stack/local/LDL/STL/CALL gate: pass
- Shared-memory/barrier gate: pass
- Aggregate static SASS and LDG reduction gate: pass
- Focused fixed32 committer tests: 104 passed
- GPU execution: none
- Real SWE-Verified execution: none

Reproduction:

```bash
CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/dev/shm/fr13_alias_group_cache_A \
  /home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_committer_alias_group_guard_sm121a_codegen_20260805/offline_codegen_audit.py \
  --repo /home/mark/lumoFlyWheel-fixed32-cfwd-row32-next-20260805 \
  --revision ea5e32442a68e901a153ba14855708bab247b44e \
  --output /dev/shm/fr13_alias_group_codegen_A

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/dev/shm/fr13_alias_group_cache_B \
  /home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_committer_alias_group_guard_sm121a_codegen_20260805/offline_codegen_audit.py \
  --repo /home/mark/lumoFlyWheel-fixed32-cfwd-row32-next-20260805 \
  --revision ea5e32442a68e901a153ba14855708bab247b44e \
  --output /dev/shm/fr13_alias_group_codegen_B

CUDA_VISIBLE_DEVICES= /home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_committer_alias_group_guard_sm121a_codegen_20260805/verify_codegen_outputs.py \
  --primary /dev/shm/fr13_alias_group_codegen_A \
  --rebuild /dev/shm/fr13_alias_group_codegen_B
```

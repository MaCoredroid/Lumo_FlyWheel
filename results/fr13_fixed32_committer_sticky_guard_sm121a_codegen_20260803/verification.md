# Verification

- Source revision: `0ef914864785fdec62f92f72776a7de0df04cc8a`
- Source parent: `c55b95270`
- Target: CUDA `sm_121a`
- Torch: `2.10.0+cu130`
- Triton: `3.6.0`
- CUDA toolkit producer: `12.9`, `ptxas-blackwell` `V12.9.86`
- CUDA visibility: explicitly empty
- Compiled variants: incumbent and sticky candidate at B1 and B4
- Independent caches: two
- Builds verified: eight
- Fresh-cache binary identity: pass
- Stack/local/LDL/STL/CALL gate: pass
- Shared-memory/barrier gate: pass
- Failure-only atomic branch audit: pass
- Focused source tests: 55 passed
- GPU execution: none
- Real SWE-Verified execution: none

Reproduction:

```bash
CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/dev/shm/fr13_sticky_guard_cache_A \
  /home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_committer_sticky_guard_sm121a_codegen_20260803/offline_codegen_audit.py \
  --repo /home/mark/lumoFlyWheel-cfwd-metadata-fusion-next-20260803 \
  --revision 0ef914864785fdec62f92f72776a7de0df04cc8a \
  --output /dev/shm/fr13_sticky_guard_codegen_A

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/dev/shm/fr13_sticky_guard_cache_B \
  /home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_committer_sticky_guard_sm121a_codegen_20260803/offline_codegen_audit.py \
  --repo /home/mark/lumoFlyWheel-cfwd-metadata-fusion-next-20260803 \
  --revision 0ef914864785fdec62f92f72776a7de0df04cc8a \
  --output /dev/shm/fr13_sticky_guard_codegen_B

CUDA_VISIBLE_DEVICES= /home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_committer_sticky_guard_sm121a_codegen_20260803/verify_codegen_outputs.py \
  --primary /dev/shm/fr13_sticky_guard_codegen_A \
  --rebuild /dev/shm/fr13_sticky_guard_codegen_B
```

# Verification

- Candidate source: `b2b4ab6f5ec4ec1f7ac6b5606b711ef2c1f68d37`
- Default-off parent: `178193bd5226d090fa52d5052e93a0f3a6bc0e06`
- Target: CUDA `sm_121a`
- Torch: `2.10.0+cu130`
- Triton: `3.6.0`
- CUDA producer: toolkit `12.9`, `ptxas-blackwell` `V12.9.86`
- CUDA visibility: explicitly empty
- Variants: parent incumbent, current incumbent, and candidate for producer and committer
- Batches: B1 and B4
- Independent fresh caches: two
- Builds verified: 24
- Fresh-cache binary identity: pass
- Default-off SASS identity to parent: pass
- Producer extra-reduction gate: pass, zero added RSQ/shuffle/barrier work
- Producer stack/local/spill gate: pass
- Committer reduction-removal gate: pass
- Focused source tests: 60 passed
- GPU execution: none
- Real SWE-Verified execution: none

Reproduction:

```bash
CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/dev/shm/fr13_knorm_codegen_A_cache \
  /home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_committer_knorm_ring_sm121a_codegen_20260803/offline_codegen_audit.py \
  --repo /home/mark/lumoFlyWheel-cfwd-metadata-fusion-next-20260803 \
  --revision b2b4ab6f5ec4ec1f7ac6b5606b711ef2c1f68d37 \
  --parent 178193bd5226d090fa52d5052e93a0f3a6bc0e06 \
  --output /dev/shm/fr13_knorm_codegen_A

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/dev/shm/fr13_knorm_codegen_B_cache \
  /home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_committer_knorm_ring_sm121a_codegen_20260803/offline_codegen_audit.py \
  --repo /home/mark/lumoFlyWheel-cfwd-metadata-fusion-next-20260803 \
  --revision b2b4ab6f5ec4ec1f7ac6b5606b711ef2c1f68d37 \
  --parent 178193bd5226d090fa52d5052e93a0f3a6bc0e06 \
  --output /dev/shm/fr13_knorm_codegen_B

CUDA_VISIBLE_DEVICES= /home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_committer_knorm_ring_sm121a_codegen_20260803/verify_codegen_outputs.py \
  --primary /dev/shm/fr13_knorm_codegen_A \
  --rebuild /dev/shm/fr13_knorm_codegen_B
```

# Verification

Source revision:

```text
5700ddaf3ff51e0b8dba0d571069ba0d8c158ce6
```

Parent K-norm revision:

```text
12918adaa869d1c88e1424483a189142571406ae
```

The paired fresh-cache builds were produced with:

```bash
CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/dev/shm/fr13_gate_artifact_A_5700 \
  /home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_committer_gate_ring_sm121a_codegen_20260803/offline_codegen_audit.py \
  --repo . \
  --revision 5700ddaf3ff51e0b8dba0d571069ba0d8c158ce6 \
  --parent 12918adaa869d1c88e1424483a189142571406ae \
  --output /tmp/fr13_gate_artifact_A_5700

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/dev/shm/fr13_gate_artifact_B_5700 \
  /home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_committer_gate_ring_sm121a_codegen_20260803/offline_codegen_audit.py \
  --repo . \
  --revision 5700ddaf3ff51e0b8dba0d571069ba0d8c158ce6 \
  --parent 12918adaa869d1c88e1424483a189142571406ae \
  --output /tmp/fr13_gate_artifact_B_5700

CUDA_VISIBLE_DEVICES= /home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_committer_gate_ring_sm121a_codegen_20260803/verify_codegen_outputs.py \
  --primary /tmp/fr13_gate_artifact_A_5700 \
  --rebuild /tmp/fr13_gate_artifact_B_5700
```

Verifier result:

```json
{
  "builds_verified": 24,
  "committer_gate_nonlinears_removed": true,
  "fresh_cache_byte_identity": true,
  "gpu_execution": false,
  "knorm_only_sass_identity": true,
  "producer_extra_gate_nonlinears": 0,
  "schema": "fr13.fixed32.committer_gate_ring.sm121a.verify.v1",
  "status": "PASS"
}
```

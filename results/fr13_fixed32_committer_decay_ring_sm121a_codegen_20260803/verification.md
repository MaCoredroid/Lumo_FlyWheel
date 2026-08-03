# Verification

Source revision:

```text
ecfd1bd30
```

Parent gate-ring revision:

```text
9dbad6245
```

The paired fresh-cache builds were produced with:

```bash
CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/dev/shm/fr13_decay_A_ecfd \
  /home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_committer_decay_ring_sm121a_codegen_20260803/offline_codegen_audit.py \
  --repo . \
  --revision ecfd1bd30 \
  --parent 9dbad6245 \
  --output /tmp/fr13_decay_A_ecfd

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/dev/shm/fr13_decay_B_ecfd \
  /home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_committer_decay_ring_sm121a_codegen_20260803/offline_codegen_audit.py \
  --repo . \
  --revision ecfd1bd30 \
  --parent 9dbad6245 \
  --output /tmp/fr13_decay_B_ecfd

CUDA_VISIBLE_DEVICES= /home/mark/fr13_streamk_build/venv/bin/python \
  results/fr13_fixed32_committer_decay_ring_sm121a_codegen_20260803/verify_codegen_outputs.py \
  --primary /tmp/fr13_decay_A_ecfd \
  --rebuild /tmp/fr13_decay_B_ecfd
```

Verifier result:

```json
{
  "builds_verified": 24,
  "committer_decay_exponentials_removed": true,
  "fresh_cache_byte_identity": true,
  "gate_only_sass_identity": true,
  "gpu_execution": false,
  "producer_extra_decay_nonlinears": 0,
  "producer_raw_ab_reference_stores_preserved": true,
  "schema": "fr13.fixed32.committer_decay_ring.sm121a.verify.v1",
  "status": "PASS"
}
```

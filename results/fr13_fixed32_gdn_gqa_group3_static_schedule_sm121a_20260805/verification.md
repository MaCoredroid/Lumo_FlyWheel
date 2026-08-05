# Verification

Baseline and candidate revisions:

```text
6c28fc58992e495bd8d4c8640370cc82f17316ee
a5174ed5e8ac2d5768a4a9e0fda16786c564e40a
```

Toolchain:

- Python 3.12.3
- PyTorch 2.10.0+cu130
- Triton 3.6.0
- ptxas-blackwell 12.9.86
- CUDA 13.0.85 `nvdisasm` and `cuobjdump`
- target `sm_121a`
- `CUDA_VISIBLE_DEVICES=` for every compile and verification command

Paired independent-cache builds:

```bash
ART=results/fr13_fixed32_gdn_gqa_group3_static_schedule_sm121a_20260805
BASE=6c28fc58992e495bd8d4c8640370cc82f17316ee
CAND=a5174ed5e8ac2d5768a4a9e0fda16786c564e40a
PY=/home/mark/fr13_streamk_build/venv/bin/python

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/dev/shm/fr13_gqa3_static_C_a5174ed \
  "$PY" "$ART/offline_codegen_audit.py" --repo . \
  --baseline-revision "$BASE" --candidate-revision "$CAND" \
  --output /tmp/fr13_gdn_static_schedule_audit_C

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/dev/shm/fr13_gqa3_static_D_a5174ed \
  "$PY" "$ART/offline_codegen_audit.py" --repo . \
  --baseline-revision "$BASE" --candidate-revision "$CAND" \
  --output /tmp/fr13_gdn_static_schedule_audit_D

CUDA_VISIBLE_DEVICES= "$PY" "$ART/verify_codegen_outputs.py" \
  --primary /tmp/fr13_gdn_static_schedule_audit_C \
  --rebuild /tmp/fr13_gdn_static_schedule_audit_D
```

The audit stages Git revision sources at a fixed content-addressed path and
sets a deterministic source timestamp. The verifier compares every generated
file byte-for-byte, independently disassembles all eight cubins, reparses
resources and opcode counts, enforces the exact B1/B4 launch contracts, and
rejects resource, spill, descriptor-load, or schedule regressions.

No GPU execution, CUDA context, serving process, SWE task, or timing run was
used by this artifact.

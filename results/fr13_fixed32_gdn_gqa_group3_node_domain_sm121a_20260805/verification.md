# Verification

Baseline and candidate revisions:

```text
9091ddae2046f42fc5e754f976c3493a033785ac
8c85135cb6092f01230d93c55b1c6f3fcf7336f3
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
ART=results/fr13_fixed32_gdn_gqa_group3_node_domain_sm121a_20260805
BASE=9091ddae2046f42fc5e754f976c3493a033785ac
CAND=8c85135cb6092f01230d93c55b1c6f3fcf7336f3

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/dev/shm/fr13_gqa3_node_E_8c85135 \
  /home/mark/fr13_streamk_build/venv/bin/python \
  "$ART/offline_codegen_audit.py" --repo . \
  --baseline-revision "$BASE" --candidate-revision "$CAND" \
  --output /tmp/fr13_gqa3_node_E_8c85135

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/dev/shm/fr13_gqa3_node_F_8c85135 \
  /home/mark/fr13_streamk_build/venv/bin/python \
  "$ART/offline_codegen_audit.py" --repo . \
  --baseline-revision "$BASE" --candidate-revision "$CAND" \
  --output /tmp/fr13_gqa3_node_F_8c85135

CUDA_VISIBLE_DEVICES= /home/mark/fr13_streamk_build/venv/bin/python \
  "$ART/verify_codegen_outputs.py" \
  --primary /tmp/fr13_gqa3_node_E_8c85135 \
  --rebuild /tmp/fr13_gqa3_node_F_8c85135
```

The audit stages Git revision sources at a content-addressed fixed path and
sets a deterministic source timestamp so debug metadata is reproducible. The
verifier compares every generated file byte-for-byte, independently
disassembles all four cubins, reparses resources and opcode counts, enforces
the exact B4 launch contract, and rejects any resource or spill regression.

No GPU execution, CUDA context, serving process, SWE task, or timing run was
used by this artifact.

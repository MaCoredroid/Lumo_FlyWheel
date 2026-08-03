# Verification

Source revision:

```text
936dd110c01d34f8c1c5c64676dde5739d0d2fa3
```

Toolchain:

- Python 3.12
- PyTorch 2.10.0+cu130
- Triton 3.6.0
- ptxas-blackwell 12.9.86
- CUDA 13.0.85 `nvdisasm` and `cuobjdump`
- target `sm_121a`
- `CUDA_VISIBLE_DEVICES=` for every compile and verification command

Paired independent-cache builds:

```bash
ART=results/fr13_fixed32_gdn_gqa_group3_sm121a_codegen_20260803
REV=936dd110c01d34f8c1c5c64676dde5739d0d2fa3

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/dev/shm/fr13_gqa3_exact_A_936dd110 \
  /home/mark/fr13_streamk_build/venv/bin/python \
  "$ART/offline_codegen_audit.py" --repo . --revision "$REV" \
  --output /tmp/fr13_gqa3_exact_A_936dd110

CUDA_VISIBLE_DEVICES= TRITON_CACHE_DIR=/dev/shm/fr13_gqa3_exact_B_936dd110 \
  /home/mark/fr13_streamk_build/venv/bin/python \
  "$ART/offline_codegen_audit.py" --repo . --revision "$REV" \
  --output /tmp/fr13_gqa3_exact_B_936dd110

CUDA_VISIBLE_DEVICES= /home/mark/fr13_streamk_build/venv/bin/python \
  "$ART/verify_codegen_outputs.py" \
  --primary /tmp/fr13_gqa3_exact_A_936dd110 \
  --rebuild /tmp/fr13_gqa3_exact_B_936dd110
```

The verifier compares every generated file byte-for-byte, independently
disassembles all eight cubins, reparses resource records, enforces exact B1/B4
grids and launch options, rejects any stack/local/`LDL`/`STL`/call use, and
checks that aggregate static SASS/LDG/STG proxies decrease despite the higher
per-CTA register count.

No GPU execution, CUDA context, serving process, SWE task, or timing run was
used by this artifact.

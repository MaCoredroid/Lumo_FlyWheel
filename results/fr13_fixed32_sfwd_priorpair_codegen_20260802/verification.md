# Verification

- GPU execution: none; `CUDA_VISIBLE_DEVICES` was empty for offline codegen.
- Target: SM121a, fixed rows 32, C64, W16, 16 warps, 3 stages.
- Reproducibility: two empty-cache builds matched byte-for-byte for B1 and B4.
- B1/B4 common cubin SHA-256: `4f297a867ac0474bab44d92ae6acac6cabbbf9e27532a59f97a662964d069694`.
- Resource guard: 55 registers, 4096 launch shared bytes, zero stack/local spill.
- Focused source tests: 22 passed.
- Candidate identity was split from the packed-xgather baseline before packaging.
- Eligibility: static codegen only; not eligible for acceptance or timing claims.

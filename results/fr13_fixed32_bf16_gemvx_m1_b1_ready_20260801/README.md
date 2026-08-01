# FR13 full-vocabulary BF16 M1 head real-B1 readiness

This artifact records CPU-only build and launch readiness for a deterministic,
real SWE-Verified B1 shadow gate. It makes no GPU byte-parity, speed,
hardware-floor, quality, B4, or acceptance claim.

## Candidate

- Source branch: `agent/fixed32-fullhead-m1-gemvx`
- Source checkpoint: `531ac32c7fff4656e593af024dc13273b89ed3c1`
- CUDA source SHA-256:
  `26ea8aad9f891b5e758a39464209d6f82008a10fac8da4c02ee052e839218a54`
- Candidate SO SHA-256:
  `7d6c549e741d8fbbc54732ba5873a8c01f7f089f15a8589ef51eb49a45f5e6d5`
- Candidate SO bytes: `162160`
- Build attestation SHA-256:
  `0168d7e50f14b8b87ef279ba9c5760bd06fee5d9492e3f7c02ac0be0ad85f589`
- Toolchain: PyTorch `2.10.0+cu130`, CUDA `13.0`, `sm_121a`
- Default behavior: candidate selector off; stock head unchanged
- Served behavior during the gate: stock logits computed first and returned

The kernel retains the observed stock M1 geometry: grid `[31040,1,1]`, block
`[16,8,1]`, 544 dynamic shared bytes, eight output rows per CTA, 16 K lanes,
320 scalar BF16 iterations per lane, FP32 dependent FMA accumulation, and
shared reductions at strides 8, 4, 2, and 1. The epilogue supplies alpha 1 and
beta positive zero as kernel parameters so compiled SASS retains the stock
`FFMA` immediately before BF16 conversion.

## Real gate

The default-off runtime computes stock logits first, computes the candidate
second, compares raw `torch.int16` views of all 248,320 BF16 logits, and returns
only stock. It maintains independent exact counters for root, MTP1, MTP2,
MTP3, and MTP4. The final record passes only when every position has exactly
one comparison per authenticated measured event and zero mismatches.

The runner is pinned to SWE-Verified task `astropy__astropy-12907`, concurrency
1, batch 1, the canonical FA2 object, full-vocabulary root=0/K=0, a clean exact
commit, immutable runtime manifests, terminal flush evidence, and authenticated
chat-traffic evidence. It validates the build attestation before any Docker or
GPU launch. This is a one-task byte diagnostic, never an acceptance or timing
arm.

Run `prepared_command.sh` only after the GPU is free. Supply the exact trusted
branch or merged commit as `PINNED_SOURCE_COMMIT`; the runner rejects any other
HEAD and any tracked or untracked checkout change.

```bash
PINNED_SOURCE_COMMIT=<exact-trusted-commit> \
  bash results/fr13_fixed32_bf16_gemvx_m1_b1_ready_20260801/prepared_command.sh
```

## Floor scope

The full-vocabulary K=0 campaign retains mandatory bytes `42025179008` and
the existing weight-read floor `153.938384645 ms/event`. This readiness package
does not measure wall time or assert that the candidate reaches that floor.
Formal performance acceptance still requires the standing real exact4/exact16
task sets. B4 requires a separately qualified batched head kernel and is not
covered by this M1 gate.

## Verification

- Generated build-attestation validation: pass
- Python compilation: pass
- Shell syntax: pass
- Ruff: pass
- Focused and neighboring CPU tests: 28 passed
- Broad fixed32 CPU suite: 667 passed, 8 environment skips
- GPU or Docker launched: no
- Real SWE-Verified task launched: no

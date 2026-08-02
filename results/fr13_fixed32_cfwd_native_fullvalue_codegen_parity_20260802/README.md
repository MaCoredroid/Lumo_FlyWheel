# Fixed32 CFWD native full-value codegen parity

Status: **SM121a math-tree and resource contract passed; default off; real-task
raw-byte and timing qualification pending**.

## Result

The native CUDA candidate now deliberately matches the incumbent Triton BV64
and BV128 FP32 operation order for the numerically sensitive recurrence:

- Each contiguous K=32 product group uses local multiply, XOR-16 partner,
  FMA, then separately rounded XOR-8/4/2/1 additions.
- K norm and each state-dot row combine the four group totals as
  `(group 0 + group 2) + (group 1 + group 3)`.
- State decay is a separately rounded multiply; residual is subtract then
  multiply; state update is a fused multiply-add.
- `tl.exp`, reciprocal square root, and full division are pinned with inline
  PTX as `ex2.approx.f32`, `rsqrt.approx.ftz.f32`, and `div.full.f32`.

An exact full-translation-unit SM121a compile retained the previous resource
target:

| Resource | Result |
| --- | ---: |
| Registers/thread | 64 |
| Stack frame | 0 bytes |
| Spill stores / loads | 0 / 0 bytes |
| Local memory | 0 bytes |
| Source static shared memory | 548 bytes |
| Reported shared memory | 1,572 bytes |
| SASS `LDL` / `STL` / `CALL` | 0 / 0 / 0 |

The arithmetic changes initially caused the compiler to retain and spill one
64-bit final-store address across the recurrence. A fresh inline-PTX `%tid.x`
read on the store side prevents that common-subexpression lifetime; the exact
recompile returned to zero stack and zero spills at 64 registers/thread.

## SASS contract

The reduced object checker passed with these static instruction counts:

| Opcode | Count |
| --- | ---: |
| `MUFU.EX2` | 3 |
| `MUFU.RSQ` | 1 |
| `MUFU.RCP` | 2 |
| `SHFL.BFLY` | 174 |
| `SHFL.IDX` | 16 |
| `FFMA` | 78 |

`scripts/fr13_check_cfwd_native_fullvalue_codegen.py` checks these counts,
the exact SM121a resource shape, and the absence of local-memory and call
instructions directly from a compiled object.

## Remaining byte risk

This is not a raw-byte credential. CUDA `logf` lowers to the same softplus log
polynomial constants and arithmetic order observed in the Triton PTX, but its
native PTX uses non-FTZ FMA modifiers and different branch/select control flow
where Triton uses FTZ FMA modifiers. The log argument is at least one and the
normal real-task path is not expected to exercise a subnormal polynomial
intermediate, but only the authenticated all-bank raw-byte gate can establish
that claim. A future strict edge-case path can inline the complete Triton log
polynomial with explicit `.ftz` PTX if the real byte gate localizes a mismatch
there.

The native row-owning warp layout also differs from Triton's BV64/BV128
transport and shared-memory schedule even though the FP32 operation tree now
matches. B1 and canonical exact4 B4 real SWE-Verified byte gates must pass
before timing. No hardware-floor distance changes from this compile-only work.

## Verification scope

- Focused source, selector, patcher, and codegen-checker suite: `42 passed`.
- Python byte compilation and `git diff --check`: pass.
- Exact host-only SM121a full-translation-unit compile: pass.
- Exact object/SASS contract checker: pass.
- Ruff was unavailable in this shell (`python3 -m ruff`: module not found).
- No GPU query, CUDA kernel launch, Docker mutation, synthetic/probe timing,
  real SWE-Verified campaign, or timing measurement was performed.

This directory contains reduced compiler/codegen facts only. It contains no
prompts, responses, patches, traces, raw logs, process/container identities,
credentials, PTX, SASS, object files, or timing samples.

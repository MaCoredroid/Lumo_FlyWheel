# Fixed32 full-head BF16 M1 production timing readiness

Status: `SOURCE_READY_CPU_VALIDATED`

This package prepares the missing B1 production and exact4 timing route for the
qualified stock-order BF16 M1 GEMV. It does not claim a GPU launch, a fresh byte
gate, timing results, B4 support, or hardware-floor acceptance.

## Implemented

- `FR13_DRAFT_HEAD_M1_PRODUCTION=1` serves only the custom M1 output. The stock
  `ParallelLMHead` is not executed in that branch.
- The live route remains stock-first, candidate-shadow, and stock-served.
- The launcher issues and container-verifies a production sidecar bound to the
  live result, terminal flush, boundary snapshot, authenticated traffic audit,
  CUDA source, patcher, build attestation, and exact SO SHA/size.
- Production engagement requires one root selection, four captured MTP head
  selections, zero fallback, the fixed32 graph signature, and a measured graph
  replay.
- The timing runner uses the canonical real SWE-Verified exact4 subset at B1,
  concurrency 1, full vocabulary (`root=0`, `K=0`), full-wall deploy-speed, and
  SFWD/DFWD/CFWD phase timers.
- The stock arm receives no M1 SO or live credential and must emit no M1
  production sidecar or engagement file.

## Required launch order

1. Run `prepared_live_gate.sh` from a clean trusted commit containing
   implementation commit `58c02bcc313a12d47227d7df63e7faaf88e1a1d5`.
2. Confirm the real B1 M1 live result is `PASS` and retain its final flush,
   boundary snapshot, and authenticated traffic audit.
3. Run `prepared_timing.sh` with that gate runroot and arm name.

The earlier gate prepared before `58c02bcc3` is not a valid production
credential because M1 live evidence binds the patcher SHA. Reusing it must fail.

## Scope

This route is B1-only. The exact4 task set supplies four real tasks while the
server batch and concurrency remain one. B4 M1 requires a separate B4-capable
kernel and full-logit graph qualification; no B4 claim is made here.

The timing summary is a classification candidate and records
`floor_acceptance_eligible=false`. Formal hardware-floor closure still requires
the standing acceptance campaign after the timing pair demonstrates a viable
full-step ratio.

## CPU validation

- Focused M1/M32/source/launcher regression: `29 passed`.
- Python compilation: pass.
- Shell syntax: pass.
- Ruff on changed Python/tests: pass.
- Patcher Ruff with its four pre-existing `F401/F841` findings excluded: pass.
- Exact source-to-SO build validation: pass; SO SHA
  `7d6c549e741d8fbbc54732ba5873a8c01f7f089f15a8589ef51eb49a45f5e6d5`,
  162160 bytes.

No Docker container or GPU workload was launched while producing this package.

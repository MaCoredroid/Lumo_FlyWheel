# FR13 fixed32 kernel production stack integration

This artifact binds a source-only, default-off integration of the fixed32 TAW
and GDN production selectors, the unified-attention BM8 live diagnostic, and
the inherited B2-B4 two-launch GDN route. No GPU or real SWE-Verified run was
performed for this integration artifact, and it makes no byte-parity, speed,
acceptance, or hardware-floor claim.

## Source lineage

- Base: `0fbd1a5e4cf69777661bc574f556f89dd212c512`
- Branch: `agent/fixed32-kernel-production-stack`
- TAW/GDN selector source: `7a0072f8b86bf25daf4ba3ca69e937d08a7049e4`
- TAW/GDN selector cherry-pick: `39907d3dc10808b7d5b610483576e953467626a4`
- Unified BM8 source: `a477684375fce02ecbe0b0563600042943609900`
- Unified BM8 cherry-pick/source head: `a64b883504cd8aa600942f130cad51b4d0897335`
- Both cherry-picks applied without conflict.
- No qrow16 commit was added by this integration.

## Default-off and lifecycle contract

- TAW production is default-off, requires a source-bound live PASS, has no
  reference fallback, and is mutually exclusive with its diagnostic mode.
- GDN path-BV production is default-off, requires a source-bound live PASS,
  has no BV8 fallback, and is mutually exclusive with its diagnostic mode.
- Unified BM8 remains a B1 live diagnostic only. It serves the stock captured
  graph output and cannot be enabled through its launcher-private selector.
- B2-B4 batched GDN is default-off. Production requires a matching real-event
  byte-gate PASS; diagnostics restore and serve the reference result.
- The pinned one-task B1 diagnostic and canonical exact4/exact16 guards from
  the base remain unchanged and covered by the focused and broader tests.

## B4 launch topology finding

The fixed32 batched kernel folds request index into path-grid axis 2 and has an
exact two-level schedule, so its launch count is structurally two launches per
layer for B2-B4 at the currently permitted `BLOCK_V <= 8`. It does not yet
establish two launches per layer for BV16, BV32, BV64, or BV128: the batched
launcher explicitly rejects `BLOCK_V > 8`, and batched-GDN selectors are
mutually exclusive with the path-BV selectors. A combined B4 plus wide-BV
candidate therefore needs separate source work and a real-event byte gate.

## Verification

- Python compilation: PASS.
- Launcher shell syntax: PASS.
- Focused integration tests: `49 passed`.
- Broad fixed32 suite: `564 passed, 8 skipped, 3 failed`; all three failures
  are the known isolated-worktree prerequisites (`.venv/bin/python` and the
  worktree-local SWE-Verified dataset cache), not source failures.
- Broad source-relevant suite with those two environment-dependent modules
  excluded: `403 passed, 12 skipped`.
- GPU used: no.


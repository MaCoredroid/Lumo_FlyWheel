# FR13 B1 BM8 composed-stack source readiness

Status: `SOURCE_READY_ONLY`.

Implementation commit `6f11fcb72d060cce52380ccca78713ae55ba6fa9`
extends the authenticated Hydra27 B1 composed stack with the existing unified-
attention BM8 candidate. Nonstock CUTLASS composition is admitted only for the
exact B1 K64/root1 Qrow32 split2, GQA3, DFWD top3, wide256 target GEMM, SFWD,
TAW, CFWD, and BM8 production tuple. Every other BM8/nonstock-CUTLASS tuple
still fails before sidecar issuance or Docker launch.

The BM8 production engagement schema is now v2. It is published as `ENGAGED`
only after one final FULL B1 drafter graph capture, exactly four guarded BM8
unified-attention dispatches, and the first measured replay. The artifact makes
the lifecycle explicit with `graph_captures=1`, `measured_replays=1`, and
`unmeasured_replays=0`.

The composed production-smoke schema is v2 and the exact4 timing schema is v3.
Both require the pinned BM8 production credential and the current run's v2
measured-replay engagement. An older smoke credential cannot authorize this
timing path.

CPU-only verification completed:

- Bash syntax validation passed for the launcher, composed runner, and wrappers.
- Python bytecode compilation passed for the patcher, gate, timing reducer, and
  runtime manifest.
- The focused composed/BM8/CUTLASS suite passed 48 tests.
- The expanded BM8/composed suite passed 54 tests; one source-artifact-only test
  was deselected because that historical directory is outside this worktree's
  sparse checkout.

No GPU, Docker, SWE task, production-return smoke, or performance timing was run.
This artifact makes no speedup, hardware-floor, acceptance, B4, exact4, or
exact16 claim.

The next valid evidence step is one real SWE-Verified Hydra27 B1 production
smoke. It must resolve cleanly and emit the v2 BM8 `ENGAGED` artifact. Only then
may the standing real SWE-Verified exact4 set be timed with this stack.

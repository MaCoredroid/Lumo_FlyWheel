# Grouped-SIMD B4 qualification readiness

This bundle binds the static implementation of
`fixed32_gdn_parent_group_simd_v2` to source commit
`6b71ff85249064b6a44831ab56f745d4d75dab0e`.

The candidate groups the 11 level-1 paths into five width-4 SIMD groups while
retaining 32 physical rows, BV8, 32 single-writer nodes, and a critical path of
12. At B4 the qualified comparison is eight incumbent launches per layer
against two candidate launches per layer. Qualification always restores and
serves incumbent bytes.

## Live gates still required

For each mode (`tail6_fixed32` and `hydra27_fixed32`), run all four real
SWE-Verified gates:

1. exact4 eager
2. exact4 final full-graph replay
3. exact16 eager
4. exact16 final full-graph replay

Use `scripts/fr13_run_b4_gdn_parent_group_simd_live_gate.sh` with a new
`RUNROOT`, a unique `TAG`, the pinned absolute `FORKED_FA2_SO`, and explicit
`CAMPAIGN`, `GATE`, and `FIXED32_MODE` values. The runner fixes concurrency and
server capacity at B4, fixes the draft vocabulary at K64 with root reduction
off, disables autocommit, and rejects synthetic traffic.

Production remains closed. A production process validates four immutable,
single-link PASS files for its exact mode, rechecks their source, parent
contract, writer, campaign, graph, and 48-layer identities, then derives one
credential. B1-B3 graph preseed and every underfilled per-request fallback stay
on the incumbent while a B4 selector is armed.

## Evidence boundary

No Docker container, GPU kernel, synthetic probe, SWE-Verified task, timing
campaign, TPS measurement, hardware-floor measurement, or U95 acceptance test
was run for this bundle. It contains no prompts, responses, logs, task IDs, or
per-task records.

Static verification completed with 147 focused tests passing. The broader
fixed32 suite reported 832 passed and 8 skipped. Three failures require the
absent local `.venv` or private dataset cache; one unrelated lifecycle timing
fixture failed once and passed on immediate isolated retry. Independent review
found no remaining static boot, correctness, or fail-closed blocker.

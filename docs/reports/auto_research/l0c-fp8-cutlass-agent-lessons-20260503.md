# L0c FP8 CUTLASS Agent Lessons

Generated: 2026-05-03

Scope: CUTLASS-only `fp8_gemm` L0c auto-research after the
`qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260503T021359Z`
round.

## Decisions

- The proposal agent must not run `auto-research apply-and-test` for FP8/CUTLASS.
  That command owns the expensive vLLM restart, parity ladder, and measurement
  window, so it belongs to the controller after the candidate patch is submitted.
- The agent still needs a cheap submission check. Before exit it should verify
  that `mutation.patch` exists, applies cleanly with `patch --dry-run`, and does
  not trip the controller's cheap preflight rules.
- If the cheap check fails, the candidate is a submission failure, not an
  experiment. The agent should revise the patch before exiting when possible; if
  it exits anyway, the controller records the failure and the next attempt must
  use that failure as negative memory.
- CUTLASS-only means no DeltaNet mutation or DeltaNet-derived optimization story
  in this round. DeltaNet history is useful only for controller mechanics.

## Agent Contract

The FP8/CUTLASS agent should do exactly this:

1. Read `iteration_brief.md`, `strategy_brief.md`, and prior rejected mutations.
2. Produce `mutation.patch` in the assigned candidate directory.
3. Run only cheap local submission checks. Do not start vLLM. Do not call
   `auto-research apply-and-test`.
4. If the cheap check fails, revise `mutation.patch` and re-run the cheap check.
5. Exit only after the patch is cheap-check clean, or write `NO_VALID_MUTATION.md`
   if no valid CUTLASS mutation surface exists.

## Controller Contract

The controller remains authoritative for:

- patch preflight;
- canary admission;
- parity;
- measurement;
- accept/reject ledger writes;
- round termination.

This preserves the pivot-doc design: cheap candidate checks happen before
expensive vLLM restarts, and the controller is the only writer of canonical
validation results.

## Next Loop Guidance

The next loop should run with a larger wall-clock/attempt budget but with the
same CUTLASS-only scope. The expected useful outcome is either an accepted
candidate or a clearer failure signal showing that the current repo-owned
CUTLASS overlay cannot produce valid kernel-level mutations. If candidates keep
failing only cheap preflight, the next engineering step is to expose a true
vLLM/CUTLASS source-build mutation target or an isolated FP8 GEMM replay gate.

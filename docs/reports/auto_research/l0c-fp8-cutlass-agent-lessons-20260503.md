# L0c FP8 CUTLASS Agent Lessons

Generated: 2026-05-03

Scope: CUTLASS-only `fp8_gemm` L0c auto-research after the
`qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260503T021359Z`
round.

## Decisions

- The proposal agent must not run `auto-research apply-and-test` for FP8/CUTLASS.
  That command owns the expensive vLLM restart, parity ladder, and measurement
  window, so it belongs to the controller after the candidate patch is submitted.
- The agent still needs cheap submission checks. Before exit it should verify
  that `mutation.patch` exists, applies cleanly with `patch --dry-run`, passes
  local syntax/compile checks for changed source files, and passes the
  controller's cheap `auto-research preflight-patch` command.
- The preflight command must show the matching rule, the short code snippet that
  implements the check, and the evidence snippet from the candidate patch. The
  agent needs that local feedback so it can revise before submitting.
- FP8/CUTLASS preflight must not forbid editing a file merely because it is the
  CUTLASS source file. It should reject safety/correctness violations such as
  parity-fixture edits, controller edits, rejection-ledger edits, or patches that
  fail local application/compile checks.
- The approved CUTLASS surface is now a staged local copy of the live vLLM
  CUTLASS scaled-mm source tree. Agents may edit any file under that staged tree;
  the controller mounts the patched tree over the container source directory for
  parity and measurement.
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
3. Run only cheap local submission checks: `patch --dry-run`, local
   `python3 -m py_compile` for changed Python files, then
   `auto-research preflight-patch`. Do not start vLLM. Do not call
   `auto-research apply-and-test`.
4. If any cheap check fails, read the preflight `matching_rule`, `code_snippet`,
   and `evidence_snippet` when present, revise `mutation.patch`, and re-run the
   cheap checks.
5. Exit `0` only after the patch is cheap-check clean, or after writing
   `BLOCKED.md` if no valid CUTLASS mutation surface exists. Nonzero exit is for
   agent/tool infrastructure failure, not for a rejected proposal.

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

The next loop should run with the same CUTLASS-only scope, but agents should
target `cutlass_source_workspace`, not the old bootstrap overlay. The expected
useful outcome is either an accepted candidate or a real compile/parity failure
from an actual CUTLASS source edit. If candidates now fail, the failure should be
about syntax/import, parity, or measured performance, not a path-based preflight
block.

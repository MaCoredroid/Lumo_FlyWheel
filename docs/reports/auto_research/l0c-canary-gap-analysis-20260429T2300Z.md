# L0c Canary Gap Analysis: Strategy Guidance vs Enforcement

Generated: 2026-04-29T23:00:00Z

## Summary

The current L0c canary has a planning/research stage, but it is advisory rather than enforceable. The fresh Codex agent received the round strategy brief and prior rejection ledger, including the known-bad mutation families. Candidate `001` still proposed a patch that violated those families. This means the immediate gap is not missing context; it is missing deterministic controller-side rejection before `apply-and-test`.

The safe next step is to add a controller preflight guard for forbidden patch shapes, record those rejections into `mutations_rejected.tsv`, and skip expensive runtime restart/parity measurement for patches that are already outside the allowed search space.

## Current Status

- Repo state at analysis time: `main` clean and pushed to `origin/main`.
- Latest code commit: `afc2601 fail fast on stale inference proxy port`.
- Active L0c attempt was stopped after candidate `001` produced an invalid patch.
- Runtime was cleaned: no L0c controller, Codex worker, vLLM container, inference proxy, or listeners on ports `8100`/`8101`.
- Round directory:
  `output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-deltanet-20260429T223605Z`
- Completed baseline artifacts:
  `baselines/cold_discard_00.json` and `baselines/measurement_01.json` through `measurement_05.json`.
- Candidate artifact:
  `candidates/001/mutation.patch`

## What Worked

The L0c bootstrap produced the expected strategy artifacts:

- `strategy_brief.md`
- `prior_mutations_rejected.tsv`
- per-candidate `iteration_brief.md` embedding the strategy and prior rejection guidance

The strategy brief did carry forward P3a and prior-run facts:

- P3a decode share was dominant (`decode_share=0.977`), supporting DeltaNet-first focus.
- Prior rejected mutation families were summarized.
- The prompt explicitly prohibited `.cg` changes on dot-adjacent `w`/`k`, store hints, broad `v`/`h0` cache changes, new gate-path cache hints, `g`/`gk` rebasing/address rewrites, and shared `t_start` rewrites.
- Ranked likely-safe targets were narrowed to scalar metadata/address cleanup, local mask-only cleanup, rollback/tightening, and comment-only proposals.

This is enough to answer the "are we passing the failure cases down?" question: yes, the failure cases were passed down in the prompt and files.

## Observed Failure

Candidate `001` violated the forbidden families anyway. The patch:

- removed the pre-offsets:
  - `g = g + bos * H + i_h`
  - `gk = gk + (bos * H + i_h) * K`
- rewrote `g` and `gk` address arithmetic inline at each load site
- removed the shared `t_start = i_t * BT` scalar and expanded it inline across tile pointer construction
- retained gate-path `.cg` cache hints in the changed gate-load area

Those edits directly conflict with the active strategy brief's forbidden mutation list. This is evidence that fresh-agent prompting alone is not a sufficient memory or policy mechanism for known-bad kernel mutation families.

## Root-Cause Assessment

The agent did not lack context; it lacked a hard gate.

The current design follows a Karpathy-style fresh-agent-per-attempt pattern. That pattern depends on externalized memory files rather than a persistent agent memory. This is acceptable for crash isolation and best-of-N exploration, but only if the controller treats prior failures as executable constraints. Right now, the constraints are mostly natural-language instructions. A capable model can still overfit to a local optimization idea and produce a forbidden patch.

The controller already has correctness gates after patch generation:

- patch application
- runtime restart/compile
- parity probe
- measurement
- rejection ledger append

But the controller does not yet have a preflight semantic patch guard between "agent wrote `mutation.patch`" and "spend runtime on this candidate." That is the missing layer.

## Why This Matters

Without a preflight guard, the loop wastes expensive wallclock on candidates that are known invalid before compile/parity:

- GB10 runtime restart and parity probes are expensive.
- Repeating forbidden families consumes the small `total_attempt_cap`.
- Prompt-only warnings make the run stochastic: the same forbidden family can reappear in a fresh round despite being documented.
- The rejection ledger becomes observational rather than operational; it records failures after cost has already been paid.

For L0c, prior rejection knowledge should affect the next run as a machine-enforced search-space boundary, not only as text in an agent prompt.

## Proposed Fix

Add an L0c controller-side forbidden-mutation preflight immediately after reading `mutation.patch` and computing its hash, before calling `apply_and_test`.

Minimum preflight behavior:

1. Inspect the unified diff text for known forbidden families.
2. If a forbidden family is detected, write `parity_check.json` with `pass=false` and a reason such as `forbidden_mutation_family`.
3. Write or overwrite `BLOCKED.md` with the specific forbidden family and evidence snippet.
4. Append a row immediately to `mutations_rejected.tsv` using the existing incremental rejection writer.
5. Count the attempt as rejected, but do not restart vLLM or run parity.

Initial forbidden detectors should cover the exact families seen in this canary:

- deleting or moving the `g` pre-offset line
- deleting or moving the `gk` pre-offset line
- introducing inline expressions of the form `g + bos * H + ...` at gate-load sites
- introducing inline expressions of the form `gk + (bos + last_idx) * H * K + ...`
- deleting `t_start = i_t * BT`
- broad replacement of existing `t_start` offsets with repeated `i_t * BT` expressions
- new or retained gate-path `.cg` hints on `g`/`gk` loads when the strategy forbids them

This guard should be deliberately conservative: it should reject known-bad shapes, not try to prove arbitrary kernel safety.

## Design Question

The implementation choice is whether to hard-code this as an L0c DeltaNet guard or generalize it into a strategy-driven forbid list parser.

Recommended near-term choice: hard-code a small DeltaNet L0c preflight guard first.

Reasoning:

- The known failures are specific to the current DeltaNet kernel surface.
- A hard-coded guard is faster to test and less likely to misparse natural language.
- The strategy brief can still explain the rule to the agent, but the controller becomes authoritative.
- If this pattern repeats across targets, promote the guard to a structured strategy schema later.

## Relaunch Criteria

Before relaunching L0c:

- Commit the preflight guard and focused tests.
- Verify that candidate `001`'s current patch would be rejected without runtime restart.
- Verify `mutations_rejected.tsv` is written immediately for preflight rejection.
- Verify normal allowed patches still proceed to `apply_and_test`.

After relaunch:

- Monitor candidate `001` for whether it stays inside the ranked safe targets.
- If the agent repeats a forbidden family, the round should reject it quickly and continue to candidate `002`.
- If the guard rejects most attempts, revisit the prompt and candidate target ranking before spending more canary time.

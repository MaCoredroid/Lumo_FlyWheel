# Evaluator Contract

## Scope
This evaluator scores one solver attempt against `task_spec.md` for the `pr-thread-contract-remediation` family.

## Required Solver Inputs
- `task_spec.md`
- `codex/config.toml`
- `skills/review-thread-remediation/SKILL.md`

## 100-Point Breakdown
- `25` Review-state triage accuracy
  - identifies actionable vs resolved vs outdated review state
  - avoids acting on stale guidance
- `20` Proposed remediation correctness
  - targets handler, service, and serializer surfaces coherently
  - preserves explicit request-side `owner` semantics
- `15` Ordering and serialization regression coverage
  - addresses omitted-vs-null behavior across output paths
  - preserves original relative ordering for equal-count buckets
- `15` Release-note and contract communication quality
  - updates the required release note surface
  - explains backward-compatible response semantics
- `15` Reviewer-reply quality
  - replies keyed to actionable thread ids only
  - cites concrete evidence instead of generic closure text
- `10` Constraint discipline
  - narrow patch scope
  - no action on resolved or outdated suggestions

## Caps
- Cap at `20/100` if the attempt does not ground its triage in thread state or explicit thread ids.
- Cap at `25/100` if the attempt never names concrete code surfaces in `queue_api/**`.
- Cap at `15/100` if the attempt replies to resolved or outdated threads, or treats them as actionable.
- Cap at `20/100` if the attempt omits regression coverage for ordering or serializer-path consistency.

## Scoring Procedure
1. Score each category independently from the written attempt output.
2. Apply all relevant caps after the raw score is totaled.
3. Record the final score, capped score explanation, and pass/fail hardness judgment in `benchmark_run.md`.

## Evidence Rules
- Award full triage points only when the solver distinguishes thread state explicitly.
- Award remediation points only when the solver names plausible file-level changes rather than broad rewrite language.
- Award reply points only when the solver supplies thread-specific response text or structure.
- Do not award release-note points for `CHANGELOG.md` alone when the task requires `docs/release_notes.md`.

## Hardness Target
- Calibration target for a naive `GPT-5.4/high` solver: about `20/100`
- Upper guardrail: if a naive solver appears to score above `30/100`, harden the task and rerun
- Lower guardrail: if a serious solver falls below `10/100` because the blueprint is incoherent, clarify the task and rerun

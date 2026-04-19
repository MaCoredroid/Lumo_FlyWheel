# Evaluator Contract

## Scope
This evaluator scores one solver attempt against `task_spec.md` for the `codex-config-skill-remediation` family.

## Required Solver Inputs
- `task_spec.md`
- `codex/config.toml`
- `skills/codex-route-contract/SKILL.md`

## 100-Point Breakdown
- `25` Review-state grounding
  - treats `review/pr_077_threads.json` as authoritative
  - resists stale guidance in `review/review_summary.md`
- `25` Workflow-local config correction
  - fixes only the `release-brief` route and policy scope
  - keeps browser and computer-use disabled
- `20` Skill-doc correction quality
  - updates examples to the local route
  - removes browser-first or externally routed guidance
- `15` Rollout-doc and validation quality
  - explains the route change and disabled surfaces
  - includes a concrete reviewer validation command
- `15` Integrity and scope discipline
  - does not weaken validator or visible tests
  - does not broaden the global allowlist

## Caps
- Cap at `20/100` if the attempt is not grounded in thread state from `pr_077_threads.json`.
- Cap at `15/100` if the attempt broadens global permissions instead of fixing the workflow-local route.
- Cap at `10/100` if the attempt weakens or bypasses the validator.
- Cap at `20/100` if the attempt updates docs only and leaves config semantics unresolved.

## Scoring Procedure
1. Score the five categories from the solver's written attempt output.
2. Apply caps after raw scoring.
3. Record raw score, caps, and final score in `benchmark_run.md`.

## Evidence Rules
- Full grounding points require explicit treatment of stale vs final review state.
- Full config points require workflow-local edits, not broad config churn.
- Full skill points require coherent examples, not pasted contract fragments.
- Full rollout points require a concrete validation command.

## Hardness Target
- Calibration target for a naive `GPT-5.4/high` solver: about `20/100`
- Upper guardrail: if a naive solver appears to score above `30/100`, harden the task and rerun
- Lower guardrail: if a serious solver falls below `10/100` because the blueprint is incoherent, clarify the task and rerun

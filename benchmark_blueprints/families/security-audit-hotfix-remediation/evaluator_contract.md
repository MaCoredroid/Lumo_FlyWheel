# Evaluator Contract

## Scope
This evaluator scores one solver attempt against `task_spec.md` for the `security-audit-hotfix-remediation` family.

## Required Solver Inputs
- `task_spec.md`
- `codex/config.toml`
- `skills/audit-hotfix-triage/SKILL.md`

## 100-Point Breakdown
- `25` Artifact-grounded audit triage
  - classifies SARIF and AppSec items using identifiers or rule ids
  - separates actionable from stale or false-positive findings
- `25` Traversal containment correctness
  - percent-decoding, normalization, resolved-root ancestry check
  - rejects absolute, drive-qualified, and escaped paths
- `15` Legitimate behavior preservation
  - preserves nested tenant-root exports
  - avoids breaking legitimate request samples
- `15` Deploy-note quality
  - includes rollback trigger, scope, operator verification
  - ties behavior claims to request samples
- `10` Regression coverage
  - encoded, double-encoded, mixed-slash, and symlink escape cases
- `10` Scope discipline
  - no broad refactor
  - no unrelated auth or dependency churn

## Caps
- Cap at `20/100` if the attempt does not cite audit identifiers or rule ids.
- Cap at `20/100` if the containment approach is string-blocklist or string-prefix based.
- Cap at `15/100` if the attempt disables nested exports instead of preserving valid tenant-root behavior.
- Cap at `20/100` if the deploy note is generic and not tied to concrete request samples.

## Scoring Procedure
1. Score each category from the solver's concrete attempt output.
2. Apply all relevant caps after raw scoring.
3. Record both raw and final score in `benchmark_run.md`.

## Evidence Rules
- Full triage points require artifact identifiers, not summary prose alone.
- Full containment points require resolved-root ancestry semantics.
- Do not award preservation points when the attempt blocks all subdirectories.
- Do not award deploy-note points for schema-like boilerplate without operational claims.

## Hardness Target
- Calibration target for a naive `GPT-5.4/high` solver: about `20/100`
- Upper guardrail: if a naive solver appears to score above `30/100`, harden the task and rerun
- Lower guardrail: if a serious solver falls below `10/100` because the blueprint is incoherent, clarify the task and rerun

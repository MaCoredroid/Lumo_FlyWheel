# Evaluator Contract: Artifact Retention Janitor

## Purpose
Evaluate whether the solver implemented a safe retention selector instead of hiding protected artifacts in dry-run output or over-preserving everything.

## Required submission surfaces
- Retention selector code
- Retention policy config
- Dry-run report artifact
- Operator note or policy doc update

## Visible gates
1. Visible tests pass for current-run preservation, policy loading, and dry-run formatting.
2. The dry-run report includes stable reason tokens for protected and eligible artifacts.

## Trusted checks
- Current runs are never emitted as deletion candidates.
- Manifest-pinned results are never emitted as deletion candidates.
- `retain_on_failure` snapshots remain protected.
- Report rows and the actual deletion candidate set agree exactly.

## Hidden checks
- Incomplete DB row with current filesystem marker must remain protected.
- Incomplete DB row for a non-current artifact must not gain blanket protection.
- Old failed run without `retain_on_failure` must become `eligible`.
- Symlink target resolution must distinguish pinned/current targets from unpinned/non-current targets.
- Manifest layout variation rejects hardcoded visible-path logic.

## 100-point scoring breakdown
- `15` points: correct diagnosis of the retention failure modes and protected-state sources.
- `25` points: observed selector fix for current, pinned, and `retain_on_failure` artifacts. Proposed-only edits score at most `5/25`.
- `15` points: observed reconciliation across DB metadata, filesystem markers, and manifest state. Proposed-only edits score at most `3/15`.
- `10` points: positive deletion behavior for hidden eligible artifacts. No executed candidate evidence yields `0/10`.
- `10` points: selector-result parity with the dry-run report. No generated report from a live decision set yields `0/10`.
- `10` points: retention policy config alignment. Proposed-only edits score at most `2/10`.
- `15` points: hidden robustness for symlink, corrupt-row, and manifest-shape cases. No executed evidence yields `0/15`.

## Hard caps
- Cap at `20` if only the report hides protected artifacts while selector behavior remains unsafe.
- Cap at `20` if the solver protects all failed or all corrupt artifacts indiscriminately.
- Cap at `30` if selector and report outputs diverge.
- Cap at `25` for diagnosis-only or plan-only submissions that do not produce an observed selector result set and dry-run artifact from the concrete workspace.

## Shortcut defenses
- Check real candidate sets, not just rendered markdown.
- Require positive deletion of at least one hidden eligible artifact.
- Reject broad monotonic “never delete anything suspicious” behavior.

## Final hardness judgment
- Current naive GPT-5.4/high outlook: under `30/100`, but narrower than the other families.
- Confidence: medium.
- Main reason: the contract now grades both false-positive avoidance and correct deletion of eligible artifacts.

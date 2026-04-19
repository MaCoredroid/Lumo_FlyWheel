# Benchmark Run

## Run 1
- `date`: `2026-04-18`
- `agent_id`: `019da331-d84e-70e3-ab23-56fb67b7d01b`
- `model`: `gpt-5.4`
- `reasoning_effort`: `high`
- `bundle_root`: `benchmark_blueprints/families/security-audit-hotfix-remediation`
- `attempt_mode`: blueprint-family bundle only; the referenced `audit/`, `artifacts/`, and `repo/` trees were not present

## Submission Summary
- The solver bounded the hotfix to export-path containment rather than broad refactor churn.
- It proposed a credible resolved-root containment fix with percent decoding, slash normalization, absolute-path rejection, drive-qualified rejection, and symlink escape handling.
- It preserved legitimate nested tenant-root exports and proposed a strong regression matrix.
- It drafted a deploy-note summary with rollback and verification concepts.
- It could not enumerate artifact ids or classify each audit item because `audit.sarif`, the AppSec note, and the triage matrix were missing from the family bundle.

## Scoring
- Artifact-grounded audit triage: `8/25`
- Traversal containment correctness: `23/25`
- Legitimate behavior preservation: `14/15`
- Deploy-note quality: `12/15`
- Regression coverage: `10/10`
- Scope discipline: `10/10`

Raw score: `77/100`

## Caps Applied
- Cap at `20/100` applied because the attempt could not cite SARIF or AppSec identifiers and therefore did not meet the evaluator's artifact-grounded triage requirement.

Final score: `20/100`

## Judgment
- Calibration target met: `yes`
- Naive `GPT-5.4/high` can describe a strong containment strategy, but the family still resists scores above `30/100` without artifact-id grounded triage.
- Rerun required: `no`

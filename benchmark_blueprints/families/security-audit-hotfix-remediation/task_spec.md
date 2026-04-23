# Task Spec: `t4-security-audit-hotfix-remediation-export-path-traversal`

## Family
- `family_id`: `security-audit-hotfix-remediation`
- `scenario_type`: `audit_remediation`

## Task Prompt
The workspace contains a seeded security audit, an AppSec review note, and a partially filled triage matrix for an export-path hotfix. Inspect `audit/audit.sarif`, `audit/appsec_review.md`, `audit/triage_matrix.csv`, the repo under `repo/`, and any optional `repo_evidence/`, `release_context/`, or `incident_context/` directories that exist in the variant. Determine which findings are actionable for this patch, implement the smallest safe remediation for the real traversal exploit, preserve legitimate nested exports under the tenant root, and document the hotfix.

The grader is artifact-grounded. Generic prose that does not cite the stable audit or AppSec identifiers will not receive triage credit.

## Workspace Bundle
- `repo/`
  - `export_api/routes.py`
  - `export_api/storage.py`
  - `export_api/auth.py`
  - `ops/deploy_hotfix.json`
  - `tests/test_routes.py`
  - `tests/test_storage.py`
- `audit/`
  - `audit.sarif`
  - `appsec_review.md`
  - `triage_matrix.csv`
  - `triage_note.json`
- `artifacts/`
  - `request_samples.json`
  - `triage_note_schema.json`
  - `deploy_note_schema.json`
- optional variant corpora:
  - `repo_evidence/`
  - `release_context/`
  - `incident_context/`

## Required Surfaces
- `review`
- `shell`
- `apply_patch`
- terminal test execution

## Validation Tooling
- `python3 bin/cnb55-hotfix-artifacts validate audit/triage_note.json repo/ops/deploy_hotfix.json`
- `make test`

## Expected Deliverables
- narrow code patch in `repo/export_api/storage.py` or `repo/export_api/routes.py`
- `audit/triage_note.json`
- `repo/ops/deploy_hotfix.json`
- optional `repo/tests/test_hotfix_regressions.py`

## Visible Checks (30 points total)
- `make test` passes against the shipped visible tests
- the triage and deploy-note JSON documents validate against the shipped schemas
- if a regression file is added, it must be importable and picked up by `make test`

## Hidden Checks (70 points total)
- triage note references every SARIF or AppSec item by stable identifier
- dispositions match the artifact-grounded actionability of each item
- the code rejects traversal through percent-encoded separators, mixed slash styles, absolute paths, drive-qualified inputs, normalized dot segments, double-encoded separators, and symlink escapes
- the code still allows legitimate nested exports under the tenant root
- the deploy note cites request-sample ids, includes rollback trigger, impact scope, operator verification steps, and variant-gated release or incident acknowledgements when present
- the patch stays within hotfix scope and does not broaden into auth, dependency, or archive churn

## Red-Team Traps
- blocking only the literal string `".."` and leaving normalized or encoded traversal live
- relying on string-prefix ancestry instead of resolved-root ancestry
- disabling nested exports entirely
- marking every finding as fixed or actionable without artifact-grounded triage
- acknowledging neither the release freeze nor the prior rollback when those variant corpora exist

## Saturation and Renewal Plan
Trigger saturation review when the mean `P_benchmark` stays above `80` for two consecutive probe rounds. Renewal queue:
1. add a variant where the symlink escape arrives mid-session through a checked-in patch review
2. add a variant with contradictory AppSec follow-up notes that must be reconciled explicitly
3. retire `v1-clean-baseline` if it becomes a pure floor check

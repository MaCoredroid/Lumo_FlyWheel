# Task Spec: `t4-security-audit-hotfix-remediation-export-path-traversal`

## Family
- `family_id`: `security-audit-hotfix-remediation`
- `scenario_type`: `audit_remediation`

## Task Prompt
The repo includes a seeded security audit and an AppSec review note after a hotfix request. Use `audit/audit.sarif`, `audit/appsec_review.md`, and `audit/triage_matrix.csv` to determine which findings are actionable for this patch and which are non-actionable. Implement the smallest safe fix for the real exploit, preserve expected export behavior for legitimate callers, and add a deploy note with rollback guidance. Do not broaden the patch into a general refactor. The grader will require artifact-grounded triage evidence rather than accepting generic remediation text.

## Workspace Bundle
- `repo/`
  - `export_api/routes.py`
  - `export_api/storage.py`
  - `export_api/auth.py`
  - `tests/test_routes.py`
  - `tests/test_storage.py`
  - `ops/deploy_hotfix.md`
  - `pyproject.toml`
- `audit/`
  - `audit.sarif`
  - `appsec_review.md`
  - `triage_matrix.csv`
- `artifacts/`
  - `request_samples.json`
  - `deploy_note_schema.json`

## Seeded Review Or Audit Artifact
The audit bundle mixes actionable and non-actionable findings across SARIF output, a human AppSec review, and a prefilled triage matrix. The spec intentionally does not pre-label which entries are real, stale, or false positive. The solver must classify each item from the artifacts and limit the patch to the true hotfix scope.

## Required Surfaces
- `review`
- `shell`
- `apply_patch`
- terminal test execution

## Expected Deliverables
- narrow code patch in route or storage normalization
- regression tests for traversal attempts and legitimate nested export paths
- `audit/triage_note.md` classifying every SARIF or AppSec item with artifact identifier, affected file or rule, disposition, and one-sentence rationale
- `ops/deploy_hotfix.md` satisfying `artifacts/deploy_note_schema.json`

## Grader Contract
Phase 2 functional checks:
- run the repo test target and require passing route and storage tests
- validate the deploy note against `artifacts/deploy_note_schema.json`

Phase 3 trusted checks:
- require `audit/triage_note.md` to reference each SARIF or AppSec item by stable identifier or rule id; generic prose without artifact identifiers does not count
- resolve the candidate export path by percent-decoding once, normalizing slash styles, joining against the resolved tenant root, and verifying ancestry against the resolved tenant root rather than by string prefix
- preserve valid subdirectory exports under the tenant root
- reject traversal through encoded separators, mixed slash styles, dot-segment normalization, absolute paths, drive-qualified inputs, and symlink escapes that resolve outside the tenant root
- confirm the triage note classifies non-actionable findings with artifact-grounded rationale instead of blanket “fixed” language
- confirm the deploy note includes rollback trigger, impact scope, operator verification step, and at least one behavior-preservation claim tied to `artifacts/request_samples.json`
- any failure of trusted traversal containment or artifact-grounded triage evidence caps total score at `20/100`

Hidden checks:
- percent-encoded and double-encoded traversal payloads
- Windows-style separators and drive-qualified inputs on POSIX
- absolute-path inputs and mixed slash-style normalization
- real symlink escape fixtures whose resolved target leaves the tenant root
- checksum guard on audit artifacts, request samples, and visible tests
- triage-note evidence checks for exact SARIF or AppSec identifiers and dispositions

## Red-Team Traps
- ban only the literal string `".."` and miss normalized traversal
- disable nested exports entirely to make the test pass
- silently broaden the patch into auth or dependency churn unrelated to the exploit
- mark every SARIF finding as fixed without a triage rationale
- write a deploy note with generic hotfix text and no rollback trigger

## Quality Gate
- The task must force the solver to separate one real exploit from surrounding audit noise.
- Hidden traversal cases must prevent simplistic string-blocklist fixes from earning partial credit too easily.
- The hotfix note must be structured and schema-validated so boilerplate cannot pass.
- Post-probe hardening:
  - Removed the leaked statement that only one finding is real and forced artifact-grounded classification.
  - Added resolved-root ancestry and symlink escape requirements so string-prefix fixes fail.
  - Added a hard score cap when trusted traversal or triage checks fail.
- Adversarial probe verdict before hardening: `no`.
- Naive GPT-5.4/high under-30 target after hardening: `yes`, only if failing trusted traversal or artifact-grounded triage checks caps the score below 30.
- Latest benchmark-run calibration: `20/100` on `2026-04-18` against the family-local blueprint bundle.

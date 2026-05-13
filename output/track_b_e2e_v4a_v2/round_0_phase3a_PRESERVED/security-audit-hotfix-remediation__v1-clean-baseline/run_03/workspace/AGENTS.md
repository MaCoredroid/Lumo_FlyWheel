# security-audit-hotfix-remediation

Fix the real export-path traversal bug using the workspace evidence only.

Required:
- inspect `audit/audit.sarif`, `audit/appsec_review.md`, `audit/triage_matrix.csv`, `artifacts/request_samples.json`, and the repo under `repo/`
- if `repo_evidence/`, `release_context/`, or `incident_context/` exists, use it
- implement the smallest safe fix in `repo/export_api/storage.py` or `repo/export_api/routes.py`
- write `audit/triage_note.json`
- write `repo/ops/deploy_hotfix.json`
- you may add `repo/tests/test_hotfix_regressions.py`
- run `python3 bin/cnb55-hotfix-artifacts validate audit/triage_note.json repo/ops/deploy_hotfix.json`
- run `make test`

Do not:
- modify shipped audit artifacts, schemas, request samples, visible tests, or variant corpora
- broaden the patch into auth, dependency, archive, or unrelated config changes
- replace the hotfix with a literal-string blocklist or a string-prefix ancestry check

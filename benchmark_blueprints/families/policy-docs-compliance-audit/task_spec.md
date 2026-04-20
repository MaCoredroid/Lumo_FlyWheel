# Policy / Docs Compliance Audit

## Task Identity
- `task_id`: `policy-docs-compliance-audit/compliance-finding-pack`
- `family_id`: `policy-docs-compliance-audit`
- `scenario_type`: `evidence_grounded_research`

## Task Prompt
Audit whether a proposed operational workflow complies with a frozen local policy corpus and implementation notes. Return only violations or green findings that you can prove from the corpus, and classify each finding as policy violation, documentation drift, or compliant exception.

## Workspace Bundle
- `corpus/policies/`, `corpus/runbooks/`, `corpus/implementation_notes/`
- `queries/workflow_request.md`
- `artifacts/finding_schema.json`
- `tests/test_compliance_audit.py`

## Seeded Research Ambiguity
- One runbook conflicts with the higher-priority policy doc.
- A compliant exception exists but only for a narrow environment class.
- One implementation note uses policy-like language without actually carrying authority.

## Required Surfaces
- `shell`
- corpus search
- compliance classification
- finding-pack writing

## Expected Deliverables
- Compliance audit findings.
- Authority-ranked evidence per finding.
- Explicit exception handling where applicable.

## Grader Contract
- Visible checks:
  - `pytest -q tests/test_compliance_audit.py`
- Hidden checks:
  - Authority ordering is correct.
  - Violations and compliant exceptions are classified correctly.
  - Non-authoritative notes are not over-weighted.

## Quality Gate
- Target naive score: `20/100`.

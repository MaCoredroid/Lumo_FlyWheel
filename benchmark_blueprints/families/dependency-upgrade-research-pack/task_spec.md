# Dependency Upgrade Research Pack

## Task Identity
- `task_id`: `dependency-upgrade-research-pack/upgrade-decision-pack`
- `family_id`: `dependency-upgrade-research-pack`
- `scenario_type`: `evidence_grounded_research`

## Task Prompt
Prepare an upgrade research pack for a dependency jump using a frozen local corpus of changelogs, migration guides, CVE notes, internal code-search results, and prior incident notes. Recommend whether to upgrade now, defer, or stage behind preparatory work, and justify the recommendation with concrete evidence.

## Workspace Bundle
- `corpus/changelogs/`, `corpus/migrations/`, `corpus/cves/`, `corpus/internal_search/`
- `queries/upgrade_request.md`
- `artifacts/risk_matrix_schema.json`
- `tests/test_upgrade_pack.py`

## Seeded Research Ambiguity
- Security urgency and migration cost point in different directions.
- A stale internal note overstates a breaking change that was later reverted.
- Correct recommendation depends on both external release evidence and internal call-site evidence.

## Required Surfaces
- `shell`
- corpus search
- risk synthesis
- recommendation writing

## Expected Deliverables
- Upgrade recommendation.
- Risk matrix with evidence.
- Staged execution note if immediate upgrade is not justified.

## Grader Contract
- Visible checks:
  - `pytest -q tests/test_upgrade_pack.py`
- Hidden checks:
  - Recommendation matches the benchmark-owned resolution.
  - Internal and external evidence are both used correctly.
  - Stale reverted risk is not treated as current fact.

## Quality Gate
- Target naive score: `20/100`.

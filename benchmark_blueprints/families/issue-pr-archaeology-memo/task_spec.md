# Issue / PR Archaeology Memo

## Task Identity
- `task_id`: `issue-pr-archaeology-memo/root-cause-brief`
- `family_id`: `issue-pr-archaeology-memo`
- `scenario_type`: `evidence_grounded_research`

## Task Prompt
Investigate a long-running regression using only a frozen local archive of issues, pull requests, release notes, and commit excerpts. Produce a root-cause memo that identifies the earliest causal change, explains why two later suspects are downstream symptoms, and recommends the smallest safe remediation path.

## Workspace Bundle
- `archive/issues/`, `archive/prs/`, `archive/commits/`: frozen project-history slices.
- `queries/regression_report.md`: user-visible symptom report.
- `artifacts/gold_timeline.json`: grader-owned causal timeline.
- `tests/test_archaeology_memo.py`: visible structure checks.

## Seeded Research Ambiguity
- The loudest PR discussion is not the originating change.
- One issue thread contains a correct diagnosis but the wrong remediation.
- Two suspect commits are temporally adjacent and easy to conflate.

## Required Surfaces
- `shell`
- corpus search
- timeline synthesis
- evidence-grounded memo writing

## Expected Deliverables
- Root-cause memo with a causal timeline.
- Explicit rejection of at least two plausible downstream suspects.
- Narrow remediation recommendation.

## Grader Contract
- Visible checks:
  - `pytest -q tests/test_archaeology_memo.py`
- Hidden checks:
  - Earliest causal change is identified correctly.
  - Downstream suspects are rejected with evidence.
  - Timeline and memo claims align to the frozen archive.
- Trusted checks:
  - Recommendation scope is consistent with the root-cause evidence.

## Quality Gate
- Target naive score: `20/100`.
- Shallow chronology summaries should not clear `30`.

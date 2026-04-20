# Proposal Ranking / Manager Judgment

## Task Identity
- `task_id`: `proposal-ranking-manager-judgment/proposal-selection-brief`
- `family_id`: `proposal-ranking-manager-judgment`
- `scenario_type`: `strategic_management`

## Task Prompt
Rank several implementation proposals for the same repo objective using frozen local evidence: code excerpts, test gaps, performance notes, staffing constraints, and rollout risk. Choose the best proposal, reject the weaker options, and justify the decision in a concise manager brief.

## Workspace Bundle
- `proposals/`
- `repo_evidence/`
- `artifacts/gold_ranking.json`
- `tests/test_ranking_brief.py`

## Seeded Strategic Ambiguity
- The fastest proposal is not the safest.
- The most ambitious proposal creates avoidable rollout risk.
- One weaker proposal looks attractive if staffing constraints are ignored.

## Required Surfaces
- `shell`
- evidence review
- strategic ranking
- concise decision writing

## Expected Deliverables
- Ranked proposal brief.
- Rejection reasoning for non-selected options.
- Risk-aware recommendation.

## Grader Contract
- Visible checks:
  - `pytest -q tests/test_ranking_brief.py`
- Hidden checks:
  - Ranking matches the gold managerial ordering.
  - Staffing and rollout constraints are incorporated.
  - Rejection reasoning targets the true weaknesses.

## Quality Gate
- Target naive score: `20/100`.

# RFC / Spec Retrieval Grounding

## Task Identity
- `task_id`: `rfc-spec-retrieval-grounding/pinned-citation-answer`
- `family_id`: `rfc-spec-retrieval-grounding`
- `scenario_type`: `evidence_grounded_research`

## Task Prompt
Answer a maintainer question about a proposed protocol or API behavior change using only the frozen benchmark corpus. Return a concise decision memo with pinned citations, note any contradictory evidence, and distinguish normative language from commentary. Do not answer from memory or intuition when the corpus disagrees.

## Workspace Bundle
- `corpus/specs/`: frozen RFCs, internal design notes, release notes, and draft amendments.
- `queries/request.md`: the maintainer question plus one misleading shorthand summary.
- `artifacts/expected_claims.json`: grader-owned claim inventory and hard negatives.
- `docs/citation_style.md`: required citation format.
- `tests/test_claim_alignment.py`: visible citation-shape checks.

## Seeded Research Ambiguity
- A high-ranking draft note paraphrases the true rule but weakens one normative requirement.
- One stale migration memo uses legacy terminology that looks authoritative.
- Correct resolution requires synthesizing at least two documents, not quoting one sentence in isolation.

## Required Surfaces
- `shell`
- corpus search
- evidence synthesis
- citation-grounded writing

## Expected Deliverables
- Evidence-grounded answer memo.
- Claim-by-claim citations tied to the frozen corpus.
- Short contradiction note covering the most plausible distractor.

## Grader Contract
- Visible checks:
  - `pytest -q tests/test_claim_alignment.py`
- Hidden checks:
  - Required supporting documents are cited.
  - Hard-negative documents are not used as primary support.
  - Normative claims are grounded at sentence or section level.
- Trusted checks:
  - Final answer matches the benchmark-owned resolution key.
  - Citations are pinned and reproducible against the local corpus.

## Quality Gate
- Target benchmark-design outcome: naive `gpt-5.4/high` around `20/100`.
- Over-credit for generic summaries is disallowed.
- Scores above `30` require hidden support-document and hard-negative checks to pass.

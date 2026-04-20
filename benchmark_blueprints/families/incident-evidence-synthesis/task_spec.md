# Incident Evidence Synthesis

## Task Identity
- `task_id`: `incident-evidence-synthesis/postmortem-packet`
- `family_id`: `incident-evidence-synthesis`
- `scenario_type`: `evidence_grounded_research`

## Task Prompt
Assemble a concise incident evidence packet from frozen logs, tickets, timeline snippets, and remediation notes. The packet must identify the triggering condition, the failed guardrail, and the highest-confidence follow-up action while explicitly flagging any unresolved ambiguity.

## Workspace Bundle
- `corpus/logs/`, `corpus/tickets/`, `corpus/timeline/`, `corpus/remediation/`
- `queries/incidence_request.md`
- `artifacts/gold_findings.json`
- `tests/test_incident_packet.py`

## Seeded Research Ambiguity
- One ticket has the right symptom but the wrong trigger.
- A remediation note was written before the final root cause was known.
- Correct synthesis requires combining operational and code-adjacent evidence.

## Required Surfaces
- `shell`
- corpus search
- incident synthesis
- evidence packet writing

## Expected Deliverables
- Incident evidence packet.
- Ranked findings with confidence.
- Explicit unresolved ambiguity section.

## Grader Contract
- Visible checks:
  - `pytest -q tests/test_incident_packet.py`
- Hidden checks:
  - Triggering condition is identified correctly.
  - Failed guardrail is named correctly.
  - Uncertainty is stated only where warranted.

## Quality Gate
- Target naive score: `20/100`.

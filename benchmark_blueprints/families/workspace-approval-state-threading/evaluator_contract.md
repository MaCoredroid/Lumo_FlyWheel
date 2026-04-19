# Evaluator Contract

## Scope
- Family: `workspace-approval-state-threading`
- Task: `cnb55-core-workspace-approval-state-threading-admin-ui`
- Runtime class: small monorepo with backend, CLI export, React admin UI, config, and runbook

## Evaluation Intent
Measure whether the solver can thread a new field across backend, API, CLI, frontend, config, and docs without relying on serializer aliases or UI fabrication. The evaluator should reward true cross-surface consistency and backward compatibility.

## Required Submission Artifacts
- Backend and frontend code patch
- Tests for legacy-row fallback and UI rendering
- `config/defaults.toml` update
- Runbook update and screenshot artifact reference
- Short rollout note

## Scoring Model
- `100` total points, awarded by category:
  - `35`: backend plus serializer correctness
  - `20`: CLI and API consistency
  - `20`: frontend and screenshot-backed UI correctness
  - `15`: config plus runbook alignment
  - `10`: backward compatibility and anti-shortcut robustness

## Score Caps
- Cap at `10/100` for docs-only, screenshot-only, or frontend-mock-only submissions.
- Cap at `20/100` for serializer-only or UI-only threading.
- Cap at `20/100` if the solver proposes reasonable edits but cannot prove service/API/CLI consistency.
- Cap at `25/100` if `approval_state` is threaded visibly but legacy-row fallback is missing or vague.
- Cap at `30/100` if config or runbook remain stale.
- Cap at `50/100` if code is correct but screenshot or rollout evidence is missing.

## Visible Gates
- Backend serializer tests pass
- Frontend table tests pass
- Screenshot harness produces the expected image
- Same-dataset consistency test compares service, API, and CLI export values

## Hidden Gates
- Legacy rows missing `approval_state` still render via documented fallback
- API and CLI export must agree for mixed datasets
- Screenshot diff checks column label, badge text, and filtered row state
- Grep or AST guard rejects stale alias logic that papers over missing schema threading
- Config and runbook must use `approval_state`, not `approval_mode`

## Integrity And Shortcut Checks
- Reject frontend-only fabricated state
- Reject serializer aliasing that merely mirrors `risk_level`
- Reject screenshot-name hardcoding without visual correctness
- Reject removal of fallback coverage

## Variant Hardness Notes
- `V1`: straightforward rollout, but cross-surface consistency is still required
- `V2`: stale mocks and warnings should not change the target behavior
- `V3`: unrelated local edits must survive
- `V4`: screenshot and rollout note are mandatory
- `V5`: follow-up injects inconsistency between serializer and UI

## Current Hardness Judgment
- Actual recorded solver run: `20/100`
- Naive `gpt-5.4/high` above `30/100`: `unlikely under current rubric`

# Task Spec: `t4-incident-retro-runbook-closure-cache-drain-escalation`

## Family
- `family_id`: `incident-retro-runbook-closure`
- `scenario_type`: `operations_review_remediation`

## Task Prompt
An incident retrospective concluded that the on-call runbook, the helper command, and the existing Codex thread-wake automation used during queue-drain incidents drifted apart. Use `retro/action_items.json` as the authoritative source of truth for the required verification command, verification sequence, and escalation target; use `retro/incident_2026_02_14_retro.md` and `retro/operator_timeline.txt` only as supporting context. Update the runbook, make the smallest behavior-preserving script change required, and repair the existing automation so its next wake-up follows the documented sequence. Preserve unrelated operational content.

## Workspace Bundle
- `repo/`
  - `runbooks/queue_drain.md`
  - `scripts/queue_drain_helper.py`
  - `.codex/automations/queue-drain-watch/automation.toml` (the wake-up prompt text is stored inline in this file; there is no separate prompt file)
  - `tests/test_queue_drain_helper.py`
  - `tests/test_runbook_contract.py`
  - `ops/notes/queue_drain_followup.md`
- `retro/`
  - `incident_2026_02_14_retro.md`
  - `action_items.json`
  - `operator_timeline.txt`
- `artifacts/`
  - `automation_expectations.json`
  - `ops_note_schema.json`

## Seeded Review Or Audit Artifact
The retro materials indicate:
- the runbook skipped the post-drain verification command that would have caught the lingering stuck shard
- the helper script still uses the old verification subcommand name
- the automation prompt wakes the thread but asks for the wrong verification step and stale escalation target
- two retro notes are informational only and should not trigger repo changes

## Required Surfaces
- `automation`
- `shell`
- `apply_patch`
- terminal test execution

## Expected Deliverables
- runbook patch
- narrow helper-script patch
- repaired automation TOML
- short operator-facing follow-up note

## Grader Contract
Phase 2 visible checks:
- run `pytest -q repo/tests/test_queue_drain_helper.py repo/tests/test_runbook_contract.py`
- validate automation fields and inline prompt text against `artifacts/automation_expectations.json`
- validate the follow-up note against `artifacts/ops_note_schema.json`

Phase 3 trusted checks:
- derive the expected verification command, verification sequence, and escalation target from `retro/action_items.json`
- reject solutions that rely on prose-only retro interpretations when they conflict with `retro/action_items.json`
- `repo/runbooks/queue_drain.md`, `repo/scripts/queue_drain_helper.py`, and the automation prompt must all reference exactly the same non-retired verification command
- reject solutions that mention both the retired and current verification commands as acceptable operator steps
- reject solutions where the automation fields validate but the prompt still asks for the stale verification step or stale escalation target
- reject solutions that silently change automation cadence or destination beyond the requested repair
- reject solutions that modify tests or artifact expectation files

Scoring caps:
- cap at `25/100` if any one of runbook, helper, or automation prompt remains stale, even if visible tests and schema validation pass
- cap at `15/100` if the helper still preserves the retired subcommand as the primary documented or operator path
- cap at `10/100` if the solution only updates documentation, note text, or automation fields without repairing helper-script behavior

Hidden checks:
- follow-up test where the script alias works locally but the runbook text still points to the retired command
- automation prompt inspection to ensure it describes the corrected verification step, not just the schedule
- checksum guard on retro artifacts and visible contract tests

## Red-Team Traps
- patch only the runbook and ignore the helper or automation
- change the automation schedule instead of the stale task prompt
- add both old and new commands to the runbook without choosing the correct one
- copy retro bullets into the follow-up note without resolving the command mismatch
- rewrite unrelated operational sections to farm easy doc diff

## Quality Gate
- This task must stay multi-surface; any solution that does not correctly repair all three authoritative surfaces (`runbook`, `helper`, `automation prompt`) is capped below `30/100`.
- Hidden checks must grade against `retro/action_items.json`, not self-consistency alone.
- The automation prompt text must be graded as a first-class surface, not only TOML cadence or destination fields.
- Post-probe hardening:
  - Made `retro/action_items.json` the authoritative grading source.
  - Added score caps for partial alignment so doc-only or automation-only fixes fail closed.
  - Declared the inline automation prompt as an explicit graded surface.
- Adversarial probe verdict before hardening: `no`.
- Naive GPT-5.4/high under-30 target after hardening: `yes`, if the three-surface alignment and score-cap rules are enforced.
- Latest benchmark-run calibration: `20/100` on `2026-04-18` against the family-local blueprint bundle.

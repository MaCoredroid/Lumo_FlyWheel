# Transcript Merge Regression

## Task Metadata
- `task_id`: `cnb55-core-transcript-merge-regression-reducer-diagnosis`
- `family_id`: `transcript-merge-regression`
- `scenario_type`: `investigate_then_fix`

## Task Prompt
Saved Codex session replays are dropping or duplicating tool output blocks after transcript merge. Investigate the reducer logic, identify the real merge invariant that is being violated, and repair the implementation without regressing the incident summary report. Include a short incident note that explains the bug and the safe boundary of the fix. Do not silence the symptom by filtering duplicate lines after rendering.

## Workspace Bundle
- Small telemetry repo rooted at `/workspace`.
- `replay/merge.py` combines JSONL transcript fragments.
- `replay/render.py` formats merged sessions for operator review.
- `artifacts/sessions/*.jsonl` includes visible failing examples and noisy debug traces.
- `tests/test_merge.py`, `tests/test_render.py`, `tests/test_incident_summary.py`.
- `reports/incidents/transcript-merge.md` is the operator-facing note that is now stale.

## Seeded Breakage
- Merge logic keys tool output blocks by `(role, tool_name)` instead of stable event identity.
- Debug fragments arriving after `response.completed` are mistakenly treated as a new renderable assistant block.
- Incident summary counts merged tool outputs after rendering, so suppressing visible duplicates changes the report but hides the root bug.
- One visible session contains benign duplicate debug lines that look suspicious but are not the real defect.
- Incident note still describes the previous dedupe strategy as safe.

## Required Surfaces
- `shell`
- `apply_patch`
- transcript artifact inspection
- terminal tests

## Expected Deliverables
- Reducer or merge-layer code patch.
- Tests protecting the stable event identity rule and incident summary invariants.
- Incident note update documenting the bug and why output filtering is not an acceptable fix.

## Grader Contract
- Visible gate:
  - `pytest -q tests/test_merge.py tests/test_render.py tests/test_incident_summary.py`
- Hidden checks:
  - Withheld session where two tool outputs share tool name and role but differ in event identity.
  - Interleaved fragment case where both same-name tool events must survive merge under distinct stable identities.
  - Debug-only fragment arriving after completion must not create a new render block.
  - Legitimate deferred tool output after completion must still survive while separate debug-only post-completion noise is ignored.
  - Incident summary totals must remain stable relative to the trusted merged representation.
  - Grep check rejects post-render duplicate filtering as the primary fix.
- Fail conditions:
  - Render-layer suppression instead of merge invariant repair.
  - Fixture-specific filtering of the visible debug lines.
  - Tests green but incident summary semantics drift.

## Red-Team Traps
- Filter duplicate rendered lines after merge.
- Drop all post-completion fragments, including legitimate deferred tool output.
- Re-key by tool name plus sequence index derived from visible fixtures only.
- Update the incident note to justify the wrong fix.
- Delete the incident summary regression test.

## Quality Gate
- Probe target:
  - A naive `gpt-5.4/high` solver should land around `20/100`, not `30/100`, unless it identifies the stable event identity invariant and proves the reducer fix.
- Hardening applied after adversarial solver run:
  - The prompt forbids symptom-only render filtering.
  - Hidden checks separate real duplicate corruption from benign debug noise.
  - Incident summary coupling makes shallow replay cleanup insufficient.
  - Hidden cases now distinguish legitimate deferred post-completion tool output from debug-only noise, which blocks the blunt "drop everything after completion" shortcut.
- Actual run result:
  - Child subagent solver attempt using `gpt-5.4/high` scored `20/100` under the family rubric.
  - The run named the right invariant but still lacked code, transcript-backed verification, and summary proof.
  - Final assessment: `target hardness met; naive gpt-5.4/high stays around 20/100`.

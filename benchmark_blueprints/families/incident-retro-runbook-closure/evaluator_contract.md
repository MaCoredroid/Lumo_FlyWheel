# Evaluator Contract

## Scope

This evaluator scores one solver attempt against the `incident-retro-runbook-closure` family contract.

## Required Solver Inputs

- `AGENTS.md`
- `retro/action_items.json`
- the four writable surfaces under `repo/`

## 100-Point Breakdown

- `25` Visible pytest contract
  - helper emits the authoritative verification command
  - runbook contains the authoritative verification sequence
  - retired command is not presented as an acceptable primary step
- `20` Helper repair correctness
  - helper promotes the current command to the primary operator path
  - repair stays narrow and behavior-preserving
- `20` Runbook correctness
  - command, sequence, and escalation language match `retro/action_items.json`
- `20` Automation prompt fidelity
  - prompt text uses the current command and escalation target
  - cadence and destination remain unchanged
- `10` Follow-up note discipline
  - actionable items only
  - informational notes stay out of the repo note
- `5` Constraint / invariants discipline
  - no immutable slice changes
  - no unexpected files or test tampering

## Caps

- Cap at `20/100` with `ungrounded_authority` if supporting prose beats `retro/action_items.json`.
- Cap at `25/100` with `partial_surface_repair` if all three authoritative surfaces are not repaired.
- Cap at `15/100` with `dual_command_path` if both retired and current commands remain acceptable.
- Cap at `10/100` with `doc_only_closure` if docs or note are updated but helper behavior remains stale.
- Cap at `35/100` with `informational_note_promotion` if informational-only retro notes are promoted into the runbook, helper, or automation prompt.
- Cap at `30/100` with `objective_drift` on v4+ if the solver deprioritizes required post-drain verification.
- Cap at `25/100` with `schedule_drift` if the automation cadence or destination changes.

## Scoring Procedure

1. Run visible repo tests.
2. Apply deterministic hidden checks over the helper, runbook, automation prompt, and follow-up note.
3. Run trusted final-state integrity checks.
4. Apply post-aggregation ceilings.
5. Emit dual-band result fields and the 5-slot milestone vector.

## Evidence Rules

- `retro/action_items.json` is the only authoritative source of truth for command, sequence, and escalation target.
- Supporting prose may explain *why* the action item exists, but it cannot override the JSON action item.
- The automation prompt text is a first-class graded surface; field-only TOML edits are insufficient.
- The follow-up note may summarize only actionable repairs. Informational retro notes must not be promoted into required repo changes.

## Hidden-Test Floors

- Oracle repair: `>= 90`
- Empty workspace: `0`
- Schedule-drift shortcut: `<= 25`
- Delete-tests adversarial: integrity must fire and `M_training == 0`

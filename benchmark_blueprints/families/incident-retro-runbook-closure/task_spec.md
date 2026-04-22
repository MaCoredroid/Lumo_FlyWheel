# `incident-retro-runbook-closure` Task Spec

**Track:** 04 — Review & Remediation
**Family id:** `incident-retro-runbook-closure`
**Spec version:** CNB-55 v1.0
**Variants:** 5 (v1 through v5)

## Task Prompt (canonical)

You are repairing the closure workflow for a queue-drain incident. The incident retro found that four surfaces drifted apart:

1. the operator runbook
2. the helper script that prints the post-drain verification command
3. the existing thread-wake automation prompt
4. the short operator follow-up note

Treat `retro/action_items.json` as the only authoritative source for:

- the required verification command
- the verification sequence
- the escalation target
- which retro notes are informational-only and must **not** become repo changes

`retro/incident_2026_02_14_retro.md`, `retro/operator_timeline.txt`, and any variant-specific `release_context/` or `incident_context/` files are supporting context only. They may contain stale wording, old command names, or stale escalation targets. Reconcile them in favor of `retro/action_items.json`.

Fix the repo by editing only the intended repair surfaces under `repo/`:

- `repo/runbooks/queue_drain.md`
- `repo/scripts/queue_drain_helper.py`
- `repo/.codex/automations/queue-drain-watch/automation.toml`
- `repo/ops/notes/queue_drain_followup.md`

Do not modify tests, retro artifacts, or expectation files.

## Scenario Type

`operations_review_remediation` — a multi-surface operational repair where the authoritative retro artifact must overrule stale prose and stale automation text.

## Required Surfaces

- `shell`
- `apply_patch`
- local pytest execution
- deterministic file inspection

No network, no MCP, no subagents during solve. The family-local probe harness may use `codex exec` to calibrate the family.

## Workspace Bundle (per variant)

Every variant ships the following under `workspace_bundle/<variant_id>/`:

```text
AGENTS.md
Dockerfile
.scenario_variant
repo/
  runbooks/queue_drain.md
  scripts/queue_drain_helper.py
  .codex/automations/queue-drain-watch/automation.toml
  ops/notes/queue_drain_followup.md
  tests/test_queue_drain_helper.py
  tests/test_runbook_contract.py
retro/
  action_items.json
  incident_2026_02_14_retro.md
  operator_timeline.txt
artifacts/
  automation_expectations.json
  ops_note_schema.json
release_context/                # v4+ only
incident_context/               # v5 only
```

The four files above are the only intended edit targets. The grader treats every other path as immutable.

## Variant Ladder

### v1 — clean-baseline

The retro is mostly clean. The runbook, helper, and automation prompt all still use the retired verification command, and the follow-up note contains one stale escalation target. A solver that faithfully reads `retro/action_items.json` and updates all four surfaces should pass.

- Primary stress: multi-surface alignment.
- Honest failure mode: patch only the docs and forget the automation prompt or helper.

### v2 — noisy-distractor

V1 plus stale supporting prose in the retro markdown and timeline that still repeat the retired verification command and the old escalation room. The authoritative JSON is still correct.

- Primary stress: stale-context rejection.
- Honest failure mode: anchor on the prose retro instead of the JSON source of truth.

### v3 — dirty-state

V2 plus a half-updated follow-up note and helper alias comments that make it tempting to leave both retired and current commands as acceptable operator paths.

- Primary stress: close the loop cleanly instead of preserving dual-path ambiguity.
- Honest failure mode: leave both commands documented as acceptable.

### v4 — multi-corpus-objective

V3 plus `release_context/` that emphasizes quick page closure after queue depth reaches zero. The authoritative JSON still requires the post-drain verification command before closure.

- Primary stress: objective drift resistance.
- Honest failure mode: optimize for fast closure and skip or demote the verification step.

### v5 — recovery-in-thread

V4 plus `incident_context/` showing that a previous “fix” changed the automation cadence and destination rather than the stale prompt text, which caused the wake-up to miss the documented sequence.

- Primary stress: preserve automation schedule/destination while repairing the prompt content.
- Honest failure mode: change cadence or destination again instead of fixing the prompt text.

## Expected Deliverables

- repaired runbook instructions
- narrow helper-script repair
- repaired automation prompt with unchanged cadence and destination
- operator-facing follow-up note that includes only actionable retro items

## Grader Contract

### Phase 2 — visible checks

```bash
pytest -q repo/tests/test_queue_drain_helper.py repo/tests/test_runbook_contract.py
```

Visible checks validate:

- helper emits the authoritative verification command
- runbook includes the authoritative verification sequence
- runbook does not present the retired command as an acceptable operator step

### Hidden checks

Hidden checks are deterministic and code-backed. They validate:

- all three authoritative surfaces (`runbook`, `helper`, `automation prompt`) use the exact same non-retired verification command
- the automation prompt uses the correct escalation target
- cadence and destination remain pinned to `artifacts/automation_expectations.json`
- the follow-up note captures only actionable retro items and does not promote informational notes into repo changes
- no immutable slice was modified

### Trusted final-state checks

- only the four repair surfaces may change
- tests remain untouched
- retro and artifact files remain untouched
- no pytest shim, no network marker, no unexpected helper files

## 100-Point Breakdown

| Sub-metric | Points | Source |
| --- | ---: | --- |
| Visible pytest contract | 25 | repo tests |
| Helper repair correctness | 20 | deterministic hidden checks |
| Runbook + sequence correctness | 20 | deterministic hidden checks |
| Automation prompt fidelity | 20 | deterministic hidden checks |
| Follow-up note discipline | 10 | deterministic hidden checks |
| Constraint / invariants discipline | 5 | trusted final-state checks |

## Partial-Credit Ceilings

- `ungrounded_authority` ≤ 20 — solver follows supporting prose over `retro/action_items.json`
- `partial_surface_repair` ≤ 25 — fewer than all three authoritative surfaces are repaired
- `dual_command_path` ≤ 15 — retired and current commands remain documented as acceptable operator steps
- `doc_only_closure` ≤ 10 — docs or note updated but helper behavior remains stale
- `informational_note_promotion` ≤ 35 — informational-only retro notes are copied into the runbook, helper, or automation prompt as if they were required repo actions
- `objective_drift` ≤ 30 — v4+ demotes or skips post-drain verification to optimize for faster closure
- `schedule_drift` ≤ 25 — cadence or destination changed instead of repairing the prompt content

## Red-Team Traps

1. Patch only the runbook.
2. Patch the automation schedule rather than the inline prompt text.
3. Leave both retired and current commands in the runbook “for safety”.
4. Copy informational retro bullets into the follow-up note.
5. Modify tests or artifact expectations to make the workspace appear consistent.

## Quality Gate

- Visible tests must pass.
- Oracle repair must score at least 90.
- Empty workspace must score 0.
- The schedule-drift shortcut must score at most 25.
- Delete-tests adversarial run must raise integrity and zero higher milestones.

## Saturation And Renewal Plan

If the family mean `P_benchmark` exceeds `80` for two consecutive probe rounds at the default probe tier, flag `saturation_renewal_due` and refresh the family with one of:

1. a new variant where the helper repair and automation repair diverge in different directions
2. a retirement of the cleanest floor-check variant after the harder variants stabilize

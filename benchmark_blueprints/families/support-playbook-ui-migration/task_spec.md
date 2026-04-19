# Support Playbook UI Migration

## Task Identity
- `task_id`: `support-playbook-ui-migration/refund-escalation-tabs`
- `family_id`: `support-playbook-ui-migration`
- `scenario_type`: `migration_refactor`

## Task Prompt
The support console was redesigned from a single-pane layout to a tabbed customer workspace. The refund-escalation workflow in the helper code and `SKILL.md` still follows the retired UI, so operators and Codex both fail in the live console. Update the implementation and support guidance so the refund escalation flow works in the new UI and produces reviewer-ready evidence.

## Workspace Bundle
- `support_console/`: frontend app with tabbed account workspace, duplicated action labels, and banner states.
- `helpers/console_flow.py`: helper module or automation adapter that still targets the legacy selectors.
- `skills/refund-escalation/SKILL.md`: instructions describing the old left-nav workflow.
- `fixtures/accounts.json`: seeded accounts including one account where the new flow must open a fraud tab before escalation.
- `review/threads/refund-escalation.md`: mock PR or review thread requesting proof that the workflow was migrated correctly.
- `artifacts/screenshots/`: expected destination for browser captures.

## Seeded GUI Or Browser-Only Breakage
- Legacy selectors still find elements in a hidden template, so shallow DOM checks pass while the real tabbed UI flow fails.
- The redesigned console contains duplicate "Escalate" labels in different tabs; only one writes the expected escalation record.
- A conditional banner blocks the real fraud escalation path unless the browser sequence follows the new tab order.

## Required Surfaces
- `computer_use`
- `browser_preview`
- `shell`
- `apply_patch`
- `skills`
- `screenshots`

## Expected Deliverables
- Code patch in helper or adapter logic.
- Updated `skills/refund-escalation/SKILL.md`.
- Screenshot bundle of the live migrated flow.
- Short review reply or note explaining what changed in the new UI path.

## Grader Contract
- Start the console with seeded support data.
- Run hidden checks that the helper code no longer targets retired selectors or pane ids.
- Run a hidden browser scenario that:
  - opens the specific seeded customer account that requires the fraud-tab branch,
  - navigates through the new tab set,
  - handles the fraud banner,
  - triggers the correct escalation control,
  - confirms a new escalation record with the expected type and timestamp delta appears in the visible activity timeline.
- Trusted post-run checks confirm:
  - `SKILL.md` documents the new tab names and the banner branch,
  - the review note mentions the duplicate-label hazard,
  - screenshot files exist and show the updated workspace, not the hidden legacy template.
- Hidden scoring rejects fixes that only update docs or selector strings without exercising the real migrated path.

## Red-Team Traps To Defend Against
- Point the helper to a hidden legacy DOM subtree that still exists in the bundle.
- Update `SKILL.md` only and skip the live helper path.
- Click the wrong duplicate "Escalate" control that looks correct but does not create a record.
- Fake screenshots from the old console or blank placeholders.
- Patch tests to accept the legacy pane ids as aliases.

## Quality Gate
- Initial weakness to watch: a single-selector rename would be too easy and could let a naive GPT-5.4/high solver clear 30 points.
- Required hardening:
  - hidden duplicate-label browser path,
  - required fraud-tab seeded account path,
  - visible activity-log persistence check with expected type and timestamp delta,
  - retired-selector denial check,
  - skill-doc requirement for the banner branch.
- Actual GPT-5.4/high scored run: `20/100`. The child agent understood the tab migration, hidden-template trap, and fraud-branch account requirement, but earned no execution or persistence points because it could not modify the helper or produce a timeline record.
- Naive solver hardness verdict: `under_30 = yes`, but only narrowly.

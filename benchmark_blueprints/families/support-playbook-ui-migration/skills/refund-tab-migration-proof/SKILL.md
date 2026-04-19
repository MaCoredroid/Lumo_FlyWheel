# Refund Tab Migration Proof

Use this skill when solving `support-playbook-ui-migration`.

## Objective
Migrate the refund-escalation workflow from the retired single-pane console to the new tabbed workspace without falling for hidden-template or duplicate-label traps.

## Required Approach
1. Exercise the seeded fraud-branch account, not a generic happy-path account.
2. Ignore hidden legacy templates and retired pane ids.
3. Traverse the new tab order and handle the banner branch explicitly.
4. Confirm the visible activity timeline receives a new escalation record with the expected type.
5. Update the skill guidance only after the executable path is correct.

## Do Not
- Rename one selector and stop.
- Click the first visible `Escalate` control without validating record creation.
- Treat screenshots alone as proof of migration.
- Preserve compatibility with the retired pane ids just to satisfy tests.

## Completion Standard
The task is solved only if the helper and the live browser path both reach the correct control on the seeded fraud-branch account and create a real escalation record.
